from typing import List, Optional, Dict, Any
from datetime import datetime
import re
import json

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body
from pydantic import BaseModel
from sqlmodel import SQLModel, Field, create_engine, Session, select, col

# Resume parsing deps
import pdfplumber
import docx2txt

# Semantic search / matching
from sentence_transformers import SentenceTransformer, util

# ---------- Models ----------

class Job(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    description: str
    required_keywords: str = ""      # "python, sql, airflow"
    nice_to_have_keywords: str = ""  # "aws, docker"
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Candidate(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    email: str
    phone: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Application(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: int
    candidate_id: int
    resume_text: str
    stage: str = "NEW"  # NEW -> SCREEN -> INTERVIEW -> OFFER -> HIRED/REJECTED
    score: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)

class StageEvent(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    application_id: int
    from_stage: Optional[str] = None
    to_stage: str
    note: Optional[str] = None
    at: datetime = Field(default_factory=datetime.utcnow)

class AppNote(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    application_id: int
    author: str = "recruiter"
    text: str
    at: datetime = Field(default_factory=datetime.utcnow)

# ---------- Schemas ----------

class JobCreate(BaseModel):
    title: str
    description: str
    required_keywords: str = ""
    nice_to_have_keywords: str = ""

class JobOut(JobCreate):
    id: int
    created_at: datetime

class CandidateCreate(BaseModel):
    name: str
    email: str
    phone: Optional[str] = None

class CandidateOut(CandidateCreate):
    id: int
    created_at: datetime

class ApplicationCreate(BaseModel):
    job_id: int
    candidate_id: int
    resume_text: str

class ApplicationOut(BaseModel):
    id: int
    job_id: int
    candidate_id: int
    stage: str
    score: float
    created_at: datetime

class RankResult(BaseModel):
    application_id: int
    candidate_id: int
    candidate_name: str
    score: float
    stage: str
    why: Dict[str, Any]

class ParseOut(BaseModel):
    sections: Dict[str, str]

class DashboardMetrics(BaseModel):
    by_stage: Dict[str, int]
    score_bins: Dict[str, int]

# ---------- App init ----------

engine = create_engine("sqlite:///./ats.db")
SQLModel.metadata.create_all(engine)

app = FastAPI(title="Better Mini ATS", version="0.3")

# Semantic model (small, fast)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ---------- Utilities ----------

SECTION_HEADINGS = [
    "experience", "work experience", "professional experience",
    "education", "skills", "projects", "certifications", "summary", "profile"
]

def parse_resume_simple(text: str) -> Dict[str, str]:
    lines = [l.strip() for l in text.splitlines()]
    sections: Dict[str, List[str]] = {}
    current = "summary"
    def norm(h: str) -> str:
        return re.sub(r"\s+", " ", h.lower()).strip()
    for line in lines:
        candidate = norm(line.strip(":").lower())
        if candidate in SECTION_HEADINGS:
            current = candidate
            sections.setdefault(current, [])
        else:
            sections.setdefault(current, [])
            sections[current].append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items()}

def extract_text_from_pdf(path: str) -> str:
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            text += t + "\n"
    return text.strip()

def extract_text_from_docx(path: str) -> str:
    return docx2txt.process(path).strip()

def kw_list(s: str) -> List[str]:
    return [k.strip().lower() for k in s.split(",") if k.strip()]

def normalize_tokens(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9+.#/ ]+", " ", text)
    tokens = [t.strip() for t in text.split() if t.strip()]
    return tokens

def score_keywords(resume_text: str, required_kw: List[str], nice_kw: List[str]) -> Dict[str, Any]:
    tokens = normalize_tokens(resume_text)
    token_set = set(tokens)
    def hitset(keywords: List[str]) -> List[str]:
        hits = []
        hay = " ".join(tokens)
        for kw in keywords:
            if not kw:
                continue
            if " " in kw:
                if kw in hay:
                    hits.append(kw)
            elif kw in token_set:
                hits.append(kw)
        return hits

    req_hits = hitset(required_kw)
    nice_hits = hitset(nice_kw)

    req_score = 3 * len(req_hits)
    nice_score = 1 * len(nice_hits)
    bonus = 2 if required_kw and len(req_hits) == len([k for k in required_kw if k]) else 0
    total = req_score + nice_score + bonus

    return {
        "score": float(total),
        "matched_required": req_hits,
        "matched_nice": nice_hits,
        "missing_required": [k for k in required_kw if k and k not in req_hits],
        "bonus_all_required": bool(bonus),
    }

def semantic_similarity(a: str, b: str) -> float:
    ea = embedder.encode(a, convert_to_tensor=True)
    eb = embedder.encode(b, convert_to_tensor=True)
    return float(util.cos_sim(ea, eb).item())

def blended_score(kw: float, sem: float) -> float:
    # sem is 0..1, scale to 0..100
    return 0.6 * kw + 0.4 * (sem * 100.0)

STAGES = ["NEW", "SCREEN", "INTERVIEW", "OFFER", "HIRED", "REJECTED"]

# ---------- Endpoints: Jobs & Candidates ----------

@app.post("/jobs", response_model=JobOut)
def create_job(payload: JobCreate):
    with Session(engine) as session:
        job = Job(**payload.dict())
        session.add(job)
        session.commit()
        session.refresh(job)
        return JobOut(**job.dict())

@app.get("/jobs", response_model=List[JobOut])
def list_jobs():
    with Session(engine) as session:
        jobs = session.exec(select(Job).order_by(Job.created_at.desc())).all()
        return [JobOut(**j.dict()) for j in jobs]

@app.post("/candidates", response_model=CandidateOut)
def create_candidate(payload: CandidateCreate):
    with Session(engine) as session:
        cand = Candidate(**payload.dict())
        session.add(cand)
        session.commit()
        session.refresh(cand)
        return CandidateOut(**cand.dict())

@app.get("/candidates", response_model=List[CandidateOut])
def list_candidates():
    with Session(engine) as session:
        cands = session.exec(select(Candidate).order_by(Candidate.created_at.desc())).all()
        return [CandidateOut(**c.dict()) for c in cands]

# ---------- Endpoints: Applications & Upload ----------

@app.post("/applications", response_model=ApplicationOut)
def create_application(payload: ApplicationCreate):
    with Session(engine) as session:
        job = session.get(Job, payload.job_id)
        cand = session.get(Candidate, payload.candidate_id)
        if not job or not cand:
            raise HTTPException(404, "Job or Candidate not found")

        kw = score_keywords(payload.resume_text, kw_list(job.required_keywords), kw_list(job.nice_to_have_keywords))
        sem = semantic_similarity(payload.resume_text, job.description)
        final = blended_score(kw["score"], sem)

        appn = Application(job_id=job.id, candidate_id=cand.id, resume_text=payload.resume_text, score=final)
        session.add(appn)
        session.commit()
        session.refresh(appn)

        # initial stage event
        session.add(StageEvent(application_id=appn.id, from_stage=None, to_stage="NEW", note="created"))
        session.commit()

        return ApplicationOut(id=appn.id, job_id=appn.job_id, candidate_id=appn.candidate_id,
                              stage=appn.stage, score=appn.score, created_at=appn.created_at)

@app.post("/upload_resume")
async def upload_resume(job_id: int, candidate_id: int, file: UploadFile = File(...)):
    ext = (file.filename or "").split(".")[-1].lower()
    tmp_path = f"/tmp/{datetime.utcnow().timestamp()}_{file.filename}"
    with open(tmp_path, "wb") as f:
        f.write(await file.read())

    if ext == "pdf":
        text = extract_text_from_pdf(tmp_path)
    elif ext in ["docx", "doc"]:
        text = extract_text_from_docx(tmp_path)
    else:
        raise HTTPException(400, "Only PDF/DOCX supported")

    with Session(engine) as session:
        job = session.get(Job, job_id)
        cand = session.get(Candidate, candidate_id)
        if not job or not cand:
            raise HTTPException(404, "Job or Candidate not found")

        kw = score_keywords(text, kw_list(job.required_keywords), kw_list(job.nice_to_have_keywords))
        sem = semantic_similarity(text, job.description)
        final = blended_score(kw["score"], sem)

        appn = Application(job_id=job.id, candidate_id=cand.id, resume_text=text, score=final)
        session.add(appn)
        session.commit()
        session.refresh(appn)

        session.add(StageEvent(application_id=appn.id, from_stage=None, to_stage="NEW", note=f"uploaded {file.filename}"))
        session.commit()

        return {"application_id": appn.id, "score": final, "why": {"keywords": kw, "semantic": sem}}

@app.get("/applications", response_model=List[ApplicationOut])
def list_applications(stage: Optional[str] = None, job_id: Optional[int] = None):
    with Session(engine) as session:
        stmt = select(Application)
        if stage:
            stmt = stmt.where(Application.stage == stage.upper())
        if job_id:
            stmt = stmt.where(Application.job_id == job_id)
        apps = session.exec(stmt.order_by(Application.created_at.desc())).all()
        return [ApplicationOut(id=a.id, job_id=a.job_id, candidate_id=a.candidate_id,
                               stage=a.stage, score=a.score, created_at=a.created_at) for a in apps]

@app.post("/applications/{application_id}/advance", response_model=ApplicationOut)
def advance_stage(application_id: int, to_stage: str = Form(...), note: Optional[str] = Form(None)):
    to_stage = to_stage.upper()
    if to_stage not in STAGES:
        raise HTTPException(400, f"Invalid stage. Use one of {STAGES}")
    with Session(engine) as session:
        appn = session.get(Application, application_id)
        if not appn:
            raise HTTPException(404, "Application not found")
        old = appn.stage
        appn.stage = to_stage
        session.add(appn)
        session.commit()
        session.refresh(appn)
        session.add(StageEvent(application_id=appn.id, from_stage=old, to_stage=to_stage, note=note))
        session.commit()
        return ApplicationOut(id=appn.id, job_id=appn.job_id, candidate_id=appn.candidate_id,
                              stage=appn.stage, score=appn.score, created_at=appn.created_at)

@app.post("/applications/{application_id}/notes")
def add_note(application_id: int, author: str = Form("recruiter"), text: str = Form(...)):
    with Session(engine) as session:
        appn = session.get(Application, application_id)
        if not appn:
            raise HTTPException(404, "Application not found")
        n = AppNote(application_id=application_id, author=author, text=text)
        session.add(n)
        session.commit()
        return {"ok": True, "note_id": n.id}

@app.get("/applications/{application_id}/timeline")
def get_timeline(application_id: int):
    with Session(engine) as session:
        events = session.exec(select(StageEvent).where(StageEvent.application_id == application_id).order_by(StageEvent.at)).all()
        notes = session.exec(select(AppNote).where(AppNote.application_id == application_id).order_by(AppNote.at)).all()
        return {
            "events": [e.dict() for e in events],
            "notes": [n.dict() for n in notes],
        }

@app.get("/applications/{application_id}/parsed", response_model=ParseOut)
def parse_existing(application_id: int):
    with Session(engine) as session:
        appn = session.get(Application, application_id)
        if not appn:
            raise HTTPException(404, "Application not found")
        return ParseOut(sections=parse_resume_simple(appn.resume_text))

# ---------- Rankings & Search ----------

@app.get("/rankings", response_model=List[RankResult])
def rankings(job_id: int):
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        apps = session.exec(select(Application).where(Application.job_id == job_id)).all()
        results = []
        for a in apps:
            cand = session.get(Candidate, a.candidate_id)
            kw = score_keywords(a.resume_text, kw_list(job.required_keywords), kw_list(job.nice_to_have_keywords))
            sem = semantic_similarity(a.resume_text, job.description)
            results.append(RankResult(
                application_id=a.id,
                candidate_id=cand.id,
                candidate_name=cand.name,
                score=blended_score(kw["score"], sem),
                stage=a.stage,
                why={"keywords": kw, "semantic": sem}
            ))
        # sort by score desc; NEW first among equals
        order = {"NEW": 0, "SCREEN": 1, "INTERVIEW": 2, "OFFER": 3, "HIRED": 4, "REJECTED": 5}
        results.sort(key=lambda r: (-r.score, order.get(r.stage, 99)))
        return results

@app.get("/search_candidates")
def search_candidates(q: str):
    """
    Semantic + keyword search across ALL applications (latest per candidate per job not enforced here).
    Returns top 50 matches by blended score with the query as the 'job description' comparator.
    """
    with Session(engine) as session:
        apps = session.exec(select(Application)).all()
        out = []
        for a in apps:
            cand = session.get(Candidate, a.candidate_id)
            kw = score_keywords(a.resume_text, [], kw_list(q))  # treat query comma-separated as soft keywords
            sem = semantic_similarity(a.resume_text, q)
            out.append({
                "application_id": a.id,
                "candidate_id": cand.id,
                "candidate_name": cand.name,
                "job_id": a.job_id,
                "stage": a.stage,
                "score": blended_score(kw["score"], sem),
                "why": {"keywords": kw, "semantic": sem}
            })
        out.sort(key=lambda r: -r["score"])
        return out[:50]

# ---------- Parsing raw files endpoint ----------

@app.post("/parse_resume", response_model=ParseOut)
async def parse_resume(file: UploadFile = File(...)):
    text = (await file.read()).decode(errors="ignore")
    return ParseOut(sections=parse_resume_simple(text))

# ---------- Dashboard metrics ----------

@app.get("/dashboard_metrics", response_model=DashboardMetrics)
def dashboard_metrics():
    with Session(engine) as session:
        apps = session.exec(select(Application)).all()
        # by stage
        by_stage: Dict[str, int] = {}
        for a in apps:
            by_stage[a.stage] = by_stage.get(a.stage, 0) + 1
        # score bins (0-20,20-40,40-60,60-80,80-100+)
        bins = {"0-20":0,"20-40":0,"40-60":0,"60-80":0,"80-100":0,"100+":0}
        for a in apps:
            s = a.score
            if s < 20: bins["0-20"] += 1
            elif s < 40: bins["20-40"] += 1
            elif s < 60: bins["40-60"] += 1
            elif s < 80: bins["60-80"] += 1
            elif s <= 100: bins["80-100"] += 1
            else: bins["100+"] += 1
        return DashboardMetrics(by_stage=by_stage, score_bins=bins)