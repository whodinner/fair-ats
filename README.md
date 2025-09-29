## Fair ATS (Application Tracking System)

- Resume upload (PDF/DOCX) → parsing + scoring

- Keyword + semantic matching (embeddings) with why-match breakdown

- Ranked pipeline with filters

- Semantic+keyword search across all applicants

- Bulk actions (bulk reject by score threshold)

- Dashboard (stage counts + score distribution)

- Candidate Profile (timeline, notes, parsed resume)

## How to run

1. Create a virtual env & install deps:

python -m venv venv
    source venv/bin/activate   # (Windows: venv\Scripts\activate)
    pip install -r requirements.txt

2. Start the backend
   uvicorn app:app --reload

3. Start UI in another terminal
   streamlit run ui.py

4. In the Streamlit app:

- Pick a job (create jobs via the FastAPI /jobs endpoint if needed).

- Select a candidate (create via /candidates).

- Drag-and-drop a PDF/DOCX to upload; it parses, scores, and explains the match.

- Use tabs for Pipeline, Search, Bulk Actions, Dashboard, Candidate Profile.

