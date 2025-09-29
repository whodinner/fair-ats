import streamlit as st
import requests
import pandas as pd
import plotly.express as px

BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="ATS Supreme", layout="wide")
st.title("ATS Supreme")

# Sidebar: job & candidate selectors
jobs = requests.get(f"{BASE}/jobs").json()
if not jobs:
    st.warning("No jobs found. Use the API to create jobs, then refresh.")
    st.stop()

job_map = {j["title"]: j["id"] for j in jobs}
job_name = st.sidebar.selectbox("Select a Job", list(job_map.keys()))
job_id = job_map[job_name]

cands = requests.get(f"{BASE}/candidates").json()
cand_map = {c["name"]: c["id"] for c in cands} if cands else {}
cand_name = st.sidebar.selectbox("Candidate", list(cand_map.keys()) if cand_map else ["(none)"])

uploaded = st.sidebar.file_uploader("📤 Upload PDF/DOCX", type=["pdf", "docx"])
if uploaded and st.sidebar.button("Upload & Score"):
    files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
    resp = requests.post(
        f"{BASE}/upload_resume",
        params={"job_id": job_id, "candidate_id": cand_map[cand_name]},
        files=files
    )
    if resp.ok:
        data = resp.json()
        st.sidebar.success(f"Uploaded. Score={data['score']:.2f}")
        # Parse preview
        parse_resp = requests.post(f"{BASE}/parse_resume", files=files)
        if parse_resp.ok:
            sections = parse_resp.json().get("sections", {})
            st.subheader(f"📄 Parsed Resume: {cand_name}")
            cols = st.columns(2)
            for i,(sec, txt) in enumerate(sections.items()):
                with cols[i%2]:
                    with st.expander(sec.capitalize()):
                        st.text(txt if txt else "(empty)")
    else:
        st.sidebar.error(resp.text)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Pipeline", "Search", "Bulk Actions", "Dashboard", "Candidate Profile"])

# -------- Pipeline tab --------
with tab1:
    st.subheader(f"Ranked Candidates for {job_name}")
    ranks = requests.get(f"{BASE}/rankings", params={"job_id": job_id}).json()
    df = pd.DataFrame(ranks)
    if df.empty:
        st.info("No applications yet.")
    else:
        # Filters
        min_score = st.slider("Minimum Score", 0, 120, 0)
        stage_filter = st.multiselect("Stage filter", sorted(df["stage"].unique()), default=sorted(df["stage"].unique()))
        filtered = df[(df["score"] >= min_score) & (df["stage"].isin(stage_filter))]
        st.dataframe(filtered[["application_id","candidate_name","score","stage"]], use_container_width=True)

        for _, row in filtered.iterrows():
            with st.expander(f"{row['candidate_name']} — Score {row['score']:.1f} — Stage {row['stage']}"):
                why = row["why"]
                kw = why.get("keywords", {})
                sem = why.get("semantic", 0.0)
                colA, colB, colC = st.columns(3)
                with colA:
                    st.markdown("**Matched required**")
                    st.write(", ".join(kw.get("matched_required", [])) or "—")
                with colB:
                    st.markdown("**Matched nice-to-have**")
                    st.write(", ".join(kw.get("matched_nice", [])) or "—")
                with colC:
                    st.markdown("**Missing required**")
                    st.write(", ".join(kw.get("missing_required", [])) or "—")
                st.markdown(f"**Semantic similarity:** {sem:.2f}")
                if st.button("Advance → INTERVIEW", key=f"adv-{row['application_id']}"):
                    requests.post(f"{BASE}/applications/{row['application_id']}/advance", data={"to_stage": "INTERVIEW", "note": "via UI"})
                    st.success("Advanced to INTERVIEW. Refresh to see changes.")

# -------- Search tab --------
with tab2:
    st.subheader("Semantic + Keyword Search")
    q = st.text_input("Describe the candidate you're looking for (e.g., 'streaming, PySpark, AWS, data pipelines')")
    if st.button("Search"):
        res = requests.get(f"{BASE}/search_candidates", params={"q": q}).json()
        sdf = pd.DataFrame(res)
        if sdf.empty:
            st.info("No matches.")
        else:
            st.dataframe(sdf[["application_id","candidate_name","job_id","score","stage"]], use_container_width=True)

# -------- Bulk Actions tab --------
with tab3:
    st.subheader("Bulk Actions")
    ranks = requests.get(f"{BASE}/rankings", params={"job_id": job_id}).json()
    df = pd.DataFrame(ranks)
    if df.empty:
        st.info("No applications for bulk actions.")
    else:
        thresh = st.slider("Reject all with score below", 0, 120, 40)
        if st.button("Bulk Reject"):
            count = 0
            for _, row in df.iterrows():
                if row["score"] < thresh and row["stage"] not in ("REJECTED","HIRED"):
                    requests.post(f"{BASE}/applications/{row['application_id']}/advance", data={"to_stage": "REJECTED", "note": f"bulk reject < {thresh}"})
                    count += 1
            st.success(f"Rejected {count} candidates below {thresh}. Refresh to update.")

# -------- Dashboard tab --------
with tab4:
    st.subheader("Analytics")
    metrics = requests.get(f"{BASE}/dashboard_metrics").json()
    by_stage = pd.DataFrame(list(metrics["by_stage"].items()), columns=["stage","count"])
    bins = pd.DataFrame(list(metrics["score_bins"].items()), columns=["bin","count"])
    if not by_stage.empty:
        fig1 = px.bar(by_stage, x="stage", y="count", title="Pipeline Stage Counts", text="count")
        st.plotly_chart(fig1, use_container_width=True)
    if not bins.empty:
        fig2 = px.bar(bins, x="bin", y="count", title="Score Distribution (binned)", text="count")
        st.plotly_chart(fig2, use_container_width=True)

# -------- Candidate Profile tab --------
with tab5:
    st.subheader("Candidate Profile")
    app_id = st.number_input("Application ID", min_value=1, value=1, step=1)
    if st.button("Load Profile"):
        # Timeline
        timeline = requests.get(f"{BASE}/applications/{app_id}/timeline")
        if timeline.ok:
            t = timeline.json()
            st.markdown("**Timeline**")
            for e in t["events"]:
                st.write(f"{e['at']}: {e['from_stage']} → {e['to_stage']}  ({e.get('note') or ''})")
            st.markdown("**Notes**")
            for n in t["notes"]:
                st.write(f"{n['at']} — {n['author']}: {n['text']}")
        else:
            st.error("Timeline not found.")

        # Parsed resume
        parsed = requests.get(f"{BASE}/applications/{app_id}/parsed")
        if parsed.ok:
            sections = parsed.json().get("sections", {})
            st.markdown("**Parsed Resume**")
            for sec, txt in sections.items():
                with st.expander(sec.capitalize()):
                    st.text(txt if txt else "(empty)")
        else:
            st.error("No parsed resume.")

        # Add a note
        st.markdown("**Add Note**")
        note = st.text_area("Note text")
        if st.button("Save Note"):
            resp = requests.post(f"{BASE}/applications/{app_id}/notes", data={"text": note, "author": "recruiter"})
            if resp.ok:
                st.success("Note saved.")
            else:
                st.error(resp.text)