"""
Unfaced – Identity-Consent Firewall  (MVP)
Built with Streamlit, DeepFace & Resemblyzer
"""

import streamlit as st
import json, os, uuid, shutil
from pathlib import Path
from deepface import DeepFace
from resemblyzer import VoiceEncoder, preprocess_wav
import soundfile as sf
import numpy as np

# ────────────────────────────────  CONFIG
DATA_DIR      = Path("data")
UPLOADS_DIR   = DATA_DIR / "uploads"
PENDING_DIR   = UPLOADS_DIR / "pending"
APPROVED_DIR  = UPLOADS_DIR / "approved"
REJECTED_DIR  = UPLOADS_DIR / "rejected"
USERS_JSON    = DATA_DIR / "users.json"
SUB_JSON      = DATA_DIR / "submissions.json"

for d in [PENDING_DIR, APPROVED_DIR, REJECTED_DIR]:
    d.mkdir(parents=True, exist_ok=True)
for f in [USERS_JSON, SUB_JSON]:
    if not f.exists(): f.write_text("[]")

encoder = VoiceEncoder()

# ────────────────────────────────  HELPERS
def read_json(path):   return json.loads(path.read_text())
def write_json(path,d): path.write_text(json.dumps(d, indent=2))

def face_embed(img_path):
    return DeepFace.represent(img_path=img_path, model_name="Facenet")[0]["embedding"]

def voice_embed(wav_path):
    wav, sr = sf.read(wav_path)
    if sr != 16000: wav_path = str(wav_path)  # preprocess_wav needs path str
    wav = preprocess_wav(wav_path)
    return encoder.embed_utterance(wav).tolist()

def cos_sim(a,b): return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

# ────────────────────────────────  UI
st.set_page_config(page_title="Unfaced", layout="centered")
st.title("🔒 Unfaced – Protect Your Face & Voice")

menu = st.sidebar.radio("Navigate", ["Register Identity","Upload Media","My Approvals","Gallery"])

# ── 1. REGISTER
if menu=="Register Identity":
    st.header("Register your identity")
    name  = st.text_input("Name")
    face  = st.file_uploader("Selfie (.jpg/.png)", ["jpg","png"])
    voice = st.file_uploader("Voice clip (.wav/.mp3)", ["wav","mp3"])
    if st.button("Register") and name and face and voice:
        uid = str(uuid.uuid4())
        tmp_face = PENDING_DIR/(uid+"_face.jpg");   tmp_face.write_bytes(face.read())
        tmp_voice= PENDING_DIR/(uid+"_voice.wav");  tmp_voice.write_bytes(voice.read())
        try:
            user = {
                "id": uid,
                "name": name,
                "face": face_embed(tmp_face),
                "voice": voice_embed(tmp_voice)
            }
            users = read_json(USERS_JSON); users.append(user); write_json(USERS_JSON, users)
            tmp_face.unlink(); tmp_voice.unlink()
            st.success(f"{name} registered 🎉")
        except Exception as e:
            st.error(f"Failed: {e}")

# ── 2. UPLOAD
if menu=="Upload Media":
    st.header("Upload content")
    uploader = st.text_input("Your Name (uploader)")
    media    = st.file_uploader("Image or video", ["jpg","png","mp4","mov"])
    if st.button("Scan & Submit") and uploader and media:
        fid = str(uuid.uuid4())
        tgt = PENDING_DIR / f"{fid}_{media.name}"
        tgt.write_bytes(media.read())
        st.info("Scanning for registered identities…")

        # very simple: only look at first frame if image
        matched = []
        users = read_json(USERS_JSON)
        try:
            emb = face_embed(tgt)  # will fail on videos, you can extend later
            for u in users:
                if cos_sim(emb, u["face"]) > 0.4:
                    matched.append(u["name"])
        except: pass

        submissions = read_json(SUB_JSON)
        submissions.append({
            "id": fid,
            "file": str(tgt),
            "uploader": uploader,
            "matches": matched,
            "status": "pending" if matched else "approved"
        })
        write_json(SUB_JSON, submissions)

        if matched:
            st.warning(f"⚠️ Needs approval from: {', '.join(matched)}")
        else:
            shutil.move(tgt, APPROVED_DIR/tgt.name)
            st.success("No matches – media auto-approved ✅")

# ── 3. APPROVAL DASHBOARD
if menu=="My Approvals":
    st.header("Approve or reject media")
    me = st.selectbox("I am…", [u["name"] for u in read_json(USERS_JSON)])
    subs = read_json(SUB_JSON)
    for s in subs:
        if me in s["matches"] and s["status"]=="pending":
            st.image(s["file"])
            col1,col2 = st.columns(2)
            if col1.button("Approve", key=s["id"]+"_a"):
                s["status"]="approved"; write_json(SUB_JSON,subs)
                shutil.move(s["file"], APPROVED_DIR/Path(s["file"]).name)
                st.success("Approved")
            if col2.button("Reject", key=s["id"]+"_r"):
                s["status"]="rejected"; write_json(SUB_JSON,subs)
                shutil.move(s["file"], REJECTED_DIR/Path(s["file"]).name)
                st.error("Rejected")

# ── 4. PUBLIC GALLERY
if menu=="Gallery":
    st.header("Approved media gallery")
    for s in read_json(SUB_JSON):
        if s["status"]=="approved":
            st.image(s["file"], caption=f'Uploaded by {s["uploader"]}')
