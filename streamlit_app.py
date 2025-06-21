# Unfaced – Identity-Consent Firewall (MVP)
# Built with Streamlit, DeepFace & Resemblyzer

import streamlit as st
import json, os, uuid, shutil
from pathlib import Path
from deepface import DeepFace
from resemblyzer import VoiceEncoder, preprocess_wav
import soundfile as sf
import numpy as np
import re

# ─── CONFIGURATION
DATA_DIR      = Path("data")
UPLOADS_DIR   = DATA_DIR / "uploads"
PENDING_DIR   = UPLOADS_DIR / "pending"
APPROVED_DIR  = UPLOADS_DIR / "approved"
REJECTED_DIR  = UPLOADS_DIR / "rejected"
USERS_JSON    = DATA_DIR / "users.json"
SUB_JSON      = DATA_DIR / "submissions.json"

# Create necessary folders/files
for d in [PENDING_DIR, APPROVED_DIR, REJECTED_DIR]:
    d.mkdir(parents=True, exist_ok=True)
for f in [USERS_JSON, SUB_JSON]:
    if not f.exists():
        f.write_text("[]")

encoder = VoiceEncoder()

# ─── SESSION INIT
if "auth_user" not in st.session_state:
    st.session_state.auth_user = None

# ─── UTILITY FUNCTIONS
def read_json(path):
    return json.loads(path.read_text())

def write_json(path, data):
    path.write_text(json.dumps(data, indent=2))

def face_embed(img_path):
    return DeepFace.represent(img_path=img_path, model_name="Facenet")[0]["embedding"]

def voice_embed(wav_path):
    wav, sr = sf.read(wav_path)
    if sr != 16000:
        wav_path = str(wav_path)
    wav = preprocess_wav(wav_path)
    return encoder.embed_utterance(wav).tolist()

def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_user(name, phone):
    users = read_json(USERS_JSON)
    for u in users:
        if u["name"] == name and u.get("phone") == phone:
            return u
    return None

def safe_filename(filename):
    return re.sub(r'[^\w\-\_\. ]', '_', filename).replace(" ", "_").lower()

def display_media(file_path):
    if os.path.exists(file_path):
        if file_path.lower().endswith((".mp4", ".mov", ".m4v", ".hevc")):
            st.video(file_path)
        elif file_path.lower().endswith((".mp3", ".wav", ".aac", ".m4a")):
            st.audio(file_path)
        elif file_path.lower().endswith((".jpg", ".jpeg", ".png")):
            st.image(file_path)

def generate_share_links(file_path, uploader):
    share_url = f"https://your-app-domain.com/{file_path}"  # Replace with deployed domain
    message = f"Check out this media uploaded by {uploader} on Unfaced 🔒"
    return {
        "Facebook": f"https://www.facebook.com/sharer/sharer.php?u={share_url}",
        "LinkedIn": f"https://www.linkedin.com/sharing/share-offsite/?url={share_url}",
        "Twitter": f"https://twitter.com/intent/tweet?text={message}&url={share_url}",
        "Copy Link": share_url
    }

# ─── UI CONFIG
st.set_page_config(page_title="Unfaced", layout="centered")
st.title("🔒 Unfaced – Protect Your Face & Voice")

# ─── AUTHENTICATION
if not st.session_state.auth_user:
    st.subheader("Login or Register")
    auth_mode = st.radio("Choose", ["Login", "Register"])

    if auth_mode == "Login":
        name = st.text_input("Name")
        phone = st.text_input("Phone")
        if st.button("Login"):
            user = find_user(name, phone)
            if user:
                st.session_state.auth_user = user
                st.success("Logged in successfully ✅")
                st.experimental_rerun()
            else:
                st.warning("User not found. Please register.")

    else:  # REGISTER
        st.subheader("Register your identity")
        name = st.text_input("Name")
        phone = st.text_input("Phone")
        st.markdown(":camera: On mobile? Use the camera to take a real-time selfie")
        face = st.file_uploader("Upload or take a selfie", type=["jpg", "png"])
        st.markdown(":microphone: Record a short voice message")
        voice = st.file_uploader("Voice clip (optional)", type=["wav", "mp3", "m4a", "aac"])

        if face:
            st.image(face, caption="Selfie preview")
        if voice:
            st.audio(voice)

        if st.button("Register") and name and phone and face:
            if find_user(name, phone):
                st.warning("User already exists. Please login.")
            else:
                uid = str(uuid.uuid4())
                tmp_face = PENDING_DIR / f"{uid}_face.jpg"
                tmp_face.write_bytes(face.read())

                user = {
                    "id": uid,
                    "name": name,
                    "phone": phone,
                    "face": face_embed(tmp_face)
                }

                if voice:
                    tmp_voice = PENDING_DIR / f"{uid}_voice.wav"
                    tmp_voice.write_bytes(voice.read())
                    user["voice"] = voice_embed(tmp_voice)
                    tmp_voice.unlink()

                users = read_json(USERS_JSON)
                users.append(user)
                write_json(USERS_JSON, users)

                tmp_face.unlink()

                st.session_state.auth_user = user
                st.success("Registered and logged in ✅")
                st.experimental_rerun()

else:
    user = st.session_state.auth_user
    st.sidebar.write(f"👤 Logged in as: {user['name']}")
    page = st.sidebar.radio("Navigate", ["Dashboard", "My Uploads", "Submit Media", "Requests", "Rejected", "Gallery", "Logout"])

    if page == "Dashboard":
        st.header(f"Welcome, {user['name']}!")
        st.subheader("Account Info:")
        st.write(f"📱 Phone: {user.get('phone', 'N/A')}")
        st.write(f"🆔 User ID: {user['id']}")
        st.write(f"🎤 Voice Registered: {'Yes' if user.get('voice') else 'No'}")

        all_subs = read_json(SUB_JSON)
        total = sum(1 for s in all_subs if s["uploader"] == user["name"])
        approved = sum(1 for s in all_subs if s["uploader"] == user["name"] and s["status"] == "approved")
        pending = sum(1 for s in all_subs if s["uploader"] == user["name"] and s["status"] == "pending")
        rejected = sum(1 for s in all_subs if s["uploader"] == user["name"] and s["status"] == "rejected")

        st.subheader("📊 Upload Stats")
        st.write(f"✅ Approved: {approved}")
        st.write(f"⏳ Pending: {pending}")
        st.write(f"❌ Rejected: {rejected}")
        st.write(f"📁 Total: {total}")

    if page == "Submit Media":
        st.subheader("Upload content")
        media = st.file_uploader("Image, audio, or video", type=["jpg", "png", "mp4", "mov", "m4v", "hevc", "wav", "mp3", "aac", "m4a"])
        if st.button("Scan & Submit") and media:
            fid = str(uuid.uuid4())
            cleaned_name = safe_filename(media.name)
            temp_path = PENDING_DIR / f"{fid}_{cleaned_name}"
            temp_path.write_bytes(media.read())

            matched = []
            try:
                if media.type.startswith("image"):
                    emb = face_embed(temp_path)
                    for u in read_json(USERS_JSON):
                        if cos_sim(emb, u["face"]) > 0.6:
                            matched.append(u["name"])
            except:
                pass

            status = "pending" if matched else "approved"
            final_path = (PENDING_DIR if status == "pending" else APPROVED_DIR) / temp_path.name
            shutil.move(temp_path, final_path)

            submission = {
                "id": fid,
                "file": str(final_path),
                "uploader": user["name"],
                "matches": matched,
                "approved_by": [],
                "status": status
            }
            submissions = read_json(SUB_JSON)
            submissions.append(submission)
            write_json(SUB_JSON, submissions)

            if matched:
                st.warning(f"⚠️ Needs approval from: {', '.join(matched)}")
            else:
                st.success("Auto-approved ✅")

    if page == "My Uploads":
        st.subheader("Your uploaded content")
        subs = read_json(SUB_JSON)
        updated_subs = []
        for s in subs:
            if s["uploader"] == user["name"]:
                if os.path.exists(s["file"]):
                    display_media(s["file"])
                    col1, col2 = st.columns([6, 1])
                    if col2.button("🗑️", key="del_" + s["id"], help="Delete"):
                        os.remove(s["file"])
                        continue
                updated_subs.append(s)
        write_json(SUB_JSON, updated_subs)

    if page == "Requests":
        st.subheader("Media waiting for your approval")
        subs = read_json(SUB_JSON)
        for s in subs:
            if user["name"] in s["matches"] and s["status"] == "pending":
                if os.path.exists(s["file"]):
                    display_media(s["file"])
                    col1, col2 = st.columns(2)
                    if col1.button("Approve", key=s["id"]+"_a"):
                        if user["name"] not in s["approved_by"]:
                            s["approved_by"].append(user["name"])
                            if sorted(s["approved_by"]) == sorted(s["matches"]):
                                s["status"] = "approved"
                                new_path = APPROVED_DIR / Path(s["file"]).name
                                shutil.move(s["file"], new_path)
                                s["file"] = str(new_path)
                                st.success("Approved ✅")
                            else:
                                st.info("Approval recorded. Awaiting others…")
                        write_json(SUB_JSON, subs)
                    if col2.button("Reject", key=s["id"]+"_r"):
                        s["status"] = "rejected"
                        new_path = REJECTED_DIR / Path(s["file"]).name
                        shutil.move(s["file"], new_path)
                        s["file"] = str(new_path)
                        write_json(SUB_JSON, subs)
                        st.error("Rejected ❌")

    if page == "Rejected":
        st.subheader("Rejected content")
        for s in read_json(SUB_JSON):
            if s["status"] == "rejected" and (s["uploader"] == user["name"] or user["name"] in s["matches"]):
                if os.path.exists(s["file"]):
                    display_media(s["file"])

    if page == "Gallery":
        st.subheader("Approved content with your presence")
        for s in read_json(SUB_JSON):
            if s["status"] == "approved" and (user["name"] in s["matches"] or user["name"] == s["uploader"]):
                if os.path.exists(s["file"]):
                    display_media(s["file"])
                    st.caption(f"Uploaded by {s['uploader']}")
                    links = generate_share_links(s['file'], s['uploader'])
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown(f"[🌐 Facebook]({links['Facebook']})", unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"[💼 LinkedIn]({links['LinkedIn']})", unsafe_allow_html=True)
                    with col3:
                        st.markdown(f"[🔦 Twitter]({links['Twitter']})", unsafe_allow_html=True)
                    with col4:
                        st.code(links['Copy Link'])
                        st.caption("Copy this link manually")

    if page == "Logout":
        st.session_state.auth_user = None
        st.experimental_rerun()
