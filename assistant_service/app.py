import os
import json
import threading
import time
import requests
import streamlit as st
import torch
from transformers import pipeline, TextIteratorStreamer


st.set_page_config(page_title="Surchi AI Assistant", page_icon="☕", layout="centered")

AVATARS = {
    "user": "🧑‍💻",
    "assistant": "☕"
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

config = load_config()

MODEL_DIR = os.getenv(
    "MODEL_DIR",
    os.path.join(BASE_DIR, config.get("model_dir", "models/qwen2.5-0.5b"))
)
PROMPT_FILE = os.getenv(
    "PROMPT_FILE",
    os.path.join(BASE_DIR, config.get("prompt_file", "prompt.txt"))
)
STORAGE_URL = os.getenv(
    "STORAGE_SERVICE_URL",
    config.get("storage_service_url", "http://storage_service:5000")
)
MASTER_URL = os.getenv(
    "WEB_MASTER_URL",
    config.get("web_master_url", "http://web_master:8080")
)
MAX_NEW_TOKENS = int(config.get("max_new_tokens", 120))

@st.cache_resource
def load_assistant_pipeline():
    pipe = pipeline(
        "text-generation",
        model=MODEL_DIR,
        tokenizer=MODEL_DIR,
        dtype=torch.float32,
        device="cpu"
    )
    return pipe

pipe = load_assistant_pipeline()


def fetch_storage_snapshot():
    try:
        catalog = requests.get(f"{STORAGE_URL}/api/storage/catalog", timeout=2).json()
        benchmarks = requests.get(f"{STORAGE_URL}/api/storage/benchmarks", timeout=2).json()
        return {"catalog": catalog, "benchmarks": benchmarks}
    except Exception as e:
        return {"error": f"Storage service unavailable: {str(e)}"}

    
def build_system_prompt(snapshot_data):
    try:
        with open(PROMPT_FILE, "r", encoding="utf-8") as f:
            template = f.read()
        return template.format(snapshot=json.dumps(snapshot_data, indent=2, ensure_ascii=False))
    except Exception:
        return f"You are Surchi. Answer questions using this database snapshot:\n{snapshot_data}"

st.title("☕ Surchi — AI Platform Assistant")

st.caption("Ask questions about datasets, missing values, and benchmark results.")

INITIAL_MESSAGE = {
    "role": "assistant",
    "content": "Hello! I am Surchi, your AI assistant for Impute Hub. I have access to real-time database snapshots and benchmark results. How can I help you today?"
}


if "messages" not in st.session_state:
    st.session_state.messages = [INITIAL_MESSAGE]

for msg in st.session_state.messages:
    avatar = AVATARS.get(msg["role"])
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

if user_prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user", avatar='🧑‍💻'):
        st.markdown(user_prompt)

    snapshot = fetch_storage_snapshot()
    system_prompt = build_system_prompt(snapshot)
    dialog = [{"role": "system", "content": system_prompt}]
    dialog.extend(st.session_state.messages[-6:])
    streamer = TextIteratorStreamer(
        pipe.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    prompt_text = pipe.tokenizer.apply_chat_template(
        dialog,
        tokenize=False,
        add_generation_prompt=True
    )
    gen_kwargs = dict(
        streamer=streamer,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        return_full_text=False
    )

    gen_thread = threading.Thread(target=pipe, args=(prompt_text,), kwargs=gen_kwargs, daemon=True)
    gen_thread.start()

    def stream_response():
        for chunk in streamer:
            yield chunk

    with st.chat_message("assistant", avatar='☕'):
        full_reply = st.write_stream(stream_response())

    st.session_state.messages.append({"role": "assistant", "content": full_reply})


def heartbeat_worker():
    time.sleep(5)
    while True:
        try:
            res = requests.get(f"{MASTER_URL}/health", timeout=2)
            if res.status_code != 200:
                os._exit(0)
        except Exception:
            os._exit(0)
        time.sleep(2)

#if "heartbeat_started" not in st.session_state:
    #hb_thread = threading.Thread(target=heartbeat_worker, daemon=True)
    #hb_thread.start()
    #st.session_state.heartbeat_started = True
