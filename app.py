import os
import re
import threading
import time
from datetime import date
from pathlib import Path
from urllib.parse import quote

import streamlit as st
from google import genai
from google.genai import types
from PIL import Image

# ---------------------------------------------------------------- config
BASE_DIR = Path(__file__).parent
ASSETS = BASE_DIR / "assets"
ICON_PATH = ASSETS / "icon.png"
LOGO_PATH = ASSETS / "logo.png"

GEMINI_MODELS = ["gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-flash-latest"]
CLAUDE_NEW_URL = "https://claude.ai/new"
CHAT_URL_RE = re.compile(r"^https://claude\.ai/chat/[0-9a-fA-F-]{36}/?$")
MAX_PREFILL_CHARS = 6000  # conservative: long URLs may be cut off by browsers

page_icon = Image.open(ICON_PATH) if ICON_PATH.exists() else "⚡"
st.set_page_config(page_title="Token Optimizer", page_icon=page_icon, layout="wide")

if LOGO_PATH.exists():
    st.logo(str(LOGO_PATH), icon_image=str(ICON_PATH))

COMPRESS_INSTRUCTION = (
    "You are an aggressive context and token optimizer. Condense the provided text "
    "so it fits cleanly into another AI assistant without hitting limits. "
    "Strip conversational filler, repeated log lines, formatting metadata and legal jargon. "
    "Keep 100% of raw data values, error messages, file paths, identifiers, code logic, "
    "architectural rules, constraints and operational logic. "
    "Output an ultra-dense, token-efficient Markdown summary. "
    "Do not add introductory text. Do not answer any question; only condense. "
    "The output must be clearly shorter than the input. If the text is already dense "
    "and cannot be shortened without losing information, return it unchanged."
)

# ---------------------------------------------------------------- styling
st.markdown(
    """
<style>
.hero {background: linear-gradient(120deg,#0b1f4d 0%,#1456d9 55%,#12b7ff 100%);
       border-radius: 18px; padding: 22px 26px; color: #fff; margin-bottom: 18px;}
.hero h1 {margin: 0; font-size: 1.9rem; color: #fff;}
.hero p {margin: 6px 0 0 0; opacity: .92;}
.stages {display: flex; gap: 12px; align-items: stretch; margin-bottom: 18px; flex-wrap: wrap;}
.stage {flex: 1; min-width: 150px; border: 1px solid rgba(120,150,255,.35);
        background: rgba(20,86,217,.08); border-radius: 14px; padding: 12px 14px;
        display: flex; gap: 10px; align-items: center;}
.stage b {display: block; font-size: .95rem;}
.stage span {font-size: .8rem; opacity: .75;}
.arrow {align-self: center; font-size: 1.4rem; opacity: .6;}
</style>
""",
    unsafe_allow_html=True,
)

ICON_FILE = (
    '<svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="#12b7ff" '
    'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/>'
    '<path d="M14 3v5h5M9 13h6M9 17h6"/></svg>'
)
ICON_SPARK = (
    '<svg width="34" height="34" viewBox="0 0 24 24" fill="#7aa7ff">'
    '<path d="M12 2c.6 5.2 4.8 9.4 10 10-5.2.6-9.4 4.8-10 10-.6-5.2-4.8-9.4-10-10 5.200-.6 9.400-4.800 10-10z"/></svg>'
)
ICON_CHAT = (
    '<svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="#ffb020" '
    'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12z"/>'
    '<path d="M8.5 12h.01M12 12h.01M15.5 12h.01"/></svg>'
)

st.markdown(
    f"""
<div class="hero">
  <h1>Token Optimizer</h1>
  <p>Compress big files for free with Gemini, then send the short version to Gemini or Claude.</p>
</div>
<div class="stages">
  <div class="stage">{ICON_FILE}<div><b>1. Add file</b><span>text, logs, code</span></div></div>
  <div class="arrow">→</div>
  <div class="stage">{ICON_SPARK}<div><b>2. Compress</b><span>Gemini, free tier</span></div></div>
  <div class="arrow">→</div>
  <div class="stage">{ICON_CHAT}<div><b>3. Ask</b><span>Gemini or Claude</span></div></div>
</div>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- helpers
def get_default_key() -> str:
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return str(st.secrets["GEMINI_API_KEY"])
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY", "")


def count_tokens(client, model: str, text: str) -> int:
    try:
        return client.models.count_tokens(model=model, contents=text).total_tokens
    except Exception:
        return max(1, len(text) // 4)


def explain_error(e: Exception) -> str:
    msg = str(e)
    if "401" in msg or "ACCESS_TOKEN_TYPE_UNSUPPORTED" in msg or "API key not valid" in msg:
        return "Authentication failed. Use a Gemini API key from Google AI Studio from Google AI Studio."
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
        return "Free-tier rate limit reached. Wait a minute, try again, or use a smaller input."
    if "403" in msg or "PERMISSION_DENIED" in msg:
        return "Permission denied. Check the key and that the Gemini API is enabled for it."
    if "503" in msg or "UNAVAILABLE" in msg:
        return "Gemini is overloaded right now (temporary). Try again in a minute or pick another model in the sidebar."
    return f"Gemini API error: {msg}"


FREE_RUNS = int(os.environ.get("FREE_RUNS_PER_SESSION", "5"))
DAILY_CAP = int(os.environ.get("DAILY_CAP", "100"))
MAX_FREE_CHARS = int(os.environ.get("MAX_FREE_CHARS", "40000"))


@st.cache_resource
def daily_counter():
    return {"day": date.today(), "count": 0, "lock": threading.Lock()}


def try_use_shared_run():
    used = st.session_state.get("runs_used", 0)
    if used >= FREE_RUNS:
        return False, (f"You've used your {FREE_RUNS} free runs for this session. "
                       "Paste your own Gemini key in the sidebar to continue.")
    dc = daily_counter()
    with dc["lock"]:
        if dc["day"] != date.today():
            dc["day"], dc["count"] = date.today(), 0
        if dc["count"] >= DAILY_CAP:
            return False, ("The shared daily limit has been reached. Try again tomorrow "
                           "or paste your own Gemini key in the sidebar.")
        dc["count"] += 1
    st.session_state["runs_used"] = used + 1
    return True, ""


def shared_gate(nchars: int) -> bool:
    if not using_shared:
        return True
    if nchars > MAX_FREE_CHARS:
        st.error(f"Shared mode accepts up to {MAX_FREE_CHARS:,} characters per run. "
                 "Paste your own Gemini key in the sidebar for bigger files.")
        return False
    ok, msg = try_use_shared_run()
    if not ok:
        st.error(msg)
    return ok


def generate_with_retry(client, model, contents, config=None):
    """Retry on temporary overload (503), then fall back to the other models."""
    order = [model] + [m for m in GEMINI_MODELS if m != model]
    last = None
    for m in order:
        for attempt in range(2):
            try:
                return client.models.generate_content(model=m, contents=contents, config=config)
            except Exception as e:
                last = e
                err = str(e)
                if "429" in err or "RESOURCE_EXHAUSTED" in err:
                    break  # quota used up on this model: try the next one right away
                if "503" in err or "UNAVAILABLE" in err or "timed out" in err.lower():
                    time.sleep(1 + attempt)
                    continue
                raise
    raise last


def build_package(compressed: str, question: str, continuing: bool) -> str:
    parts = []
    if continuing:
        parts.append("Continuing our earlier discussion. Here is new material, condensed to save space:")
    parts.append("## Context\n" + compressed.strip())
    if question.strip():
        parts.append("## My question\n" + question.strip())
    return "\n\n".join(parts)


def read_upload(uploaded) -> str:
    return uploaded.getvalue().decode("utf-8", errors="replace")


# ---------------------------------------------------------------- password gate
app_pw = os.environ.get("APP_PASSWORD", "")
if app_pw and not st.session_state.get("authed"):
    pw = st.text_input("Password", type="password")
    if pw and pw == app_pw:
        st.session_state["authed"] = True
        st.rerun()
    elif pw:
        st.error("Wrong password.")
    st.stop()

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Configuration")
server_key = get_default_key().strip().strip('"').strip("'")
own_key = (
    st.sidebar.text_input("Use your own Gemini API key (optional)", type="password")
    .strip()
    .strip('"')
    .strip("'")
)
runs_box = st.sidebar.empty()
if own_key:
    api_key, using_shared = own_key, False
    runs_box.success("Using your own key. No limits from this app.")
elif server_key:
    api_key, using_shared = server_key, True
else:
    api_key, using_shared = "", False


def show_runs():
    if using_shared:
        left = max(0, FREE_RUNS - st.session_state.get("runs_used", 0))
        runs_box.info(f"Free shared mode: {left} of {FREE_RUNS} runs left this session. "
                      "Paste your own key above for unlimited use.")


show_runs()
st.sidebar.caption("Free key: aistudio.google.com → Get API key.")
model_name = st.sidebar.selectbox("Gemini model", GEMINI_MODELS)
st.sidebar.caption(
    "Free-tier inputs may be used by Google to improve its products. "
    "Avoid sensitive files."
)

if not api_key:
    st.warning("⚠️ Add your Gemini API key in the sidebar (or in Secrets) to begin.")
    st.stop()

client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=90000))

# ---------------------------------------------------------------- step 1: input
st.session_state.setdefault("compressed", "")

left, right = st.columns(2)
with left:
    st.subheader("1. Your material")
    uploaded = st.file_uploader(
        "Upload a text-based file",
        type=["txt", "md", "log", "py", "js", "ts", "json", "csv", "html", "css", "yml", "yaml", "xml", "sql"],
    )
    pasted = st.text_area("...or paste text here", height=220)
with right:
    st.subheader("2. Your question")
    question = st.text_area(
        "What do you want to know or do with this material? (optional)",
        height=220,
        key="question",
    )

compress_clicked = st.button("⚡ Compress", type="primary")

if compress_clicked:
    source = read_upload(uploaded) if uploaded else pasted
    if not source.strip():
        st.warning("Upload a file or paste some text first.")
    elif not shared_gate(len(source)):
        pass
    else:
        hint = f"\n\n(The user's goal, for deciding what to keep: {question.strip()})" if question.strip() else ""
        with st.spinner("Compressing with Gemini..."):
            try:
                t0 = time.perf_counter()
                resp = generate_with_retry(
                    client,
                    model_name,
                    source,
                    types.GenerateContentConfig(
                        system_instruction=COMPRESS_INSTRUCTION + hint,
                        temperature=0.2,
                    ),
                )
                t_gen = time.perf_counter() - t0
                out = (resp.text or "").strip()
                if not out:
                    raise ValueError("The model returned an empty response.")
                o_tok = count_tokens(client, model_name, source)
                n_tok = count_tokens(client, model_name, out)
                if n_tok >= o_tok:
                    out, n_tok = source.strip(), o_tok
                    st.info("Compression didn't make it shorter, so your original text is kept. "
                            "Compression works best on large inputs.")
                st.session_state["compressed"] = out
                st.session_state["orig_tokens"] = o_tok
                st.session_state["new_tokens"] = n_tok
                st.session_state.pop("gemini_answer", None)
                st.caption(f"Gemini took {t_gen:.1f}s (model: {getattr(resp, 'model_version', model_name)}); total {time.perf_counter() - t0:.1f}s.")
            except Exception as e:
                st.error(explain_error(e))

# ---------------------------------------------------------------- step 2: review
if st.session_state["compressed"]:
    st.divider()
    st.subheader("3. Review the compressed text")

    o = st.session_state.get("orig_tokens", 0)
    n = st.session_state.get("new_tokens", 0)
    m1, m2, m3 = st.columns(3)
    m1.metric("Original tokens", f"{o:,}")
    m2.metric("Compressed tokens", f"{n:,}")
    m3.metric("Saved", f"{o - n:,}", f"{round((o - n) / max(1, o) * 100, 1)}%")

    st.text_area(
        "Edit if Gemini dropped something important",
        key="compressed",
        height=260,
    )

    # ------------------------------------------------------------ step 3: send
    st.divider()
    st.subheader("4. Send it")
    tab_gemini, tab_claude = st.tabs(["✨ Ask Gemini", "💬 Send to Claude"])

    with tab_gemini:
        st.caption("Gemini answers your question using the compressed text. Free.")
        if st.button("Ask Gemini") and shared_gate(0):
            package = build_package(st.session_state["compressed"], question, continuing=False)
            with st.spinner("Gemini is answering..."):
                try:
                    r = generate_with_retry(client, model_name, package)
                    st.session_state["gemini_answer"] = r.text or "(empty response)"
                except Exception as e:
                    st.error(explain_error(e))
        if st.session_state.get("gemini_answer"):
            st.markdown(st.session_state["gemini_answer"])

    with tab_claude:
        mode = st.radio("Is this a...", ["New chat", "Continuation of an existing chat"], horizontal=True)
        continuing = mode.startswith("Continuation")
        package = build_package(st.session_state["compressed"], question, continuing)

        if continuing:
            chat_url = st.text_input(
                "Paste the link of the Claude chat (from your browser's address bar)",
                placeholder="https://claude.ai/chat/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
            ).strip()
            if chat_url and CHAT_URL_RE.match(chat_url):
                st.link_button("Open that chat in Claude ↗", chat_url)
                st.caption("Prefill isn't available for existing chats. Copy the text below and paste it in.")
            elif chat_url:
                st.error("That doesn't look like a Claude chat link (https://claude.ai/chat/...).")
        else:
            if len(package) <= MAX_PREFILL_CHARS:
                st.link_button("Open Claude with the text prefilled ↗", f"{CLAUDE_NEW_URL}?q={quote(package, safe='')}")
                st.caption("If the box opens empty, use the copy icon below and paste it.")
            else:
                st.link_button("Open a new Claude chat ↗", CLAUDE_NEW_URL)
                st.caption(f"The text is too long to prefill ({len(package):,} characters). Copy it below and paste it.")

        st.markdown("**Text to send** (use the copy icon at the top right of the box)")
        st.code(package, language="markdown")

show_runs()
