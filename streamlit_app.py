import html
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import markdown as md
import streamlit as st
from api_client import APIClient
from api import API_BASE_URL, APP_TITLE

SUGGESTED_QUESTIONS = [
    "How many annual leaves do I have?",
    "Can unused leaves be carried forward?",
    "What is the hotel reimbursement limit for domestic travel?",
    "Can I work from home more than three days a week?",
]

CLIENT = APIClient()


def load_css() -> None:
    css_path = Path("styles.css")
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def get_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def render_markdown(text: str) -> str:
    return md.markdown(text, extensions=["fenced_code", "tables", "nl2br", "sane_lists"])


def add_message(role: str, content: str, sources: Optional[List[str]] = None) -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    entry = {
        "role": role,
        "content": content,
        "timestamp": get_timestamp(),
        "sources": sources or [],
    }
    st.session_state.messages.append(entry)


def clear_messages() -> None:
    st.session_state.messages = []


def set_default_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "backend_status" not in st.session_state:
        st.session_state.backend_status = "unknown"
    if "prompt" not in st.session_state:
        st.session_state.prompt = ""
    if "response_error" not in st.session_state:
        st.session_state.response_error = ""
    if "is_loading" not in st.session_state:
        st.session_state.is_loading = False


def show_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            "<div class='sidebar-card'><div class='sidebar-brand'>🧾</div><h2>Employment Policy Assistant</h2><p>Enterprise-ready policy question assistant for HR, payroll, and compliance.</p></div>",
            unsafe_allow_html=True,
        )

        new_chat, clear_chat = st.columns([1, 1])
        if new_chat.button("New Chat"):
            clear_messages()
            st.session_state.response_error = ""
        if clear_chat.button("Clear Chat"):
            clear_messages()
            st.session_state.response_error = ""

        st.markdown("### System status")
        status_label = st.session_state.backend_status
        if status_label == "online":
            st.markdown("<span class='status-pill'>● Backend Online</span>", unsafe_allow_html=True)
        elif status_label == "offline":
            st.markdown("<span class='status-pill badge-offline'>● Backend Offline</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='status-pill'>● Checking backend...</span>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Uploaded documents")
        if st.session_state.uploaded_files:
            for file_name in st.session_state.uploaded_files:
                st.markdown(f"<div class='upload-chip'>{html.escape(file_name)}</div>", unsafe_allow_html=True)
        else:
            st.info("No policy documents uploaded yet.")

        st.markdown("---")
        with st.expander("Upload policy documents", expanded=True):
            uploaded_files = st.file_uploader(
                "Drag and drop PDF, DOCX or TXT files",
                type=["pdf", "docx", "txt"],
                accept_multiple_files=True,
                key="upload_widget",
            )
            if st.button("Upload documents"):
                if not uploaded_files:
                    st.warning("Please select at least one file to upload.")
                else:
                    upload_progress = st.progress(0)
                    try:
                        for idx, value in enumerate([10, 30, 60, 90, 100]):
                            upload_progress.progress(value)
                        result = CLIENT.upload_files(uploaded_files)
                        st.session_state.uploaded_files = result.get("uploaded_files", [])
                        st.success("Files uploaded successfully.")
                    except Exception as exc:
                        st.error(f"Upload failed: {exc}")
                    finally:
                        upload_progress.empty()

        st.markdown("---")
        st.markdown("### Backend health")
        if st.session_state.backend_status == "online":
            st.success("API connected")
        elif st.session_state.backend_status == "offline":
            st.error("API unreachable")
        else:
            st.info("Checking API connection...")


def send_question(question: str) -> None:
    if not question.strip():
        st.session_state.response_error = "Please enter a question to continue."
        return

    st.session_state.response_error = ""
    add_message("user", question)
    st.session_state.prompt = ""
    st.session_state.is_loading = True

    with st.spinner("Generating answer..."):
        try:
            result = CLIENT.ask_question(question)
            add_message("assistant", result.get("answer", ""), sources=result.get("sources", []))
        except Exception as exc:
            st.session_state.response_error = str(exc)
            st.error(f"Unable to get an answer: {exc}")
        finally:
            st.session_state.is_loading = False


def render_message(message: Dict[str, Any], index: int) -> None:
    content_html = render_markdown(message["content"])
    timestamp_html = html.escape(message["timestamp"])
    role = message["role"]
    side_class = "user" if role == "user" else "assistant"
    header_label = "You" if role == "user" else "Assistant"
    source_html = ""

    if role == "assistant" and message.get("sources"):
        sources_list = "".join(
            f"<li>{html.escape(source)}</li>" for source in message["sources"]
        )
        source_html = (
            "<details class='source-details'><summary>Retrieved Sources</summary>"
            f"<ul>{sources_list}</ul></details>"
        )

    hidden_id = f"copy-text-{index}"
    safe_copy_text = html.escape(message["content"])

    bubble_html = (
        f"<div class='chat-message {side_class}'>"
        f"<div class='message-bubble {side_class}'>"
        f"<div class='message-meta'>{header_label} • {timestamp_html}</div>"
        f"<div class='message-text markdown-body'>{content_html}</div>"
        f"{source_html}"
    )

    if role == "assistant":
        bubble_html += (
            f"<button class='copy-button' onclick=\"navigator.clipboard.writeText(document.getElementById('{hidden_id}').innerText)\">Copy answer</button>"
            f"<div id='{hidden_id}' style='display:none;'>{safe_copy_text}</div>"
        )

    bubble_html += "</div></div>"
    st.markdown(bubble_html, unsafe_allow_html=True)


def render_chat_feed() -> None:
    chat_container = st.container()
    with chat_container:
        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
        for index, message in enumerate(st.session_state.messages):
            render_message(message, index)
        st.markdown("</div>", unsafe_allow_html=True)


def render_welcome() -> None:
    st.markdown(
        "<div class='welcome-card'>"
        "<h1>Welcome to your Employment Policy Assistant</h1>"
        "<p>Ask HR policy questions with a ChatGPT-inspired experience. Upload documents, send questions, and get grounded policy answers.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("### Suggested questions")
    suggestion_columns = st.columns(2)
    for index, suggestion in enumerate(SUGGESTED_QUESTIONS):
        if suggestion_columns[index % 2].button(suggestion):
            send_question(suggestion)


def render_footer() -> None:
    if st.session_state.messages:
        history_text = "\n\n".join(
            f"[{m['timestamp']}] {m['role'].title()}: {m['content']}" for m in st.session_state.messages
        )
        st.download_button(
            "Download conversation",
            history_text,
            file_name="policy_chat_history.txt",
            mime="text/plain",
        )


def check_backend_health() -> None:
    try:
        status = CLIENT.health_check()
        st.session_state.backend_status = "online"
        st.session_state.data_dir = status.get("data_dir", "")
        st.session_state.document_list = status.get("files", [])
    except Exception:
        st.session_state.backend_status = "offline"


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🧾", layout="wide")
    load_css()
    set_default_session_state()
    check_backend_health()

    show_sidebar()

    st.markdown("<div class='main-content'>", unsafe_allow_html=True)
    st.markdown("<div style='display:flex; justify-content:space-between; align-items:center; gap:12px;'>"
                "<div><h1>Employment Policy Assistant</h1><p>Ask questions about company policy and receive grounded responses.</p></div>"
                "</div>", unsafe_allow_html=True)

    if st.session_state.messages:
        render_chat_feed()
    else:
        render_welcome()

    with st.form(key="chat_form", clear_on_submit=False):
        prompt = st.text_area(
            "",
            value=st.session_state.prompt,
            placeholder="Ask a question...",
            key="prompt_area",
            height=120,
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Send")
        if submitted:
            send_question(prompt)

    if st.session_state.response_error:
        st.error(st.session_state.response_error)

    render_footer()
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
