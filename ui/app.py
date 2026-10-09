"""Streamlit front end: sign in, upload documents, ask questions.

It talks to the API over HTTP and holds no keys except Supabase's public (anon)
key, which is only used to sign users in.

Run with:  streamlit run ui/app.py
"""

import os
from pathlib import Path

import httpx
import streamlit as st

PUBLIC_SETTINGS = ("RAG_API_URL", "SUPABASE_URL", "SUPABASE_ANON_KEY")
UPLOAD_TYPES = ["pdf", "txt", "md"]


def load_settings() -> dict[str, str]:
    """Read the public settings from the environment, falling back to ``.env``.

    Only the three names above are ever read from the file, so the backend's
    secrets in the same file never enter this process's state.
    """
    values = {name: os.environ.get(name, "") for name in PUBLIC_SETTINGS}
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            name, _, value = line.strip().partition("=")
            if name in values and not values[name]:
                values[name] = value.strip().strip("\"'")
    values["RAG_API_URL"] = (values["RAG_API_URL"] or "http://127.0.0.1:8000").rstrip("/")
    values["SUPABASE_URL"] = values["SUPABASE_URL"].rstrip("/")
    return values


SETTINGS = load_settings()
API_URL = SETTINGS["RAG_API_URL"]
AUTH_URL = f"{SETTINGS['SUPABASE_URL']}/auth/v1"
ANON_KEY = SETTINGS["SUPABASE_ANON_KEY"]
SIGN_IN_ENABLED = bool(SETTINGS["SUPABASE_URL"] and ANON_KEY)


@st.cache_resource
def http_client() -> httpx.Client:
    return httpx.Client(timeout=180)


class SessionExpired(Exception):
    pass


# --- Supabase Auth ---------------------------------------------------------


def auth_request(path: str, payload: dict) -> httpx.Response:
    return http_client().post(f"{AUTH_URL}/{path}", headers={"apikey": ANON_KEY}, json=payload)


def store_session(body: dict) -> None:
    st.session_state.access_token = body["access_token"]
    st.session_state.refresh_token = body.get("refresh_token", "")
    st.session_state.email = (body.get("user") or {}).get("email", "")


def sign_out() -> None:
    for key in ("access_token", "refresh_token", "email", "history_loaded", "messages"):
        st.session_state.pop(key, None)


def refresh_session() -> bool:
    token = st.session_state.get("refresh_token")
    if not token:
        return False
    response = auth_request("token?grant_type=refresh_token", {"refresh_token": token})
    if response.status_code != 200:
        return False
    store_session(response.json())
    return True


def auth_error(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return f"Request failed (HTTP {response.status_code})."
    return body.get("msg") or body.get("error_description") or body.get("message") or "Failed."


def sign_in_page() -> None:
    st.title("RAG")
    st.caption("Sign in to upload documents and ask questions about them.")
    sign_in_tab, sign_up_tab = st.tabs(["Sign in", "Create account"])

    with sign_in_tab, st.form("sign-in"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Sign in", type="primary"):
            response = auth_request(
                "token?grant_type=password", {"email": email.strip(), "password": password}
            )
            if response.status_code == 200:
                store_session(response.json())
                st.rerun()
            st.error(auth_error(response))

    with sign_up_tab, st.form("sign-up"):
        email = st.text_input("Email", key="new-email")
        password = st.text_input("Password (at least 6 characters)", type="password", key="new-pw")
        if st.form_submit_button("Create account"):
            response = auth_request("signup", {"email": email.strip(), "password": password})
            if response.status_code != 200:
                st.error(auth_error(response))
            elif response.json().get("access_token"):
                store_session(response.json())
                st.rerun()
            else:
                st.success("Account created. Confirm it from the email you received, then sign in.")


# --- API calls -------------------------------------------------------------


def api(method: str, path: str, **kwargs) -> httpx.Response:
    """Call the API as the signed-in user, renewing the session once if it expired."""
    for attempt in range(2):
        headers = {}
        if SIGN_IN_ENABLED:
            headers["Authorization"] = f"Bearer {st.session_state.get('access_token', '')}"
        response = http_client().request(method, f"{API_URL}{path}", headers=headers, **kwargs)
        if response.status_code != 401 or not SIGN_IN_ENABLED:
            return response
        if attempt == 0 and refresh_session():
            continue
        raise SessionExpired
    return response


def error_detail(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None
    return detail if isinstance(detail, str) else f"Request failed (HTTP {response.status_code})."


# --- Pages -----------------------------------------------------------------


def documents_sidebar() -> None:
    st.sidebar.subheader("Your documents")
    uploads = st.sidebar.file_uploader(
        "Add PDF, text or Markdown files",
        type=UPLOAD_TYPES,
        accept_multiple_files=True,
        key=f"uploader-{st.session_state.get('uploader_round', 0)}",
    )
    if uploads and st.sidebar.button("Upload", type="primary"):
        for upload in uploads:
            with st.sidebar.status(f"Indexing {upload.name}…") as status:
                response = api(
                    "POST", "/documents", files={"file": (upload.name, upload.getvalue())}
                )
                if response.status_code == 201 and response.json()["status"] == "ready":
                    status.update(label=f"{upload.name}: indexed", state="complete")
                else:
                    reason = (
                        response.json().get("error")
                        if response.status_code == 201
                        else error_detail(response)
                    )
                    status.update(label=f"{upload.name}: {reason}", state="error")
        # A new key empties the uploader so the same files are not offered again.
        st.session_state.uploader_round = st.session_state.get("uploader_round", 0) + 1

    response = api("GET", "/documents")
    if response.status_code != 200:
        st.sidebar.error(error_detail(response))
        return
    documents = response.json()
    if not documents:
        st.sidebar.info("No documents yet. Upload one to start asking questions.")
    for document in documents:
        name, remove = st.sidebar.columns([5, 1])
        note = (
            f"{document['chunks']} chunks" if document["status"] == "ready" else document["status"]
        )
        name.markdown(f"**{document['filename']}**  \n:gray[{note}]")
        if remove.button("✕", key=f"delete-{document['id']}", help="Remove this document"):
            api("DELETE", f"/documents/{document['id']}")
            st.rerun()


def show_sources(sources: list[dict], answerable: bool) -> None:
    if not sources:
        return
    cited = [source for source in sources if source.get("cited")]
    label = f"Sources ({len(cited)} cited)" if answerable else "Retrieved passages (not used)"
    with st.expander(label):
        for source in sources:
            marker = "✅" if source.get("cited") else "▫️"
            title = source.get("source") or source.get("title") or source["doc_id"]
            st.markdown(f"{marker} **[{source['number']}] {title}**")
            if source.get("text"):
                st.caption(source["text"])


def load_history() -> None:
    if st.session_state.get("history_loaded"):
        return
    st.session_state.messages = []
    response = api("GET", "/messages", params={"limit": 30})
    if response.status_code == 200:
        # The API returns newest first; a conversation reads oldest first.
        st.session_state.messages = list(reversed(response.json()))
    st.session_state.history_loaded = True


def chat_page() -> None:
    st.title("Ask your documents")
    options = st.sidebar.expander("Retrieval options")
    mode = options.selectbox("Search mode", ["hybrid", "dense", "sparse"])
    top_k = options.slider("Passages given to the model", 1, 10, 5)
    rerank = options.checkbox("Rerank with a cross-encoder (slower)")

    load_history()
    for message in st.session_state.messages:
        st.chat_message("user").write(message["question"])
        with st.chat_message("assistant"):
            st.write(message["answer"])
            show_sources(message.get("sources", []), message.get("answerable", True))

    question = st.chat_input("Ask a question about your documents")
    if not question:
        return
    st.chat_message("user").write(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching and answering…"):
            response = api(
                "POST",
                "/ask",
                json={"question": question, "mode": mode, "top_k": top_k, "rerank": rerank},
            )
        if response.status_code != 200:
            st.error(error_detail(response))
            return
        body = response.json()
        st.write(body["answer"])
        show_sources(body["sources"], body["answerable"])
    st.session_state.messages.append(body)


def main() -> None:
    st.set_page_config(page_title="RAG", page_icon="📚")
    if SIGN_IN_ENABLED and "access_token" not in st.session_state:
        sign_in_page()
        return
    try:
        if SIGN_IN_ENABLED:
            st.sidebar.caption(f"Signed in as {st.session_state.get('email', '')}")
            if st.sidebar.button("Sign out"):
                sign_out()
                st.rerun()
        documents_sidebar()
        chat_page()
    except SessionExpired:
        sign_out()
        st.warning("Your session has expired. Sign in again.")
        st.rerun()
    except httpx.HTTPError:
        st.error(f"Could not reach the API at {API_URL}. Is it running?")


main()
