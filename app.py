import uuid
 
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
 
from backend import (
    chatbot,
    delete_thread,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
)
 
 
# =========================== Utilities ===========================
def generate_thread_id():
    return uuid.uuid4()
 
 
def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []
 
 
def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)
 
 
def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])
 
 
# ======================= Session Initialization ============

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []
 
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()
 
if "chat_threads" not in st.session_state:
# Retrieving all thread_id from the sqlite database if any 
    st.session_state["chat_threads"] = retrieve_all_threads() # <- backend
 
if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}
 
add_thread(st.session_state["thread_id"])  # current thread into the list 
 
thread_key = str(st.session_state["thread_id"]) # Current thread 
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})  # PDFs for this thread
threads = st.session_state["chat_threads"][::-1] # reverse list (newest first)
selected_thread = None # nothing selected yet
 
#~~~~~~~~~~~~~~~~~~~~~~~~~ Sidebar ~~~~~~~~~~~~~~~~~~~~~~
st.sidebar.title("Chatbot")
st.sidebar.markdown(f"**Thread ID:** `{thread_key}`")
 
if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()
 
if thread_docs:
    all_filenames = list(thread_docs.keys())
    total_chunks = sum(d.get("chunks", 0) for d in thread_docs.values())
    total_pages  = sum(d.get("documents", 0) for d in thread_docs.values())
    if len(all_filenames) == 1:
        st.sidebar.success(f"Using `{all_filenames[0]}` ({total_chunks} chunks, {total_pages} pages)")
    else:
        st.sidebar.success(
            f"**{len(all_filenames)} PDFs indexed** ({total_chunks} total chunks, {total_pages} total pages)\n\n"
            + "\n".join(f"- `{name}`" for name in all_filenames)
        )
else:
    st.sidebar.info("No PDF indexed yet.")
 
uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])
if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")
    else:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
            try:
                summary = ingest_pdf(
                    # This  method (function) extracts the actual, raw binary contents of the file from the pdf
                    uploaded_pdf.getvalue(),
                    thread_id=thread_key,
                    filename=uploaded_pdf.name,  # extracting the file name
                )
                thread_docs[uploaded_pdf.name] = summary
                status_box.update(label=" PDF indexed", state="complete", expanded=False)
            except ValueError as e:
                # Clean error — empty or unreadable PDF
                status_box.update(label=" Failed to index PDF", state="error", expanded=False)
                st.sidebar.error(str(e))
 
#============================ Past Conversations =====================
st.sidebar.subheader("Past conversations")
 
if not threads:
    st.sidebar.write("No past conversations yet.")
else:
    for thread_id in threads:
        thread_str = str(thread_id)
 
        # Put the load button and delete button side by side in two columns
        col_load, col_delete = st.sidebar.columns([4, 1])
 
        with col_load:
            if st.button(thread_str, key=f"load-{thread_id}"):
                selected_thread = thread_id
 
        with col_delete:
            if st.button("🗑️", key=f"delete-{thread_id}", help="Delete this chat"):
                # If the user is deleting the currently active thread, start a new chat
                if thread_str == thread_key:  # thread_key is the current thread
                    reset_chat()
 
                # Delete from SQLite and in-memory retriever
                delete_thread(thread_str)
 
                # Remove from session state lists
                if thread_id in st.session_state["chat_threads"]:
                    st.session_state["chat_threads"].remove(thread_id)
                st.session_state["ingested_docs"].pop(thread_str, None)
 
                st.rerun()
 
# ============================ Main Layout ===============
st.title("Multi Utility Chatbot")
 
# Chat area
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
 
user_input = st.chat_input("Ask about your document or use tools or any question")
 
if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
 
    CONFIG = {
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }
 
    with st.chat_message("assistant"):
        status_holder = {"box": None}
 
        def ai_only_stream():
            for message_chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f" Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f" Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )
 
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content
 
        ai_message = st.write_stream(ai_only_stream())
 
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label=" Tool finished", state="complete", expanded=False
            )
 
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )
 
    doc_meta = thread_document_metadata(thread_key)
    if doc_meta:
        st.caption(
            f"Document indexed: {doc_meta.get('filename')} "
            f"(chunks: {doc_meta.get('chunks')}, pages: {doc_meta.get('documents')})"
        )
 
st.divider()

if selected_thread:
    st.session_state["thread_id"] = selected_thread
    messages = load_conversation(selected_thread)
    

    temp_messages = []
    for msg in messages:
        # removing any tool message in the llm replay 
        if isinstance(msg,ToolMessage):
            continue

        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        if msg.content == "" or msg.content is None:
            continue
        temp_messages.append({"role": role, "content": msg.content})
    st.session_state["message_history"] = temp_messages
    st.session_state["ingested_docs"].setdefault(str(selected_thread), {})
    st.rerun()