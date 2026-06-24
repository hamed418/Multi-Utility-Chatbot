from __future__ import annotations
 
import os
import sqlite3
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict
 
from dotenv import load_dotenv
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import duckduckgo_search
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_community.embeddings.jina import JinaEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph,END
# this is a reducer to add messages into a list used in State in langgraph 
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition
import requests
 
load_dotenv()
 

# 1. LLM + embeddings

llm = ChatGroq(model="openai/gpt-oss-120b")
embeddings = JinaEmbeddings(model="jina-embeddings-v5-omni-small")
 

# 2. PDF retriever store (per thread)

THREAD_RETRIEVERS: Dict[str, Any] = {}
THREAD_METADATA: Dict[str, dict] = {}
 
 
def get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in THREAD_RETRIEVERS:
        return THREAD_RETRIEVERS[thread_id]
    return None
 
 
def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None):
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.
 
    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")
 
# creating a temporarily file here:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name
 
    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
 
        # Guard: if no text was extracted, the PDF is likely scanned or empty
        if not docs:
            raise ValueError(
                f"No pages could be read from '{filename}'. "
                "The PDF may be scanned, image-based, or password-protected."
            )
 
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600, chunk_overlap=120, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)
 
        # Guard: if pages loaded but all text was whitespace, chunks will be empty
        if not chunks:
            raise ValueError(
                f"No text content found in '{filename}'. "
                "The PDF may contain only images or have no extractable text."
            )
 
        vector_store = FAISS.from_documents(chunks, embeddings)
 
        # If a FAISS store already exists for this thread, merge into it
        # so ALL uploaded PDFs are searchable together — not just the latest one
        existing_retriever = THREAD_RETRIEVERS.get(str(thread_id))
        if existing_retriever is not None:
            existing_store = existing_retriever.vectorstore
            existing_store.merge_from(vector_store)
            retriever = existing_store.as_retriever(
                search_type="similarity", search_kwargs={"k": 4}
            )
        else:
            retriever = vector_store.as_retriever(
                search_type="similarity", search_kwargs={"k": 4}
            )
 
        THREAD_RETRIEVERS[str(thread_id)] = retriever
 
        # Keep a list of all uploaded filenames for this thread
        existing_meta = THREAD_METADATA.get(str(thread_id), {"filenames": [], "documents": 0, "chunks": 0})
        THREAD_METADATA[str(thread_id)] = {
            # adding filename with old file names if have
            "filenames": existing_meta["filenames"] + [filename or os.path.basename(temp_path)],
            "filename": filename or os.path.basename(temp_path),  # latest one (for display)
            "documents": existing_meta["documents"] + len(docs),
            "chunks": existing_meta["chunks"] + len(chunks),
        }
 
        meta = THREAD_METADATA[str(thread_id)]
        return {
            "filename": meta["filename"],
            "documents": meta["documents"],
            "chunks": meta["chunks"],
        }
    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass
 
 

# 3. Tools

search_tool = DuckDuckGoSearchRun(region="us-en")
 
 
@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
 
        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}
 
 
@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """

    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    )
    r = requests.get(url)
    return r.json()
 
 
def make_rag_tool(thread_id: str):
    """
    Build a rag_tool that already knows the thread_id.
    This way thread_id is NOT a parameter the LLM needs to pass —
    which prevents Groq from failing on Optional[str] schema generation.
    """
    @tool
    def rag_tool(query: str) -> dict:
        """Retrieve relevant information from the uploaded PDF for this chat thread."""
        retriever = get_retriever(thread_id)
        if retriever is None:
            return {
                "error": "No document indexed for this chat. Upload a PDF first.",
                "query": query,
            }
 
        result = retriever.invoke(query)
        context = [doc.page_content for doc in result]
        metadata = [doc.metadata for doc in result]
 
        return {
            "query": query,
            "context": context,
            "metadata": metadata,
            "source_file": THREAD_METADATA.get(str(thread_id), {}).get("filename"),
        }
 
    return rag_tool
 
 
# Base tools (no rag_tool here — it's added per-node with the right thread_id)
base_tools = [search_tool, get_stock_price, calculator]
 

# 4. State

class ChatState(TypedDict):
    # Annotated gives flexibility to add more helper in this case:add_messages
    messages: Annotated[list[BaseMessage], add_messages]
 
 

# 5. Nodes

def chat_node(state: ChatState, config=None):
    """LLM node that may answer or request a tool call."""
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")
 
    # Build a rag_tool that already has thread_id baked in.
    # The LLM only needs to pass "query" — no Optional params that confuse Groq.
    rag_tool = make_rag_tool(thread_id)
    tools = [*base_tools, rag_tool]
    llm_with_tools = llm.bind_tools(tools)
 
    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. If the user asks about an uploaded PDF, "
            "use the `rag_tool`. You can also use web search, stock price, and "
            "calculator tools when helpful. If no document is available, ask the user "
            "to upload a PDF."
        )
    )
 
    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}
 
 
def tool_node(state: ChatState, config=None):
    """
    Custom tool node that dispatches tool calls.
    For rag_tool we build a fresh one with the correct thread_id from config,
    so the retriever lookup always finds the right thread's PDF.
    """
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")
 
    # Build the full tool list with the correct thread_id baked into rag_tool
    rag_tool = make_rag_tool(thread_id)

    # get tools name , as langgraph defaultly give this information 
    all_tools = {t.name: t for t in [*base_tools, rag_tool]}
 
    # The last message holds the tool call(s) the LLM requested
    last_message = state["messages"][-1]
    results = []

    # if user ask a question from llm
    # than tool call from llm looks like this :
    # {
    #    "name": "rag_tool",
    #    "args": {"query": "revenue"},
    #    "id": "call_abcd123"
    # }
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id   = tool_call["id"]
 
        if tool_name in all_tools:
            output = all_tools[tool_name].invoke(tool_args)
        else:
            output = {"error": f"Unknown tool: {tool_name}"}
 
        # LangGraph expects a ToolMessage back for each tool call
        results.append(ToolMessage(content=str(output), tool_call_id=tool_id, name=tool_name))
 
    return {"messages": results}
 

# 6. Checkpointer

conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)
 

# 7. Graph

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
 
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")
graph.add_edge("chat_node",END)

chatbot = graph.compile(checkpointer=checkpointer)
chatbot





# 8. Helpers

def retrieve_all_threads():
    all_threads = set()
    try:
        for checkpoint in checkpointer.list(None): # None here means "no filter — give me ALL checkpoints from every thread"
            all_threads.add(checkpoint.config["configurable"]["thread_id"]) # 
    except Exception:
        pass  # if SQLite is empty or fails, just return empty list
    return list(all_threads)

 
def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in THREAD_RETRIEVERS
 
 
def thread_document_metadata(thread_id: str) -> dict:
    return THREAD_METADATA.get(str(thread_id), {})
 
 

# 9. Delete a thread

def delete_thread(thread_id: str):
    """
    Delete all checkpoints for a specific thread from SQLite.
    Uses the same synchronous conn object that SqliteSaver already uses.
    """
    thread_id = str(thread_id)
 
    # Step 1: Find out which tables actually exist in the database
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    # It fetches the names of all tables currently active inside chatbot.db.
    existing_tables = {row[0] for row in cursor.fetchall()}
 
    # Step 2: These are the tables LangGraph stores checkpoint data in
    tables_to_clean = ["checkpoints", "writes"]
 
    # Step 3: Delete only from tables that actually exist (safe check)
    for table in tables_to_clean:
        if table in existing_tables:
            conn.execute(f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,))
    
    # tells SQLite to write those changes permanently to disk.
    conn.commit()
 
    # Step 4: Also remove the in-memory PDF retriever for this thread (if any)
    THREAD_RETRIEVERS.pop(thread_id, None)
    THREAD_METADATA.pop(thread_id, None)
 
 

# 10. Debug helper — print all table names

def show_tables():
    """Print all table names in the SQLite database. Useful for debugging."""
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    print("SQLite tables:", tables)
    return tables
 
 
# Call once at startup so you can see your tables in the terminal
show_tables()