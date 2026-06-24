# LangGraph Multi-Tool RAG Chatbot

A conversational AI chatbot built with LangGraph that supports 
PDF question answering, web search, stock prices, and calculations.

## Features
- Upload multiple PDFs and ask questions about them (RAG)
- Web search for current information
- Real-time stock price lookup
- Calculator tool
- Persistent conversation history per thread (SQLite)
- Multiple chat threads with load and delete support

## Tech Stack
- LangGraph — agent graph and state management
- LangChain — LLM and tool abstractions  
- Groq (openai/gpt-oss-120b) — LLM
- FAISS — vector store for PDF embeddings
- Jina Embeddings — text to vector conversion
- SQLite — conversation memory
- Streamlit — frontend UI

## How to Run

1. Clone the repository
git clone https://github.com/hamed418/Multi-Utility-Chatbot.git

2. Install dependencies
pip install -r requirements.txt

3. Create a .env file with your API keys
OPENAI_API_KEY=your_key
GROQ_API_KEY=your_key
JINA_API_KEY=your_key
ALPHA_VANTAGE_API_KEY=your_key

4. Run the app
streamlit run app.py

## Architecture
User Input → Streamlit Frontend
               ↓
          LangGraph Graph
               ↓
         chat_node (LLM)
               ↓
      tools_condition check
       ↓              ↓
  tool_node          END
  (RAG/Search/
  Stock/Calc)
       ↓
  chat_node (final answer)
       ↓
  SqliteSaver → chatbot.db
