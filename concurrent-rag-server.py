"""
Enhanced Document QA System with FastHTML - Concurrent Version
"""
import os
from dotenv import load_dotenv
import uvicorn
from fasthtml import *
from fasthtml.common import *
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import asyncio
from typing import Optional

# Load environment variables at startup
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Please set OPENAI_API_KEY in environment variables or .env file")

# Initialize app
app = FastHTML(hdrs=(picolink,))

# Global variables with thread safety
class QASystemManager:
    def __init__(self):
        self.qa_system = None
        self.lock = Lock()
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)
        self.initialization_lock = Lock()
        self._initialized = False

    def initialize(self):
        if not self._initialized:
            with self.initialization_lock:
                if not self._initialized:  # Double-check pattern
                    from main import EnhancedDocumentQASystem
                    docs_dir = "documents"
                    os.makedirs(docs_dir, exist_ok=True)
                    self.qa_system = EnhancedDocumentQASystem(docs_dir, openai_api_key, reset_db=True)
                    self._initialized = True

    async def ask_question_async(self, question: str):
        def _ask():
            with self.lock:  # Ensure thread-safe access to qa_system
                return self.qa_system.ask_question(question)
        
        # Execute the blocking operation in the thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _ask)

# Initialize manager
qa_manager = QASystemManager()

@app.on_event("startup")
async def startup_event():
    qa_manager.initialize()

@app.get("/")
def home():
    return Title("EI-Hub RAG"), Main(
        H1("EI-Hub Retrieval-Augmented Generation Search"),
        P("This RAG search is designed to assist in searching and retrieving information from EI-Hub and PCG documentation."),
        Form(
            Input(id="new-question", name="question", placeholder="Enter question"),
            Button("Search"),
            enctype="multipart/form-data",
            hx_post="/respond",
            hx_target="#result"
        ),
        Br(), 
        Div(id="result"),
        cls="container"
    )

@app.post("/respond")
async def eihubrag(question: str):
    try:
        # Process question asynchronously
        response = await qa_manager.ask_question_async(question)
        
        return Main(
            H2("\nAnswer:", response['answer']),
            P("\nConfidence Score:", f"{response['confidence']:.2f}"),
            Div(
                P("Sources:"),
                Ul(*[
                    Li(f"{source['source']} (Page: {source['page']})") 
                    for source in response['sources']
                ])
            )
        )
    except Exception as e:
        return Main(P(f"Error: {str(e)}"))

# Cache for rate limiting
@lru_cache(maxsize=1000)
def get_last_request_time(client_id: str) -> Optional[float]:
    return None

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host='127.0.0.1', 
        port=int(os.getenv("PORT", default=5000)), 
        reload=True,
        workers=4  # Adjust based on your server capacity
    )
