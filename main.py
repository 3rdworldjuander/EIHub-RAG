from fasthtml.common import *
from pathlib import Path
import os, uvicorn
from dotenv import load_dotenv
import logging
from datetime import datetime
import threading
import asyncio
from typing import Optional, Dict
from concurrent.futures import ThreadPoolExecutor
import queue
from functools import partial
from rich import print

# Import your existing QA system
from enhanced_qa_system import EnhancedDocumentQASystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastHTML app
app = FastHTML(hdrs=(picolink,))

class AppState:
    _instance: Optional['AppState'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.qa_system = None
                cls._instance.system_status = "initializing"
                cls._instance.error_message = ""
                cls._instance.document_count = 0
                cls._instance._initialized = False
                cls._instance.request_queue = queue.Queue()
                cls._instance.max_concurrent_requests = int(os.getenv("MAX_CONCURRENT_REQUESTS", "5"))
                cls._instance.request_semaphore = asyncio.Semaphore(cls._instance.max_concurrent_requests)
                cls._instance.thread_pool = ThreadPoolExecutor(
                    max_workers=cls._instance.max_concurrent_requests,
                    thread_name_prefix="qa_worker"
                )
        return cls._instance

    def initialize_qa_system(self):
        """Initialize the QA system"""
        if self._initialized:
            logger.info("QA system already initialized, skipping...")
            return

        try:
            logger.info("Starting QA system initialization...")
            
            # Get OpenAI API key
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")

            # Initialize document directory
            docs_dir = os.getenv("DOCUMENTS_DIR", "documents")
            os.makedirs(docs_dir, exist_ok=True)

            # Initialize QA system without document watcher
            self.qa_system = EnhancedDocumentQASystem(
                documents_dir=docs_dir,
                openai_api_key=openai_api_key,
                reset_db=False,
                enable_watcher=False
            )

            # Update status
            self.system_status = "ready"
            self.document_count = len(self.qa_system.document_metadata)
            self._initialized = True
            logger.info("QA system initialization complete")

        except Exception as e:
            logger.error(f"Initialization error: {str(e)}")
            self.error_message = str(e)
            self.system_status = "error"
            self._initialized = False

    async def process_question(self, question: str) -> Dict:
        """Process a question using the thread pool to prevent blocking"""
        async with self.request_semaphore:
            try:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    self.thread_pool,
                    partial(self.qa_system.ask_question, question)
                )
            except Exception as e:
                logger.error(f"Error processing question: {str(e)}")
                raise

    def shutdown(self):
        """Cleanup resources"""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)

# Create state instance
state = AppState()

@app.on_event("startup")
async def startup_event():
    """Initialize the QA system on startup"""
    state.initialize_qa_system()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown"""
    state.shutdown()

@app.get("/")
def home():
    active_requests = state.thread_pool._work_queue.qsize()
    status_color = "green" if state.system_status == "ready" else "red"
    
    return Title("EI-Hub RAG"), Main(
        H1("EI-Hub Retrieval-Augmented Generation Search"),
        P("This RAG search is designed to assist in searching and retrieving information from EI-Hub and PCG documentation."),
        P(f"System Status: ", Span(state.system_status, style=f"color: {status_color}")),
        P(f"Active Requests: {active_requests}/{state.max_concurrent_requests}"),
        Form(
            Input(id="new-question", name="question", placeholder="Enter question"),
            Button("Search", disabled=state.system_status != "ready"),
            enctype="multipart/form-data",
            hx_post="/respond",
            hx_target="#result"
        ),
        Br(), Div(id="result"),
        cls="container"
    )

@app.post("/respond")
async def handle_question(question: str):
    """Handle question submission"""
    try:
        if not state.qa_system or state.system_status != "ready":
            return {"error": f"System not ready. Status: {state.system_status}. {state.error_message}"}

        # Process question asynchronously
        response = await state.process_question(question)
        print(response)
        return Main(
            P(response['answer']),
            P("\nConfidence: ", response['confidence']),
            P("\nSources: ", response['sources']),
        )
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    # Run with multiple workers
    uvicorn.run(
        "main:app", 
        host='127.0.0.1', 
        port=int(os.getenv("PORT", default=5000)), 
        reload=False,
        workers=int(os.getenv("WORKERS", "4"))  # Number of worker processes
    )