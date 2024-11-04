from fasthtml.common import *
from pathlib import Path
import os, uvicorn
from dotenv import load_dotenv
import logging
from datetime import datetime
import threading
from typing import Optional
import signal
import sys
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastHTML):
    """Lifecycle manager for the FastHTML application"""
    # Startup
    logger.info("Application starting up...")
    state.initialize_qa_system()
    
    yield
    
    # Shutdown
    logger.info("Application shutting down...")
    if state.qa_system:
        state.qa_system.shutdown()
    logger.info("Application shutdown complete")

# Import your existing QA system
from enhanced_qa_system import EnhancedDocumentQASystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastHTML app
app = FastHTML(hdrs=(picolink,), lifespan=lifespan)

def signal_handler(sig, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {sig}")
    logger.info("Initiating graceful shutdown...")
    
    # Cleanup QA system
    if state.qa_system:
        state.qa_system.shutdown()
    
    logger.info("Shutdown complete. Exiting...")
    sys.exit(0)

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
                enable_watcher=False  # Disable the document watcher
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

# Create state instance
state = AppState()

@app.on_event("startup")
async def startup_event():
    """Initialize the QA system on startup"""
    state.initialize_qa_system()

@app.get("/")
def home():
    status_color = "green" if state.system_status == "ready" else "red"
    return Title("EI-Hub RAG"), Main(
        H1("EI-Hub Retrieval-Augmented Generation Search"),
        P("This RAG search is designed to assist in searching and retrieving information from EI-Hub and PCG documentation."),
        P(f"System Status: ", Span(state.system_status, style=f"color: {status_color}")),
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
def handle_question(question: str):
    """Handle question submission"""
    try:
        if not state.qa_system or state.system_status != "ready":
            return {"error": f"System not ready. Status: {state.system_status}. {state.error_message}"}

        # Process question
        response = state.qa_system.ask_question(question)
        
        return Main(
            P(response['answer']),
            P("\nConfidence: ", response['confidence']),
            P("\nSources: ", response['sources']),
        )
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Handle termination
    
    # Run the FastHTML app
    try:
        uvicorn.run(
            "main:app",
            host='127.0.0.1',
            port=int(os.getenv("PORT", default=5000)),
            reload=False,
            log_level="info",
            access_log=True
        )
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        if state.qa_system:
            state.qa_system.shutdown()
        sys.exit(1)