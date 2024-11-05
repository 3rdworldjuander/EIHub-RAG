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
import json
import uuid

# Import your existing QA system
from enhanced_qa_system import EnhancedDocumentQASystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# # Questions database for storing questions and answers details
# tables = database('data/quests.db').t
# quests = tables.quests
# if not quests in tables:
#     quests.create(id=int, question=str, answer=str, pk='id')
# Generation = quests.dataclass()

# Initialize FastHTML app
# app = FastHTML(hdrs=(picolink,))


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

#### Response Formatting functions ###

# Sourcees table
def generate_html(data):
    from urllib.parse import quote
    rows = []

    for item in data:
        print(item['source'])
        url_tag = quote(item['source'])
        rows.append(Tr(
            Td(   A(item['source'], href=f"https://github.com/3rdworldjuander/EIHub-RAG/blob/main/documents/{url_tag}", target="_blank")),
            Td(item['page'])
        ))

    table = Table(Thead(
        Tr(Th("Document Source"), Th("Found in Page")     )),
        Tbody(*rows),
    cls="table table-striped table-hover")

    return table
# End of Sources table

# start of answer table
def extract_sections(response):
    conf_raw = response['confidence']
    answer_raw = response['answer']
    """
    Extracts specific sections from the response text.
    Returns a dictionary containing the extracted sections.
    """
    sections = {
        'answer': '',
        'sources': '',
        'confidence': '',
        'assumptions': ''
    }
    
    # Split the text into sections
    try:
        # Extract Answer (everything between "Answer:" and "Sources:")
        answer_start = answer_raw.find('Answer:') + len('Answer:')
        sources_start = answer_raw.find('Sources:')
        if answer_start != -1 and sources_start != -1:
            sections['answer'] = answer_raw[answer_start:sources_start].strip()

        # Extract Sources (everything between "Sources:" and "Confidence:")
        confidence_start = answer_raw.find('Confidence:')
        if sources_start != -1 and confidence_start != -1:
            sections['sources'] = answer_raw[sources_start + len('Sources:'):confidence_start].strip()

        # Extract Confidence (everything between "Confidence:" and "Assumptions (if any):")
        assumptions_start = answer_raw.find('Assumptions (if any):')
        if confidence_start != -1 and assumptions_start != -1:
            sections['confidence'] = answer_raw[confidence_start + len('Confidence:'):assumptions_start].strip()

        # Extract Assumptions (everything after "Assumptions (if any):")
        if assumptions_start != -1:
            sections['assumptions'] = answer_raw[assumptions_start + len('Assumptions (if any):'):].strip()

    except Exception as e:
        print(f"Error extracting sections: {e}")

    rows = []
    for section, content in sections.items():
        rows.append(Tr(
            Td(f"\n{section.upper()}:"), Td(content)
        ))

    rows.append(Tr(
            Td(f"CONFIDENCE SCORE:"), Td(f"{conf_raw*100}%")
        ))
    
    table = Table(Thead(
        Tr(H2("AI Search Results") )),
        Tbody(*rows), 
    cls="table table-striped table-hover")

    return table
 
# End of answer table
#### Response Formatting functions ###

#### Website Header Information Formatting functions ###
def FastHTML_Gallery_Standard_HDRS():
    return (
        # franken.Theme.blue.headers(),
        #Script(defer=True, data_domain="gallery.fastht.ml", src="https://plausible-analytics-ce-production-dba0.up.railway.app/js/script.js"),
        HighlightJS(langs=['python', 'javascript', 'html', 'css']),
        MarkdownJS(),)

app, rt = fast_app(hdrs=FastHTML_Gallery_Standard_HDRS())

readme = """
**Limitations:**
- This tool is provided "as is" without any warranties, either express or implied
- The accuracy of responses depends entirely on the source documentation provided
- The AI may occasionally:
   - Misinterpret questions or context
   - Provide incomplete or inaccurate responses
   - Generate incorrect confidence scores
Always verify critical information against official documentation

**Source Documentation**

- All responses are based on available PCG and EI-Hub documentation. All the documents referred are consolidated in the documents folder of this repository.
- The bot's knowledge is limited to the documents it has been trained on
- Citations are provided to help users locate original source material
- Source documentation may become outdated; users should verify against current official documentation

**Best Practices**

- Treat AI responses as preliminary guidance, not authoritative answers
- Always verify critical information against official documentation
- Use the provided confidence scores and citations to assess reliability
- Report any inconsistencies or errors to improve the system

**Liability**

- This tool is for informational purposes only
- The creators and contributors assume no responsibility for decisions made based on the bot's outputs
- Users are solely responsible for verifying information before implementation
- This tool is not a replacement for official support channels or documentation
"""

notice = """ *Note: This AI search bot is an independent project and is not officially affiliated with or endorsed by PCG or BEI. For official support, please use authorized support channels.* """

def mk_button(show):
    return Button("Hide" if show else "Readme",
        get="toggle?show=" + ("False" if show else "True"),
        hx_target="#readme", id="toggle", hx_swap_oob="outerHTML",
        cls='uk-button uk-button-primary')



#### Website Header Information Formatting functions ###


@app.on_event("startup")
async def startup_event():
    """Initialize the QA system on startup"""
    state.initialize_qa_system()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown"""
    state.shutdown()

@app.get("/")
def home(session):
    active_requests = state.thread_pool._work_queue.qsize()
    status_color = "green" if state.system_status == "ready" else "red"
    if 'session_id' not in session: session['session_id'] = str(uuid.uuid4())
    inp = Input(id="new-question", name="question", placeholder="Enter a question")
    question_div = Form(Group(inp, Button("Search")), hx_post="/respond", target_id='result', hx_swap="afterbegin")
    response_div = Div(id='result')
    hiding_content = Div(mk_button(False), Div(id="readme"))
    return Title("EI-Hub RAG"), Main(
        H1("EI-Hub Retrieval-Augmented Generation Search"),
        H3("This RAG search is designed to assist in searching and retrieving information from EI-Hub and PCG documentation."),
        # P("Note: This AI search bot is an independent project and is not officially affiliated with or endorsed by PCG or EI-Hub. For official support, please use authorized support channels."),
        # A("Click here to get tips for getting better results") #https://github.com/3rdworldjuander/EIHub-RAG/blob/main/TIPS.md
        # A("Click here to know more about the project") # https://github.com/3rdworldjuander/EIHub-RAG/blob/main/README.md

        # P(f"System Status: ", Span(state.system_status, style=f"color: {status_color}")),
        # P(f"Session ID: {session['session_id']}"),
        # P(f"Active Requests: {active_requests}/{state.max_concurrent_requests}"),
        Div(notice, cls='marked'), 
        hiding_content,
        Br(),
        question_div,



        # Form(
        #     Input(id="new-question", name="question", placeholder="Enter question"),
        #     Button("Search", disabled=state.system_status != "ready"),
        #     enctype="multipart/form-data",
        #     hx_post="/respond",
        #     hx_target="#result"
        # ),
        Br(), 
        
        
        response_div,
        
        # Div(id="result"),
        cls="container"
    )

@rt('/toggle', name='toggle')
def get(show: bool):
    return Div(
            Div(mk_button(show)),
            Div(readme if show else '', cls='marked'))

@app.post("/respond")
async def handle_question(question: str, session):
    """Handle question submission"""
    # q = quests.insert(Generation(question=question, session=str(session)))
    try:
        if not state.qa_system or state.system_status != "ready":
            return {"error": f"System not ready. Status: {state.system_status}. {state.error_message}"}

        # Process question asynchronously
        response = await state.process_question(question)

        # Create Response htmls

        is_inf_raw = response['is_inference']

        # Create Sources table
        answer_html = extract_sections(response)
        source_html = generate_html(response['sources'])

        answer = Main(
            answer_html,

            source_html,
             
            P(f"Session ID: {session['session_id']}"), cls='container'
        )

        return answer
        
    
        

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