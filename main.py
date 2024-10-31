from fasthtml.common import *
from pathlib import Path
import os, uvicorn
from dotenv import load_dotenv
import logging
from datetime import datetime

# Import your existing QA system
from enhanced_qa_system import EnhancedDocumentQASystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastHTML app
app = FastHTML(hdrs=(picolink,))

# Global variables for system state
qa_system = None
system_status = "initializing"
error_message = ""
document_count = 0

def initialize_qa_system():
    """Initialize the QA system"""
    global qa_system, system_status, error_message, document_count
    
    try:
        logger.info("Starting QA system initialization...")
        
        # Get OpenAI API key
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        # Initialize document directory
        docs_dir = os.getenv("DOCUMENTS_DIR", "documents")
        os.makedirs(docs_dir, exist_ok=True)

        # Initialize QA system
        qa_system = EnhancedDocumentQASystem(
            documents_dir=docs_dir,
            openai_api_key=openai_api_key,
            reset_db=False
        )

        # Update status
        system_status = "ready"
        document_count = len(qa_system.document_metadata)
        logger.info("QA system initialization complete")

    except Exception as e:
        logger.error(f"Initialization error: {str(e)}")
        error_message = str(e)
        system_status = "error"

@app.route("/")
def index():
    """Render the main page"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Document QA System</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-100">
        <div class="container mx-auto px-4 py-8 max-w-2xl">
            <h1 class="text-3xl font-bold mb-4">Document QA System</h1>
            
            <!-- Status Display -->
            <div id="status" class="text-sm mb-4 {status_color}">
                Status: {status}
                {document_count_display}
            </div>

            <!-- Error Display -->
            {error_section}

            <!-- Question Form -->
            <div class="bg-white rounded-lg shadow-md p-6 mb-6">
                <label class="block text-sm font-medium text-gray-700 mb-2">
                    Your Question
                </label>
                <textarea
                    id="question"
                    class="w-full p-2 border rounded-md mb-4"
                    rows="3"
                    placeholder="Enter your question here..."
                ></textarea>
                <button
                    id="submit"
                    class="bg-blue-500 text-white px-4 py-2 rounded-md hover:bg-blue-600"
                    onclick="submitQuestion()"
                >
                    Ask Question
                </button>
            </div>

            <!-- Answer Display -->
            <div id="answer-section" class="bg-white rounded-lg shadow-md p-6 hidden">
                <h2 class="text-xl font-semibold mb-4">Answer</h2>
                <div id="answer-text" class="text-gray-700 mb-4"></div>
                
                <div class="mt-4">
                    <h3 class="text-lg font-medium mb-2">Sources</h3>
                    <div id="sources" class="text-sm text-gray-600"></div>
                </div>
                
                <div id="confidence" class="text-sm text-gray-600 mt-4"></div>
            </div>
        </div>

        <script>
            async function submitQuestion() {
                const questionEl = document.getElementById('question');
                const submitBtn = document.getElementById('submit');
                const errorEl = document.getElementById('error');
                const answerSection = document.getElementById('answer-section');
                
                try {
                    submitBtn.disabled = true;
                    submitBtn.textContent = 'Processing...';
                    if (errorEl) errorEl.classList.add('hidden');
                    
                    const response = await fetch('/ask', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ question: questionEl.value }),
                    });
                    
                    const data = await response.json();
                    
                    if (data.error) {
                        if (errorEl) {
                            errorEl.textContent = data.error;
                            errorEl.classList.remove('hidden');
                        }
                        return;
                    }
                    
                    document.getElementById('answer-text').textContent = data.answer;
                    document.getElementById('confidence').textContent = 
                        `Confidence Score: ${(data.confidence * 100).toFixed(1)}%`;
                    
                    const sourcesHtml = data.sources.map(source => 
                        `<div class="mb-1">• ${source.source} ${source.page !== 'N/A' ? `(Page: ${source.page})` : ''}</div>`
                    ).join('');
                    document.getElementById('sources').innerHTML = sourcesHtml;
                    
                    answerSection.classList.remove('hidden');
                    
                } catch (error) {
                    if (errorEl) {
                        errorEl.textContent = 'Error processing question: ' + error.message;
                        errorEl.classList.remove('hidden');
                    }
                } finally {
                    submitBtn.disabled = false;
                    submitBtn.textContent = 'Ask Question';
                }
            }
        </script>
    </body>
    </html>
    """.format(
        status=system_status,
        status_color="text-green-600" if system_status == "ready" else "text-yellow-600",
        document_count_display=f" | Documents: {document_count}" if system_status == "ready" else "",
        error_section=f'<div id="error" class="text-red-600 text-sm mb-4">{error_message}</div>' if error_message else ""
    )

@app.route("/ask", methods=["POST"])
def handle_question(request):
    """Handle question submission"""
    try:
        question = request.json.get("question", "").strip()
        
        if not question:
            return {"error": "Please enter a question"}

        if not qa_system:
            return {"error": "System not ready. Please wait..."}

        # Process question
        response = qa_system.ask_question(question)
        
        return {
            "answer": response['answer'],
            "confidence": response['confidence'],
            "sources": response['sources']
        }
        
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    # Initialize QA system before starting the server
    initialize_qa_system()
    
    # Run the FastHTML app
    uvicorn.run("main:app", host='127.0.0.1', port=int(os.getenv("PORT", default=5000)), reload=True)