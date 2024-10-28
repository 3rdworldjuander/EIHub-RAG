"""
Enhanced Document QA System
Requirements:
pip install langchain langchain-community langchain-openai python-dotenv watchdog pandas chromadb pypdf unstructured openpyxl tqdm
"""

import os
import pandas as pd
import logging
import threading
import time
from typing import List, Dict, Optional
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from queue import Queue
from dataclasses import dataclass

from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ConfidenceMetrics:
    """Stores confidence metrics for a response."""
    score: float
    source_count: int
    newest_source_date: Optional[datetime]
    context_relevance: float
    
    def overall_confidence(self) -> float:
        """Calculate overall confidence score."""
        weights = {
            'score': 0.4,
            'source_count': 0.2,
            'recency': 0.2,
            'relevance': 0.2
        }
        
        recency_score = 1.0
        if self.newest_source_date:
            days_old = (datetime.now() - self.newest_source_date).days
            recency_score = max(0.1, min(1.0, 1 - (days_old / 365)))
            
        return (
            weights['score'] * self.score +
            weights['source_count'] * min(1.0, self.source_count / 3) +
            weights['recency'] * recency_score +
            weights['relevance'] * self.context_relevance
        )

class DocumentWatcher(FileSystemEventHandler):
    """Watches for document changes and triggers updates."""
    
    def __init__(self, qa_system):
        self.qa_system = qa_system
        self.update_queue = Queue()
        self._start_update_worker()

    def _start_update_worker(self):
        def worker():
            while True:
                file_path = self.update_queue.get()
                if file_path is None:
                    break
                self.qa_system.update_document(file_path)
                self.update_queue.task_done()
                
        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def on_modified(self, event):
        if not event.is_directory:
            self.update_queue.put(event.src_path)

    def on_created(self, event):
        if not event.is_directory:
            self.update_queue.put(event.src_path)

class EnhancedDocumentQASystem:
    def __init__(self, documents_dir: str, openai_api_key: str):
        """Initialize the Enhanced Document QA System."""
        self.documents_dir = documents_dir
        self.openai_api_key = openai_api_key
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key,
            chunk_size=400
        )
        self.vector_store = None
        self.document_metadata = {}
        self.confidence_threshold = 0.7
        
        # Process initial documents
        self.process_documents()
        
        # Initialize document watcher
        self._setup_document_watcher()

    def _setup_document_watcher(self):
        """Set up the document watcher for real-time updates."""
        self.event_handler = DocumentWatcher(self)
        self.observer = Observer()
        self.observer.schedule(self.event_handler, self.documents_dir, recursive=True)
        self.observer.start()

    def load_excel_as_text(self, file_path: str) -> List[Dict]:
        """Custom loader for Excel files that converts them to text."""
        try:
            df = pd.read_excel(file_path, sheet_name=None)
            documents = []
            
            for sheet_name, sheet_df in df.items():
                text = f"Sheet: {sheet_name}\n\n"
                sheet_df = sheet_df.astype(str).replace('nan', '')
                text += "Columns: " + ", ".join(sheet_df.columns) + "\n\n"
                text += sheet_df.to_string(index=False)
                
                doc = {
                    "page_content": text,
                    "metadata": {
                        "source": Path(file_path).name,
                        "sheet_name": sheet_name,
                        "date_processed": datetime.now().isoformat()
                    }
                }
                documents.append(doc)
            
            return documents
        except Exception as e:
            logger.error(f"Error processing Excel file {file_path}: {str(e)}")
            return []

    def _load_single_document(self, file_path: str) -> List[Dict]:
        """Load a single document based on its file type."""
        try:
            file_path = Path(file_path)
            if file_path.suffix.lower() == '.pdf':
                loader = PyPDFLoader(str(file_path))
                docs = loader.load()
            elif file_path.suffix.lower() in ['.doc', '.docx']:
                loader = UnstructuredWordDocumentLoader(str(file_path))
                docs = loader.load()
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                docs = self.load_excel_as_text(str(file_path))
            else:
                return []
            
            # Add metadata
            for doc in docs:
                if isinstance(doc, dict):
                    if 'metadata' not in doc:
                        doc['metadata'] = {}
                    doc['metadata'].update({
                        'source': file_path.name,
                        'date_processed': datetime.now().isoformat()
                    })
                else:
                    doc.metadata.update({
                        'source': file_path.name,
                        'date_processed': datetime.now().isoformat()
                    })
            
            return docs
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            return []

    def process_documents(self):
        """Process all documents in the directory with batch processing."""
        logger.info("Starting initial document processing...")
        
        documents = []
        # Walk through all files in the directory
        for file_path in Path(self.documents_dir).rglob('*'):
            if file_path.is_file():
                docs = self._load_single_document(str(file_path))
                if docs:
                    documents.extend(docs)
                    self.document_metadata[str(file_path)] = {
                        "last_updated": datetime.now(),
                        "chunks": len(docs)
                    }
        
        if documents:
            # Process documents
            processed_docs = self._process_documents(documents)
            
            # Process in smaller batches
            batch_size = 500  # Reduced batch size
            
            # Initialize vector store with first batch
            first_batch = processed_docs[:batch_size]
            self.vector_store = Chroma.from_documents(
                documents=first_batch,
                embedding=self.embeddings,
                persist_directory="./chroma_db"
            )
            
            # Add remaining documents in batches
            remaining_docs = processed_docs[batch_size:]
            for i in tqdm(range(0, len(remaining_docs), batch_size), desc="Processing document batches"):
                batch = remaining_docs[i:i + batch_size]
                try:
                    self.vector_store.add_documents(batch)
                except Exception as e:
                    logger.error(f"Error processing batch: {str(e)}")
                    continue
            
            logger.info(f"Processed {len(documents)} documents into {len(processed_docs)} chunks")
        else:
            logger.warning("No documents found to process")
            # Initialize empty vector store
            self.vector_store = Chroma(
                embedding_function=self.embeddings,
                persist_directory="./chroma_db"
            )

    def _process_documents(self, documents: List[Document]) -> List[Document]:
        """Process documents with enhanced chunking and metadata."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Reduced chunk size
            chunk_overlap=50,  # Reduced overlap
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""],
            is_separator_regex=False
        )
        
        processed_docs = []
        for doc in documents:
            # Convert to Document if necessary
            if isinstance(doc, dict):
                doc = Document(
                    page_content=doc['page_content'],
                    metadata=doc['metadata']
                )
            
            try:
                # Split into chunks
                chunks = text_splitter.split_text(doc.page_content)
                
                # Create documents with enhanced metadata
                for i, chunk in enumerate(chunks):
                    processed_docs.append(Document(
                        page_content=chunk,
                        metadata={
                            **doc.metadata,
                            'chunk_id': i + 1,
                            'total_chunks': len(chunks),
                            'chunk_size': len(chunk)
                        }
                    ))
            except Exception as e:
                logger.error(f"Error processing document {doc.metadata.get('source', 'unknown')}: {str(e)}")
                continue
        
        return processed_docs

    def update_document(self, file_path: str):
        """Update a single document in the vector store."""
        try:
            docs = self._load_single_document(file_path)
            if docs:
                # Remove old document entries if they exist
                if str(file_path) in self.document_metadata:
                    self.vector_store.delete(
                        where={"source": Path(file_path).name}
                    )
                
                # Process and add new document
                processed_docs = self._process_documents(docs)
                
                # Add documents in smaller batches
                batch_size = 500
                for i in range(0, len(processed_docs), batch_size):
                    batch = processed_docs[i:i + batch_size]
                    try:
                        self.vector_store.add_documents(batch)
                    except Exception as e:
                        logger.error(f"Error adding batch to vector store: {str(e)}")
                        continue
                
                # Update metadata
                self.document_metadata[str(file_path)] = {
                    "last_updated": datetime.now(),
                    "chunks": len(processed_docs)
                }
                
                logger.info(f"Updated document: {file_path}")
                
        except Exception as e:
            logger.error(f"Error updating document {file_path}: {str(e)}")

    def _calculate_confidence(
        self,
        question: str,
        answer: str,
        source_docs: List[Document]
    ) -> ConfidenceMetrics:
        """Calculate confidence metrics for the response."""
        # Basic confidence calculation based on source documents
        source_count = len(set(doc.metadata['source'] for doc in source_docs))
        
        # Get the newest source date
        newest_date = None
        for doc in source_docs:
            if 'date_processed' in doc.metadata:
                doc_date = datetime.fromisoformat(doc.metadata['date_processed'])
                if newest_date is None or doc_date > newest_date:
                    newest_date = doc_date
        
        # Simple relevance score based on source count
        relevance_score = min(1.0, source_count / 3)
        
        return ConfidenceMetrics(
            score=0.8,  # Base confidence score
            source_count=source_count,
            newest_source_date=newest_date,
            context_relevance=relevance_score
        )

    def create_qa_chain(self):
        """Create an enhanced question-answering chain."""
        llm = ChatOpenAI(
            model_name="gpt-4",
            temperature=0,
            openai_api_key=self.openai_api_key
        )
        
        template = """Use the following pieces of context to answer the question. 

        Context: {context}

        Question: {question}

        Instructions:
        1. Analyze all provided context thoroughly
        2. If you find a direct answer, quote the relevant text and cite the source
        3. If you can only find partial information, clearly state what information is available
        4. If you need to make any assumptions or inferences, explicitly state them
        5. Include confidence level and reasoning for your answer
        6. Always cite the specific source documents used
        
        Format your response as follows:
        Answer: [Your detailed answer]
        Sources: [List of source documents used]
        Confidence: [High/Medium/Low] - [Explanation of confidence level]
        Assumptions (if any): [List any assumptions or inferences made]
        
        Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        return RetrievalQA.from_chain_type(
            llm=llm,
            retriever=self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 6,
                    "fetch_k": 10,
                    "lambda_mult": 0.7
                }
            ),
            chain_type_kwargs={
                "prompt": prompt,
                "verbose": True
            },
            return_source_documents=True
        )

    def ask_question(self, question: str) -> Dict:
        """
        Ask a question and get an enhanced answer with confidence metrics.
        
        Returns:
            Dict containing answer, confidence metrics, and source information
        """
        if not self.vector_store:
            raise ValueError("Please process documents first")
        
        qa_chain = self.create_qa_chain()
        response = qa_chain.invoke({"query": question})
        
        # Calculate confidence metrics
        confidence_metrics = self._calculate_confidence(
            question,
            response['result'],
            response['source_documents']
        )
        
        # Format response
        sources = [
            {
                'source': doc.metadata['source'],
                'chunk_id': doc.metadata.get('chunk_id', 'N/A'),
                'date_processed': doc.metadata.get('date_processed', 'N/A')
            }
            for doc in response['source_documents']
        ]
        
        return {
            'answer': response['result'],
            'confidence': confidence_metrics.overall_confidence(),
            'sources': sources,
            'is_inference': confidence_metrics.overall_confidence() < self.confidence_threshold
        }

def main():
    from dotenv import load_dotenv
    load_dotenv()
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("Please set OPENAI_API_KEY in your environment variables or .env file")
    
    docs_dir = "documents"
    
    # Ensure the documents directory exists
    os.makedirs(docs_dir, exist_ok=True)
    
    qa_system = EnhancedDocumentQASystem(docs_dir, openai_api_key)
    
    print("\nDocument QA System initialized and ready for questions!")
    print("(Place your documents in the 'documents' directory)")
    
    while True:
        question = input("\nEnter your question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
            
        try:
            response = qa_system.ask_question(question)
            
            print("\nAnswer:", response['answer'])
            print("\nConfidence Score:", f"{response['confidence']:.2f}")
            
            if response['is_inference']:
                print("\nNote: This response includes inferences based on available information.")
            
            print("\nSources:")
            for source in response['sources']:
                print(f"- {source['source']} (Processed: {source['date_processed']})")
                
        except Exception as e:
            print(f"Error: {str(e)}")
            logger.error(f"Error processing question: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()