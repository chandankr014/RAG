# Fix OpenMP runtime conflict
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from flask import Flask, request, jsonify, render_template, session
from werkzeug.utils import secure_filename
import fitz  # PyMuPDF
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import re
from typing import List, Dict, Any, Optional
import networkx as nx
from collections import defaultdict
import spacy
import datetime
import uuid

# Import configuration from config.py
from config import (
    UPLOAD_FOLDER, INDEX_FOLDER, MEMORY_FOLDER, INTERMEDIATE_RESULTS_FOLDER,
    MAX_FILE_SIZE, genai_api_key, emb_model, llm_model, CHUNK_SIZE, CHUNK_OVERLAP
)

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "chandankr014")
embedding_model = GoogleGenerativeAIEmbeddings(model=emb_model)

# Initialize spaCy for entity extraction
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("[WARNING] spaCy model not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None

def hash_bytes(content):
    return hashlib.md5(content).hexdigest()

def save_intermediate_result(session_id: str, technique: str, step: str, data: Dict[str, Any]):
    """Save intermediate results to a structured folder."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"{timestamp}_{step}.json"
    
    # Create session folder
    session_folder = os.path.join(INTERMEDIATE_RESULTS_FOLDER, session_id)
    technique_folder = os.path.join(session_folder, technique)
    os.makedirs(technique_folder, exist_ok=True)
    
    # Save the data
    filepath = os.path.join(technique_folder, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.datetime.now().isoformat(),
            "technique": technique,
            "step": step,
            "data": data
        }, f, indent=2, ensure_ascii=False)
    
    print(f"[INTERMEDIATE] Saved {technique}/{step} to {filepath}")
    return filepath

def get_qa_chain(strict=True):
    template = (
        """
        Answer the question based on the given context only.
        If the answer is not found, reply with: "Answer is not available in the context."
        Context: {context}
        Question: {question}
        Answer:
        """
        if strict
        else """
        The answer is not in the context, but based on relevant content, here's the closest answer:
        Context: {context}
        Question: {question}
        Answer:
        """
    )
    temp = 0.3 if strict else 0.8
    llm = ChatGoogleGenerativeAI(model=llm_model, temperature=temp)
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return load_qa_chain(llm, chain_type="stuff", prompt=prompt)

# ===== TECHNIQUE 1: LLM-POWERED DECOMPOSITION =====
def decompose_question(question: str, session_id: str) -> List[str]:
    """Break complex queries into simpler sub-questions using LLM."""
    decomposition_prompt = PromptTemplate(
        template="""
        Break this complex query into 2-5 simpler, sequential sub-questions that need to be answered to fully address the original question.
        Each sub-question should be specific and focused on one aspect.
        Return only the sub-questions, one per line, without numbering or additional text.
        
        Complex Query: {question}
        
        Sub-questions:
        """,
        input_variables=["question"]
    )
    
    llm = ChatGoogleGenerativeAI(model=llm_model, temperature=0.3)
    chain = decomposition_prompt | llm
    
    try:
        response = chain.invoke({"question": question})
        sub_questions = [q.strip() for q in response.content.split('\n') if q.strip()]
        sub_questions = sub_questions[:5]  # Limit to 5 sub-questions
        
        # Save intermediate result
        save_intermediate_result(session_id, "decomposition", "question_decomposition", {
            "original_question": question,
            "sub_questions": sub_questions,
            "llm_response": response.content,
            "count": len(sub_questions)
        })
        
        return sub_questions
    except Exception as e:
        print(f"[WARNING] Decomposition failed: {e}")
        save_intermediate_result(session_id, "decomposition", "decomposition_error", {
            "original_question": question,
            "error": str(e)
        })
        return [question]  # Fallback to original question

def answer_with_decomposition(question: str, faiss_index, filename: str, session_id: str) -> Dict[str, Any]:
    """Answer using LLM-powered decomposition technique."""
    sub_questions = decompose_question(question, session_id)
    print(f"[INFO] Decomposed into {len(sub_questions)} sub-questions")
    
    all_answers = []
    all_sources = set()
    sub_question_results = []
    
    for i, sub_q in enumerate(sub_questions):
        print(f"[INFO] Processing sub-question {i+1}: {sub_q}")
        
        # Get embeddings and retrieve docs for sub-question
        sub_embedding = embedding_model.embed_query(sub_q)
        docs = faiss_index.similarity_search_by_vector(sub_embedding, k=3)
        
        # Save intermediate result for each sub-question
        sub_result = {
            "sub_question_index": i + 1,
            "sub_question": sub_q,
            "retrieved_docs_count": len(docs),
            "retrieved_docs": [
                {
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata
                } for doc in docs
            ]
        }
        
        if docs:
            # Answer sub-question
            chain = get_qa_chain(strict=True)
            response = chain({"input_documents": docs, "question": sub_q})
            answer = response.get("output_text", "").strip()
            
            if "not available in the context" in answer.lower():
                chain = get_qa_chain(strict=False)
                response = chain({"input_documents": docs, "question": sub_q})
                answer = response.get("output_text", "").strip()
            
            sub_result["answer"] = answer
            sub_result["used_strict_mode"] = "not available in the context" not in answer.lower()
            
            all_answers.append(f"Sub-question {i+1}: {sub_q}\nAnswer: {answer}")
            
            # Collect sources
            for doc in docs:
                source = f"{doc.metadata.get('source', filename)} - Page {doc.metadata.get('page', '?')}, Chunk {doc.metadata.get('chunk', '?')}"
                all_sources.add(source)
        else:
            sub_result["answer"] = "No relevant documents found"
            sub_result["used_strict_mode"] = False
        
        sub_question_results.append(sub_result)
        save_intermediate_result(session_id, "decomposition", f"sub_question_{i+1}_result", sub_result)
    
    # Synthesize final answer
    synthesis_prompt = PromptTemplate(
        template="""
        Synthesize a comprehensive answer to the original question based on the answers to the sub-questions.
        Provide a coherent, well-structured response that addresses all aspects of the original question.
        
        Original Question: {original_question}
        
        Sub-question Answers:
        {sub_answers}
        
        Comprehensive Answer:
        """,
        input_variables=["original_question", "sub_answers"]
    )
    
    llm = ChatGoogleGenerativeAI(model=llm_model, temperature=0.3)
    chain = synthesis_prompt | llm
    
    try:
        response = chain.invoke({
            "original_question": question,
            "sub_answers": "\n\n".join(all_answers)
        })
        final_answer = response.content.strip()
        
        # Save synthesis result
        save_intermediate_result(session_id, "decomposition", "final_synthesis", {
            "original_question": question,
            "sub_question_results": sub_question_results,
            "synthesis_prompt": synthesis_prompt.template,
            "llm_response": response.content,
            "final_answer": final_answer,
            "all_sources": list(all_sources)
        })
        
    except Exception as e:
        print(f"[WARNING] Synthesis failed: {e}")
        final_answer = "\n\n".join(all_answers)
        save_intermediate_result(session_id, "decomposition", "synthesis_error", {
            "original_question": question,
            "sub_question_results": sub_question_results,
            "error": str(e),
            "fallback_answer": final_answer
        })
    
    return {
        "answer": final_answer.replace("\n\n", "<br><br>").replace("\n", "<br>"),
        "sources": list(all_sources),
        "technique": "LLM Decomposition",
        "sub_questions": sub_questions,
        "intermediate_results": {
            "session_id": session_id,
            "sub_question_results": sub_question_results
        }
    }

# ===== TECHNIQUE 2: CHAIN-OF-THOUGHT PROMPTING =====
def get_cot_chain():
    """Create a Chain-of-Thought QA chain."""
    cot_template = """
    You are a helpful assistant that thinks through questions step by step.
    
    Context: {context}
    Question: {question}
    
    Let's approach this step by step:
    1. First, let me understand what information is available in the context
    2. Then, I'll identify the key points relevant to the question
    3. Finally, I'll synthesize an answer based on this reasoning
    
    Step-by-step reasoning:
    """
    
    llm = ChatGoogleGenerativeAI(model=llm_model, temperature=0.4)
    prompt = PromptTemplate(template=cot_template, input_variables=["context", "question"])
    return load_qa_chain(llm, chain_type="stuff", prompt=prompt)

def answer_with_cot(question: str, faiss_index, filename: str, session_id: str) -> Dict[str, Any]:
    """Answer using Chain-of-Thought prompting technique."""
    # Get more documents for CoT reasoning
    question_embedding = embedding_model.embed_query(question)
    docs = faiss_index.similarity_search_by_vector(question_embedding, k=8)
    
    # Save retrieval result
    save_intermediate_result(session_id, "cot", "document_retrieval", {
        "question": question,
        "retrieved_docs_count": len(docs),
        "retrieved_docs": [
            {
                "content": doc.page_content[:200] + "...",
                "metadata": doc.metadata
            } for doc in docs
        ]
    })
    
    if not docs:
        save_intermediate_result(session_id, "cot", "no_documents_found", {
            "question": question,
            "error": "No relevant documents found"
        })
        return {"error": "No relevant documents found"}
    
    # Use CoT chain
    chain = get_cot_chain()
    response = chain({"input_documents": docs, "question": question})
    output = response.get("output_text", "").strip()
    
    # Save CoT reasoning result - Fixed: Get prompt template from the chain's prompt
    cot_prompt_template = """
    You are a helpful assistant that thinks through questions step by step.
    
    Context: {context}
    Question: {question}
    
    Let's approach this step by step:
    1. First, let me understand what information is available in the context
    2. Then, I'll identify the key points relevant to the question
    3. Finally, I'll synthesize an answer based on this reasoning
    
    Step-by-step reasoning:
    """
    
    save_intermediate_result(session_id, "cot", "reasoning_process", {
        "question": question,
        "context_docs": [doc.page_content for doc in docs],
        "cot_prompt": cot_prompt_template,
        "llm_response": output,
        "final_answer": output
    })
    
    # Format the output to highlight reasoning steps
    formatted_output = output.replace("Step-by-step reasoning:", "<strong>Step-by-step reasoning:</strong>")
    formatted_output = formatted_output.replace("\n\n", "<br><br>").replace("\n", "<br>")
    
    sources = list(set([
        f"{doc.metadata.get('source', filename)} - Page {doc.metadata.get('page', '?')}, Chunk {doc.metadata.get('chunk', '?')}"
        for doc in docs
    ]))
    
    return {
        "answer": formatted_output,
        "sources": sources,
        "technique": "Chain-of-Thought",
        "intermediate_results": {
            "session_id": session_id,
            "reasoning_steps": output
        }
    }

# ===== TECHNIQUE 3: GRAPH-BASED REASONING =====
class KnowledgeGraph:
    def __init__(self, file_hash: str):
        self.file_hash = file_hash
        self.graph = nx.DiGraph()
        self.entity_memory = defaultdict(list)
        self.graph_path = os.path.join(MEMORY_FOLDER, f"{file_hash}_graph.json")
        self.load_graph()
    
    def load_graph(self):
        """Load existing graph from file."""
        if os.path.exists(self.graph_path):
            try:
                with open(self.graph_path, 'r') as f:
                    data = json.load(f)
                    self.graph = nx.node_link_graph(data, directed=True)
                    self.entity_memory = defaultdict(list, data.get('entity_memory', {}))
            except Exception as e:
                print(f"[WARNING] Failed to load graph: {e}")
    
    def save_graph(self):
        """Save graph to file."""
        try:
            data = nx.node_link_data(self.graph)
            data['entity_memory'] = dict(self.entity_memory)
            with open(self.graph_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[WARNING] Failed to save graph: {e}")
    
    def extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text."""
        if not nlp:
            # Fallback: simple keyword extraction
            words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
            return list(set(words))[:5]
        
        doc = nlp(text)
        entities = [ent.text for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT']]
        return list(set(entities))
    
    def add_document_to_graph(self, doc: Document, session_id: str = None):
        """Add document content to knowledge graph."""
        text = doc.page_content
        entities = self.extract_entities(text)
        
        # Save entity extraction result
        if session_id:
            save_intermediate_result(session_id, "graph", "entity_extraction", {
                "document_content": text[:500] + "...",
                "extracted_entities": entities,
                "document_metadata": doc.metadata
            })
        
        # Add entities as nodes
        for entity in entities:
            self.graph.add_node(entity, type='entity')
            self.entity_memory[entity].append({
                'content': text[:200] + "...",
                'page': doc.metadata.get('page', '?'),
                'chunk': doc.metadata.get('chunk', '?')
            })
        
        # Add relationships between entities
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                if entity1 != entity2:
                    self.graph.add_edge(entity1, entity2, weight=1)
        
        # Add document as a node
        doc_id = f"doc_{doc.metadata.get('page', '0')}_{doc.metadata.get('chunk', '0')}"
        self.graph.add_node(doc_id, type='document', content=text[:200] + "...")
        
        # Connect document to entities
        for entity in entities:
            self.graph.add_edge(doc_id, entity, type='contains')
    
    def query_graph(self, question: str, session_id: str = None) -> List[Dict[str, Any]]:
        """Query the knowledge graph for relevant information."""
        question_entities = self.extract_entities(question)
        relevant_info = []
        
        # Save graph query result
        if session_id:
            save_intermediate_result(session_id, "graph", "graph_query", {
                "question": question,
                "question_entities": question_entities,
                "graph_nodes_count": len(self.graph.nodes),
                "graph_edges_count": len(self.graph.edges)
            })
        
        for entity in question_entities:
            if entity in self.graph:
                # Get entity information
                entity_info = self.entity_memory.get(entity, [])
                relevant_info.extend(entity_info)
                
                # Get connected entities
                neighbors = list(self.graph.neighbors(entity))
                for neighbor in neighbors[:3]:  # Limit to 3 neighbors
                    neighbor_info = self.entity_memory.get(neighbor, [])
                    relevant_info.extend(neighbor_info[:1])  # Limit to 1 entry per neighbor
        
        # Save graph traversal result
        if session_id:
            save_intermediate_result(session_id, "graph", "graph_traversal", {
                "question_entities": question_entities,
                "found_entities": [entity for entity in question_entities if entity in self.graph],
                "relevant_info_count": len(relevant_info),
                "relevant_info": relevant_info[:5]  # Save first 5 for reference
            })
        
        return relevant_info[:10]  # Limit total results

def answer_with_graph_reasoning(question: str, faiss_index, filename: str, file_hash: str, session_id: str) -> Dict[str, Any]:
    """Answer using graph-based reasoning technique."""
    # Get documents from vector search
    question_embedding = embedding_model.embed_query(question)
    docs = faiss_index.similarity_search_by_vector(question_embedding, k=5)
    
    # Save vector search result
    save_intermediate_result(session_id, "graph", "vector_search", {
        "question": question,
        "retrieved_docs_count": len(docs),
        "retrieved_docs": [
            {
                "content": doc.page_content[:200] + "...",
                "metadata": doc.metadata
            } for doc in docs
        ]
    })
    
    if not docs:
        save_intermediate_result(session_id, "graph", "no_documents_found", {
            "question": question,
            "error": "No relevant documents found"
        })
        return {"error": "No relevant documents found"}
    
    # Initialize or load knowledge graph
    kg = KnowledgeGraph(file_hash)
    
    # Add current documents to graph
    for doc in docs:
        kg.add_document_to_graph(doc, session_id)
    
    # Query graph for relevant information
    graph_info = kg.query_graph(question, session_id)
    
    # Combine vector search results with graph information
    combined_context = []
    
    # Add vector search results
    for doc in docs:
        combined_context.append(f"Document (Page {doc.metadata.get('page', '?')}): {doc.page_content}")
    
    # Add graph information
    if graph_info:
        combined_context.append("\nRelated Information from Knowledge Graph:")
        for info in graph_info:
            combined_context.append(f"- {info.get('content', '')}")
    
    # Create graph-aware prompt
    graph_prompt = PromptTemplate(
        template="""
        Answer the question using both the provided documents and the knowledge graph information.
        Consider relationships between entities and use the graph structure to provide a comprehensive answer.
        
        Documents and Graph Information:
        {context}
        
        Question: {question}
        
        Answer (considering both document content and entity relationships):
        """,
        input_variables=["context", "question"]
    )
    
    llm = ChatGoogleGenerativeAI(model=llm_model, temperature=0.3)
    chain = graph_prompt | llm
    
    try:
        response = chain.invoke({
            "context": "\n\n".join(combined_context),
            "question": question
        })
        output = response.content.strip()
        
        # Save final reasoning result
        save_intermediate_result(session_id, "graph", "final_reasoning", {
            "question": question,
            "combined_context": combined_context,
            "graph_prompt": graph_prompt.template,
            "llm_response": response.content,
            "final_answer": output
        })
        
    except Exception as e:
        print(f"[WARNING] Graph reasoning failed: {e}")
        # Fallback to regular QA
        chain = get_qa_chain(strict=False)
        response = chain({"input_documents": docs, "question": question})
        output = response.get("output_text", "").strip()
        
        save_intermediate_result(session_id, "graph", "reasoning_fallback", {
            "question": question,
            "error": str(e),
            "fallback_answer": output
        })
    
    # Save updated graph
    kg.save_graph()
    
    sources = list(set([
        f"{doc.metadata.get('source', filename)} - Page {doc.metadata.get('page', '?')}, Chunk {doc.metadata.get('chunk', '?')}"
        for doc in docs
    ]))
    
    return {
        "answer": output.replace("\n\n", "<br><br>").replace("\n", "<br>"),
        "sources": sources,
        "technique": "Graph Reasoning",
        "graph_entities": list(kg.entity_memory.keys())[:5],  # Show top 5 entities
        "intermediate_results": {
            "session_id": session_id,
            "graph_info": graph_info
        }
    }


# ------------------- ROUTES ------------------- #

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file:
        return "No file uploaded.", 400

    filename = secure_filename(file.filename)
    if not filename.lower().endswith(".pdf"):
        return "Only PDF files are allowed.", 400

    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)
    if size > MAX_FILE_SIZE:
        return "File size exceeds 25MB limit.", 400

    # Save the file first
    file_bytes = file.read()
    file_hash = hash_bytes(file_bytes)
    filepath = os.path.join(UPLOAD_FOLDER, file_hash + "_" + filename)
    with open(filepath, "wb") as f:
        f.write(file_bytes)

    index_path = os.path.join(INDEX_FOLDER, file_hash)
    print(f"[INFO] Saved file to {filepath}")

    # Per-session last uploaded file
    session["last_uploaded"] = file_hash
    session["current_filename"] = filename

    if os.path.exists(index_path):
        print("[INFO] FAISS index already exists. Skipping re-indexing.")
        return "File already uploaded and indexed."

    # --- PDF PARSING WITH CHUNKING ---
    doc = fitz.open(filepath)
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    docs = []

    def process_page(i, page):
        text = page.get_text().strip()
        if not text:
            return []
        # Split text into chunks
        chunks = splitter.split_text(text)
        return [
            Document(
                page_content=chunk,
                metadata={
                    "source": filename,
                    "page": i + 1,
                    "chunk": j + 1
                }
            )
            for j, chunk in enumerate(chunks)
        ]

    # Threaded for fast processing of all pages
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_page, i, page) for i, page in enumerate(doc)]
        for future in as_completed(futures):
            docs.extend(future.result())

    if not docs:
        os.remove(filepath)
        return "No text extracted from PDF.", 400

    print(f"[INFO] Creating FAISS index for {len(docs)} text chunks...")
    faiss_index = FAISS.from_documents(docs, embedding_model)
    faiss_index.save_local(index_path)

    print(f"[INFO] Index saved at {index_path}")
    return "Uploaded and indexed successfully."

@app.route("/ask-question/", methods=["POST"])
def ask_question():
    data = request.get_json()
    question = data.get("question")
    technique = data.get("technique", "auto")  # auto, decomposition, cot, graph, basic
    
    if not question:
        return jsonify({"error": "Question is required"}), 400

    # Per-session file pointer
    file_hash = session.get("last_uploaded")
    filename = session.get("current_filename", "unknown.pdf")
    if not file_hash:
        return jsonify({"error": "No PDF indexed for your session."}), 400

    index_path = os.path.join(INDEX_FOLDER, file_hash)
    try:
        faiss_index = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"[WARNING] Failed to load index from {index_path}: {e}")
        return jsonify({"error": "Failed to load FAISS index"}), 500

    try:
        session_id = str(uuid.uuid4()) # Generate a unique session ID
        print(f"[INFO] Starting question processing with session ID: {session_id}")
        
        if technique == "decomposition":
            result = answer_with_decomposition(question, faiss_index, filename, session_id)
        elif technique == "cot":
            result = answer_with_cot(question, faiss_index, filename, session_id)
        elif technique == "graph":
            result = answer_with_graph_reasoning(question, faiss_index, filename, file_hash, session_id)
        elif technique == "basic":
            # Basic retrieval + QA
            question_embedding = embedding_model.embed_query(question)
            docs = faiss_index.similarity_search_by_vector(question_embedding, k=5)
            
            # Save intermediate result for basic technique
            save_intermediate_result(session_id, "basic", "document_retrieval", {
                "question": question,
                "retrieved_docs_count": len(docs),
                "retrieved_docs": [
                    {
                        "content": doc.page_content[:200] + "...",
                        "metadata": doc.metadata
                    } for doc in docs
                ]
            })
            
            if not docs:
                save_intermediate_result(session_id, "basic", "no_documents_found", {
                    "question": question,
                    "error": "No documents indexed yet."
                })
                return jsonify({"error": "No documents indexed yet."}), 500
            
            chain = get_qa_chain(strict=True)
            response = chain({"input_documents": docs, "question": question})
            output = response.get("output_text", "").strip()
            
            if "not available in the context" in output.lower():
                chain = get_qa_chain(strict=False)
                response = chain({"input_documents": docs, "question": question})
                output = response.get("output_text", "").strip()
            
            # Save basic QA result - Fixed: Use output_text instead of response.content
            save_intermediate_result(session_id, "basic", "qa_result", {
                "question": question,
                "strict_mode_answer": output,
                "used_strict_mode": "not available in the context" not in output.lower(),
                "llm_response": output
            })
            
            sources = list(set([
                f"{doc.metadata.get('source', filename)} - Page {doc.metadata.get('page', '?')}, Chunk {doc.metadata.get('chunk', '?')}"
                for doc in docs
            ]))
            
            result = {
                "answer": output.replace("\n\n", "<br><br>").replace("\n", "<br>"),
                "sources": sources,
                "technique": "Basic RAG",
                "intermediate_results": {
                    "session_id": session_id
                }
            }
        else:  # auto - try all techniques in order
            # Try graph reasoning first
            result = answer_with_graph_reasoning(question, faiss_index, filename, file_hash, session_id)
            if result.get("error"):
                # Try decomposition
                result = answer_with_decomposition(question, faiss_index, filename, session_id)
                if result.get("error"):
                    # Try CoT
                    result = answer_with_cot(question, faiss_index, filename, session_id)
                    if result.get("error"):
                        # Fallback to basic
                        question_embedding = embedding_model.embed_query(question)
                        docs = faiss_index.similarity_search_by_vector(question_embedding, k=5)
                        
                        save_intermediate_result(session_id, "auto_fallback", "document_retrieval", {
                            "question": question,
                            "retrieved_docs_count": len(docs),
                            "technique_attempts": ["graph", "decomposition", "cot"]
                        })
                        
                        if not docs:
                            save_intermediate_result(session_id, "auto_fallback", "no_documents_found", {
                                "question": question,
                                "error": "No documents indexed yet."
                            })
                            return jsonify({"error": "No documents indexed yet."}), 500
                        
                        chain = get_qa_chain(strict=False)
                        response = chain({"input_documents": docs, "question": question})
                        output = response.get("output_text", "").strip()
                        
                        save_intermediate_result(session_id, "auto_fallback", "final_qa_result", {
                            "question": question,
                            "fallback_answer": output,
                            "llm_response": output
                        })
                        
                        sources = list(set([
                            f"{doc.metadata.get('source', filename)} - Page {doc.metadata.get('page', '?')}, Chunk {doc.metadata.get('chunk', '?')}"
                            for doc in docs
                        ]))
                        
                        result = {
                            "answer": output.replace("\n\n", "<br><br>").replace("\n", "<br>"),
                            "sources": sources,
                            "technique": "Basic RAG (Fallback)",
                            "intermediate_results": {
                                "session_id": session_id
                            }
                        }
        
        # Add session ID to result for reference
        result["session_id"] = session_id
        print(f"[INFO] Completed question processing for session ID: {session_id}")
        
        return jsonify(result)

    except Exception as e:
        error_msg = f"Error during answer generation: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return jsonify({"error": error_msg}), 500

@app.route("/techniques", methods=["GET"])
def get_techniques():
    """Get available RAG techniques."""
    return jsonify({
        "techniques": [
            {
                "id": "auto",
                "name": "Auto (Smart Selection)",
                "description": "Automatically selects the best technique based on question complexity"
            },
            {
                "id": "decomposition",
                "name": "LLM Decomposition",
                "description": "Breaks complex questions into simpler sub-questions and synthesizes answers"
            },
            {
                "id": "cot",
                "name": "Chain-of-Thought",
                "description": "Uses step-by-step reasoning to answer questions"
            },
            {
                "id": "graph",
                "name": "Graph Reasoning",
                "description": "Uses knowledge graph and entity relationships for reasoning"
            },
            {
                "id": "basic",
                "name": "Basic RAG",
                "description": "Standard retrieval and generation approach"
            }
        ]
    })

@app.route("/intermediate-results/<session_id>", methods=["GET"])
def get_intermediate_results(session_id):
    """Get intermediate results for a specific session."""
    session_folder = os.path.join(INTERMEDIATE_RESULTS_FOLDER, session_id)
    
    if not os.path.exists(session_folder):
        return jsonify({"error": "Session not found"}), 404
    
    results = {}
    
    # Get all technique folders
    for technique_folder in os.listdir(session_folder):
        technique_path = os.path.join(session_folder, technique_folder)
        if os.path.isdir(technique_path):
            technique_results = []
            
            # Get all JSON files in the technique folder
            for filename in sorted(os.listdir(technique_path)):
                if filename.endswith('.json'):
                    filepath = os.path.join(technique_path, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            technique_results.append(data)
                    except Exception as e:
                        print(f"[WARNING] Failed to read {filepath}: {e}")
            
            results[technique_folder] = technique_results
    
    return jsonify({
        "session_id": session_id,
        "results": results
    })

@app.route("/intermediate-results", methods=["GET"])
def list_intermediate_sessions():
    """List all available intermediate result sessions."""
    if not os.path.exists(INTERMEDIATE_RESULTS_FOLDER):
        return jsonify({"sessions": []})
    
    sessions = []
    for session_id in os.listdir(INTERMEDIATE_RESULTS_FOLDER):
        session_path = os.path.join(INTERMEDIATE_RESULTS_FOLDER, session_id)
        if os.path.isdir(session_path):
            session_info = {
                "session_id": session_id,
                "techniques": [],
                "created": None
            }
            
            # Get techniques used in this session
            for technique in os.listdir(session_path):
                technique_path = os.path.join(session_path, technique)
                if os.path.isdir(technique_path):
                    session_info["techniques"].append(technique)
                    
                    # Get creation time from first file
                    files = os.listdir(technique_path)
                    if files:
                        first_file = os.path.join(technique_path, sorted(files)[0])
                        session_info["created"] = datetime.datetime.fromtimestamp(
                            os.path.getctime(first_file)
                        ).isoformat()
            
            sessions.append(session_info)
    
    # Sort by creation time (newest first)
    sessions.sort(key=lambda x: x["created"] or "", reverse=True)
    
    return jsonify({"sessions": sessions})

if __name__ == "__main__":
    app.run(debug=True)
