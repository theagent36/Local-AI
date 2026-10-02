import streamlit as st
import tempfile
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from document_processor import DocumentProcessor

# Configure the visual layout of the page
st.set_page_config(page_title="Local AI Document & Notes Agent", layout="wide", page_icon="📝")
st.title("📝 Local Document & Handwritten Notes Chat Agent")
st.markdown("Powered by Llama 3.2, Moondream Vision, and ChromaDB. 100% local and private.")

# Initialize the Streamlit memory state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "retrieval_chain" not in st.session_state:
    st.session_state.retrieval_chain = None
if "extracted_docs_preview" not in st.session_state:
    st.session_state.extracted_docs_preview = None

# --- SIDEBAR: FILE UPLOAD AND PROCESSING ---
with st.sidebar:
    st.header("Upload Document or Notes")
    uploaded_file = st.file_uploader(
        "Upload a PDF or Image of handwritten notes",
        type=["pdf", "png", "jpg", "jpeg", "webp"]
    )
    
    st.subheader("⚙️ Vision & Processing Settings")
    model_choice = st.selectbox(
        "Vision / Handwriting Model",
        options=["moondream", "llama3.2-vision", "Custom"],
        index=0,
        help="Moondream is pre-installed, lightweight, and verified for image and handwriting transcription."
    )
    
    if model_choice == "Custom":
        selected_vision_model = st.text_input("Enter Ollama model name", value="minicpm-v")
    else:
        selected_vision_model = model_choice

    ocr_mode = st.radio(
        "PDF OCR Mode",
        options=["Auto-detect handwritten pages", "Force Vision OCR on all pages"],
        index=0,
        help="Auto-detect uses fast text extraction for digital text and automatically switches to Vision AI for scanned/handwritten pages."
    )
    force_ocr = (ocr_mode == "Force Vision OCR on all pages")

    process_btn = st.button("🚀 Process Document / Notes", type="primary")

    if process_btn and uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name

        progress_bar = st.progress(0)
        status_box = st.empty()

        def update_progress(current: int, total: int, msg: str):
            progress_bar.progress(current / total)
            status_box.info(f"⏳ [{current}/{total}] {msg}")

        try:
            with st.spinner("Analyzing document and reading handwritten notes..."):
                processor = DocumentProcessor(vision_model=selected_vision_model)
                docs = processor.process_file(
                    file_bytes,
                    file_name=file_name,
                    force_ocr=force_ocr,
                    progress_callback=update_progress,
                )

                if not docs or all(not d.page_content.strip() for d in docs):
                    st.error("No text could be extracted from the document. Please check the file or model.")
                else:
                    status_box.text("✂️ Splitting document into searchable chunks...")
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                    chunks = text_splitter.split_documents(docs)

                    status_box.text("🧠 Creating embeddings and vector index...")
                    embeddings = OllamaEmbeddings(model="nomic-embed-text")
                    vector_store = Chroma.from_documents(chunks, embeddings)
                    retriever = vector_store.as_retriever(search_kwargs={"k": min(7, max(2, len(chunks)))})

                    # Setup LLM and Memory Prompts
                    llm = ChatOllama(model="llama3.2")

                    contextualize_q_prompt = ChatPromptTemplate.from_messages([
                        ("system", "Given a chat history and the latest user question, formulate a standalone question. Do NOT answer the question, just reformulate it if needed."),
                        MessagesPlaceholder(variable_name="chat_history"),
                        ("human", "{input}"),
                    ])
                    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

                    qa_prompt = ChatPromptTemplate.from_messages([
                        ("system", "You are an intelligent assistant helping the user read, understand, and analyze their documents and handwritten notes. Answer based ONLY on the provided context.\n\n<context>\n{context}\n</context>"),
                        MessagesPlaceholder(variable_name="chat_history"),
                        ("human", "{input}"),
                    ])

                    document_chain = create_stuff_documents_chain(llm, qa_prompt)
                    st.session_state.retrieval_chain = create_retrieval_chain(history_aware_retriever, document_chain)
                    
                    # Store preview of extracted notes
                    st.session_state.extracted_docs_preview = docs

                    progress_bar.progress(1.0)
                    status_box.empty()
                    st.success(f"✅ Ready! Processed {len(docs)} page(s) and {len(chunks)} chunks.")

        except Exception as e:
            status_box.empty()
            st.error(f"❌ Error processing document: {str(e)}")

    # Show preview of extracted text/notes if available
    if st.session_state.extracted_docs_preview:
        with st.expander("📝 View Extracted Notes Preview"):
            for doc in st.session_state.extracted_docs_preview:
                page_info = f"Page {doc.metadata.get('page', 1)}" if 'page' in doc.metadata else "Document"
                doc_type = doc.metadata.get('type', 'general')
                st.markdown(f"**{page_info}** `({doc_type})`:")
                st.text_area(label=f"Content - {page_info}", value=doc.page_content, height=150, disabled=True)

# --- MAIN PAGE: CHAT INTERFACE ---

# Display all previous messages on the screen
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(msg.content)

# The chat input box at the bottom of the screen
user_query = st.chat_input("Ask a question about your document or handwritten notes...")

if user_query:
    if st.session_state.retrieval_chain is None:
        st.warning("Please upload and process a document or handwritten note in the sidebar first!")
    else:
        # Show the user's message immediately
        with st.chat_message("user"):
            st.write(user_query)
        
        # Pass the query and history to the local AI
        with st.chat_message("assistant"):
            with st.spinner("Searching notes and generating answer..."):
                response = st.session_state.retrieval_chain.invoke({
                    "input": user_query,
                    "chat_history": st.session_state.chat_history
                })
                st.write(response["answer"])
                
        # Save both messages to memory for the next loop
        st.session_state.chat_history.extend([
            HumanMessage(content=user_query),
            AIMessage(content=response["answer"])
        ])