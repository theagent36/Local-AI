import os
import sys
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from document_processor import DocumentProcessor

def main():
    print("=" * 60)
    print(" Local Document & Handwritten Notes RAG Agent")
    print("=" * 60)

    # Prompt user for file path or use sample.pdf / CLI argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        default_file = "sample.pdf"
        file_path = input(f"Enter path to document/handwritten notes (default: {default_file}): ").strip()
        if not file_path:
            file_path = default_file

    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return

    print(f"\n[1/4] Processing '{file_path}' (extracting text & handwriting)...")
    processor = DocumentProcessor(vision_model="moondream")
    
    def on_progress(current, total, msg):
        print(f"  -> [{current}/{total}] {msg}")

    try:
        docs = processor.process_file(
            file_path,
            file_name=os.path.basename(file_path),
            force_ocr=False,
            progress_callback=on_progress,
        )
    except Exception as e:
        print(f"Error during document processing: {e}")
        return

    print(f"  Loaded {len(docs)} document section(s).")

    print("\n[2/4] Splitting text into search chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    print(f"  Created {len(chunks)} chunk(s).")

    print("\n[3/4] Creating embeddings and vector index in ChromaDB...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = Chroma.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": min(5, max(1, len(chunks)))})

    print("\n[4/4] Setting up history-aware retrieval chain with Llama 3.2...")
    llm = ChatOllama(model="llama3.2")

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the question based ONLY on the provided context below.\n\n<context>\n{context}\n</context>"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    document_chain = create_stuff_documents_chain(llm, qa_prompt)
    retrieval_chain = create_retrieval_chain(history_aware_retriever, document_chain)

    print("\n" + "=" * 60)
    print(" Agent is ready! You can now ask questions about the document/notes.")
    print(" (Type 'exit' to quit)")
    print("=" * 60)

    chat_history = []

    while True:
        try:
            user_question = input("\nAsk a question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not user_question:
            continue
        if user_question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        print("Searching notes and generating answer...")
        response = retrieval_chain.invoke({
            "input": user_question,
            "chat_history": chat_history,
        })

        print(f"\nAnswer: {response['answer']}")

        chat_history.extend([
            HumanMessage(content=user_question),
            AIMessage(content=response["answer"]),
        ])

if __name__ == "__main__":
    main()