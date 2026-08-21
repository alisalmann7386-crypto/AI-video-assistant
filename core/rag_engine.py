import os
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


def prepare_documents(transcript_input):
    """Converts transcript segments or raw string into LangChain Document objects

    with timestamp metadata attached.
    """
    documents = []

    # If segments list with timestamps is provided
    if isinstance(transcript_input, list) and len(transcript_input) > 0:
        for seg in transcript_input:
            text = seg.get("text", "")
            if not text.strip():
                continue
            
            # Format time strings or floats
            start = seg.get("start", "00:00")
            end = seg.get("end", "00:00")

            doc = Document(
                page_content=text,
                metadata={
                    "start": str(start),
                    "end": str(end)
                }
            )
            documents.append(doc)
    else:
        # Fallback for plain text transcript string
        text_content = transcript_input if isinstance(transcript_input, str) else ""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_text(text_content)
        for chunk in chunks:
            documents.append(Document(page_content=chunk, metadata={"start": "N/A", "end": "N/A"}))

    return documents


def build_rag_chain(transcript_input):
    """Builds the FAISS Vectorstore and constructs the LangChain RAG pipeline."""
    # 1. Prepare Documents with Metadata
    documents = prepare_documents(transcript_input)
    
    if not documents:
        raise ValueError("Transcript content is empty. Cannot build vector store.")

    # 2. Initialize Embeddings & Vectorstore
    embeddings = MistralAIEmbeddings(model="mistral-embed")
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 3. Formatter to inject timestamp metadata into context string
    def format_docs_with_timestamps(docs):
        formatted_chunks = []
        for doc in docs:
            start_time = doc.metadata.get("start", "N/A")
            end_time = doc.metadata.get("end", "N/A")
            
            if start_time != "N/A":
                formatted_chunks.append(f"[{start_time} - {end_time}]\n{doc.page_content}")
            else:
                formatted_chunks.append(f"{doc.page_content}")
                
        return "\n\n---\n\n".join(formatted_chunks)

    # 4. LLM & System Prompt
    llm = ChatMistralAI(model="mistral-small-latest", temperature=0.2)

    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an AI assistant answering questions based on video transcript excerpts.\n"
         "Each transcript snippet in the context is prefixed with its timestamp range like [MM:SS - MM:SS] or [seconds].\n\n"
         "INSTRUCTIONS:\n"
         "- Answer the user's question accurately using ONLY the provided context.\n"
         "- If the user asks WHEN or AT WHAT TIMESTAMP something was discussed, cite the timestamp range from the context snippet.\n"
         "- If the information isn't present in the context, clearly state that it's not mentioned in the video.\n\n"
         "CONTEXT:\n{context}"),
        ("human", "{question}")
    ])

    # 5. LCEL RAG Chain
    rag_chain = (
        {"context": retriever | format_docs_with_timestamps, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def ask_question(rag_chain, question: str) -> str:
    """Executes the RAG chain for a given query."""
    try:
        response = rag_chain.invoke(question)
        return response
    except Exception as e:
        return f"An error occurred while querying the video model: {str(e)}"