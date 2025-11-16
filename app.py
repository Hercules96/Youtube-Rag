import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from youtube_transcript_api import YouTubeTranscriptApi
import os
import re

# Page configuration
st.set_page_config(
    page_title="RAG Q&A Application",
    page_icon="🤖",
    layout="wide"
)

# Title and description
st.title("🤖 RAG-based Question Answering System")
st.markdown("Ask questions based on YouTube video transcripts or uploaded documents!")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key input
    api_key = st.text_input(
        "OpenRouter API Key",
        type="password",
        help="Enter your OpenRouter API key (sk-or-v1-...)"
    )
    
    # Model selection
    model_choice = st.selectbox(
        "Select Model",
        ["openai/gpt-4o-mini", "openai/gpt-3.5-turbo", "openai/gpt-4o"],
        help="Choose the LLM model for answering questions"
    )
    
    # Temperature slider
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1,
        help="Controls randomness in responses"
    )
    
    st.divider()
    st.markdown("### About")
    st.info("This app uses RAG (Retrieval Augmented Generation) to answer questions based on provided context.")

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Main content area with tabs
tab1, tab2, tab3 = st.tabs(["📹 YouTube Video", "📄 Upload Document", "💬 Chat"])

# Tab 1: YouTube Video Input
with tab1:
    st.header("Extract Transcript from YouTube")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        url1 = st.text_input(
            "YouTube Video url",
            placeholder="e.g., SwQhKFMxmDY",
            help="Enter video url")
            
    def get_youtube_id(url1):
        match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url1)
        return match.group(1) if match else None
    if url1:
        video_id = get_youtube_id(url1) 

    
    
    with col2:
        st.write("")
        st.write("")
        process_video = st.button("🔍 Process Video", use_container_width=True)
    
    if process_video and video_id and api_key:
        with st.spinner("Fetching transcript and creating embeddings..."):
            try:
                # Fetch transcript
                yt_api = YouTubeTranscriptApi()
                transcript_list = yt_api.fetch(video_id, languages=["en"])
                transcript = " ".join(chunk.text for chunk in transcript_list)
                
                # Split text
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                chunks = text_splitter.create_documents([transcript])
                
                # Create embeddings
                embeddings = OpenAIEmbeddings(
                    model="openai/text-embedding-3-small",
                    openai_api_base="https://openrouter.ai/api/v1",
                    openai_api_key=api_key
                )
                
                # Create vector store
                st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)
                
                st.success(f"✅ Successfully processed video! Created {len(chunks)} chunks.")
                st.info(f"📝 Transcript preview: {transcript[:300]}...")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    elif process_video and not api_key:
        st.warning("⚠️ Please enter your API key in the sidebar!")

# Tab 2: Document Upload
with tab2:
    st.header("Upload Your Document")
    
    uploaded_file = st.file_uploader(
        "Choose a text file",
        type=['txt'],
        help="Upload a document to create embeddings"
    )
    
    process_doc = st.button("📤 Process Document", use_container_width=True)
    
    if process_doc and uploaded_file and api_key:
        with st.spinner("Processing document..."):
            try:
                # Read file content
                text = uploaded_file.read().decode("utf-8")
                
                # Split text
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                chunks = text_splitter.create_documents([text])
                
                # Create embeddings
                embeddings = OpenAIEmbeddings(
                    model="openai/text-embedding-3-small",
                    openai_api_base="https://openrouter.ai/api/v1",
                    openai_api_key=api_key
                )
                
                # Create vector store
                st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)
                
                st.success(f"✅ Successfully processed document! Created {len(chunks)} chunks.")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    elif process_doc and not api_key:
        st.warning("⚠️ Please enter your API key in the sidebar!")

# Tab 3: Chat Interface
with tab3:
    st.header("Ask Questions")
    
    if st.session_state.vector_store is None:
        st.info("👆 Please process a YouTube video or upload a document first!")
    else:
        # Display chat history
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                st.write(answer)
        
        # Question input
        question = st.chat_input("Ask a question about the content...")
        
        if question and api_key:
            # Add user message to chat
            with st.chat_message("user"):
                st.write(question)
            
            # Generate answer
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Setup LLM
                        llm = ChatOpenAI(
                            model=model_choice,
                            temperature=temperature,
                            openai_api_base="https://openrouter.ai/api/v1",
                            openai_api_key=api_key,
                            default_headers={
                                "HTTP-Referer": "http://localhost:8501",
                                "X-Title": "Streamlit RAG App"
                            }
                        )
                        
                        # Create prompt template
                        template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, say that you don't know. 
Use three sentences maximum and keep the answer concise.

Context: {context}

Question: {question}

Answer:"""
                        
                        prompt = ChatPromptTemplate.from_template(template)
                        
                        # Format documents function
                        def format_docs(docs):
                            return "\n\n".join(doc.page_content for doc in docs)
                        
                        # Create retrieval chain using LCEL
                        retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                        
                        rag_chain = (
                            {
                                "context": retriever | format_docs,
                                "question": RunnablePassthrough()
                            }
                            | prompt
                            | llm
                            | StrOutputParser()
                        )
                        
                        # Get answer
                        answer = rag_chain.invoke(question)
                        
                        st.write(answer)
                        
                        # Show source documents
                        source_docs = retriever.invoke(question)
                        if source_docs:
                            with st.expander("📚 View Source Context"):
                                for i, doc in enumerate(source_docs):
                                    st.markdown(f"**Source {i+1}:**")
                                    st.text(doc.page_content[:300] + "...")
                                    st.divider()
                        
                        # Add to chat history
                        st.session_state.chat_history.append((question, answer))
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    Built with Streamlit 🎈 | Powered by LangChain 🦜 | OpenRouter API 🔑
    </div>
    """,
    unsafe_allow_html=True
)