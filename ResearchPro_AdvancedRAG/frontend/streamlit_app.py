import streamlit as st
import requests
import uuid
from typing import Optional
import io

# Page configuration
st.set_page_config(
    page_title="Advanced Research Assistant RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://127.0.0.1:8000"  

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False

if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []

if 'uploaded_filename' not in st.session_state:
    st.session_state.uploaded_filename = ""

# Helper functions
def upload_file_to_api(file_data, filename):
    try:
        files = {"file": (filename, file_data, "application/pdf")}
        response = requests.post(f"{API_BASE_URL}/upload_file", files=files)
        return response
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {str(e)}")
        return None



def query_api(query_text, session_id):
    """Send query to FastAPI endpoint"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={"query": query_text,"session_id": session_id}
        )
        return response
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {str(e)}")
        return None

def delete_vectorstore():
    """Delete vectorstore via API"""
    try:
        response = requests.delete(f"{API_BASE_URL}/delete")
        return response
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {str(e)}")
        return None

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/")
        return response.status_code == 200
    except:
        return False



# Sidebar for file upload and controls
with st.sidebar:
    st.title("📚 Research Assistant")
    st.markdown("---")
    
    # API Status Check
    if check_api_health():
        st.success("🟢 API Connected")
    else:
        st.error("🔴 API Disconnected")
        st.warning("Please make sure your FastAPI server is active")
    
    
    # File upload section
    st.subheader("Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a research paper or document to analyze",
        key='file_uploader'
    )
    
    if uploaded_file is not None:
        if st.button("Process Document", type="primary"):
            with st.spinner("Uploading and processing document..."):
                # Read file data
                file_data = uploaded_file.read()
                
                # Upload to API
                response = upload_file_to_api(file_data, uploaded_file.name)
                
                if response and response.status_code == 200:
                    st.session_state.file_uploaded = True
                    st.session_state.uploaded_filename = uploaded_file.name
                    st.rerun()
                    st.success(f"✅ Document ready: {st.session_state.uploaded_filename}")
                elif response:
                    st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
                else:
                    st.error("Failed to connect to API")
        
    
    st.markdown("---")
    
    # Session management
    st.subheader("Chat Session")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("New Chat", help="Start a new conversation"):
            st.session_state.chat_messages = []
            st.rerun()
    
    with col2:
        if st.button("Clear All", help="Clear chat and reset"):
            st.session_state.chat_messages = []
            if st.session_state.file_uploaded:
                with st.spinner("Clearing vectorstore..."):
                    delete_response = delete_vectorstore()
                    if delete_response and delete_response.status_code == 200:
                        st.session_state.file_uploaded = False
                        st.session_state.uploaded_filename = ""
            st.rerun()
    
    # Display session info
    st.caption(f"Session: {st.session_state.session_id[:8]}...")
    
    # Reset document
    if st.session_state.file_uploaded:
        st.markdown("---")
        if st.button("Reset/Delete Vectorstore", 
                     help="Upload a new document"):
            delete_response = delete_vectorstore()
            if delete_response and delete_response.status_code == 200:
                st.session_state.file_uploaded = False
                st.session_state.uploaded_filename = ""
                st.session_state.chat_messages = []
                st.rerun()
    
    st.markdown("---")
    st.text_input("API URL", value=API_BASE_URL, key="api_url", help="FastAPI server URL")

# Main chat interface
st.title("Academia Multimodal RAG Assistant")

if not st.session_state.file_uploaded:
    st.info("Please upload a PDF document in the sidebar to get started!")
    st.markdown("""
    ### How to use:
    1. **Upload a PDF** research paper or document using the sidebar
    2. **Click "Process Document"** to upload it to the server
    3. **Ask questions** about the content once processing is complete
    4. **Continue the conversation** - the assistant remembers context
    5. **Start new chats** or reset as needed
    """)
else:
    # Chat container
    chat_container = st.container()
    
    # Display chat messages
    with chat_container:
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your document..."):
        # Add user message to chat
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            with st.spinner("Thinking..."):
                # Query the API
                response = query_api(prompt, st.session_state.session_id)
                
                if response and response.status_code == 200:
                    response_data = response.json()
                    assistant_response = response_data.get("response", "No response received")
                    
                    # Display response
                    message_placeholder.markdown(assistant_response)
                    
                    # Add assistant response to chat
                    st.session_state.chat_messages.append({
                        "role": "assistant", 
                        "content": assistant_response
                    })
                elif response:
                    error_data = response.json()
                    error_message = f"Error: {error_data.get('detail', 'Unknown error')}"
                    message_placeholder.markdown(error_message)
                    st.session_state.chat_messages.append({
                        "role": "assistant", 
                        "content": error_message
                    })
                else:
                    error_message = "Failed to connect to API"
                    message_placeholder.markdown(error_message)
                    st.session_state.chat_messages.append({
                        "role": "assistant", 
                        "content": error_message
                    })
    
    # Show helpful tips
    if len(st.session_state.chat_messages) == 0:
        st.markdown("""
        ### Try asking questions like:
        - "What is the main topic of this document?"
        - "Can you summarize the key findings?"
        - "What methodology was used in this research?"
        - "Are there any limitations mentioned?"
        - "Can you explain the results in simple terms?"
        """)
        
        # Show document info
        st.info(f"Current document: **{st.session_state.uploaded_filename}**")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888;'>"
    "🔬 Research Assistant RAG System | API Version | Built with FastAPI"
    "</div>", 
    unsafe_allow_html=True
)