import sys
import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader
import os

# Workaround for sqlite3 issue in Streamlit Cloud
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb

# Function to initialize the OpenAI client
def initialize_openai_client():
    if 'openai_api' not in st.session_state:
        # Get the API key from Streamlit secrets
        api_key = st.secrets["openai_api_key"]
        # Initialize the OpenAI client and store it in session state
        st.session_state.openai_api = OpenAI(api_key=api_key)

# Function to create a vector database collection for documents
def initialize_document_collection():
    if 'document_vector_db' not in st.session_state:
        # Set up the ChromaDB client
        db_path = os.path.join(os.getcwd(), "chroma_storage")
        db_client = chromadb.PersistentClient(path=db_path)
        document_collection = db_client.get_or_create_collection("DocumentLabCollection")

        initialize_openai_client()

        # Define the directory containing the PDF files
        pdf_folder = os.path.join(os.getcwd(), "data_lab4")
        if not os.path.exists(pdf_folder):
            st.error(f"PDF folder not found: {pdf_folder}")
            return None

        # Process each PDF file in the folder
        for pdf_file in os.listdir(pdf_folder):
            if pdf_file.endswith(".pdf"):
                pdf_path = os.path.join(pdf_folder, pdf_file)
                try:
                    # Extract text from the PDF
                    with open(pdf_path, "rb") as file:
                        pdf_reader = PdfReader(file)
                        text_content = ''.join([page.extract_text() or '' for page in pdf_reader.pages])

                    # Generate embeddings for the extracted text
                    response = st.session_state.openai_api.embeddings.create(
                        input=text_content, model="text-embedding-3-small"
                    )
                    doc_embedding = response.data[0].embedding

                    # Add the document to ChromaDB
                    document_collection.add(
                        documents=[text_content],
                        metadatas=[{"filename": pdf_file}],
                        ids=[pdf_file],
                        embeddings=[doc_embedding]
                    )
                except Exception as error:
                    st.error(f"Error processing {pdf_file}: {str(error)}")

        # Store the collection in session state
        st.session_state.document_vector_db = document_collection

    return st.session_state.document_vector_db

# Function to query the document vector database
def search_vector_db(db_collection, search_query):
    initialize_openai_client()
    try:
        # Generate embedding for the query
        response = st.session_state.openai_api.embeddings.create(
            input=search_query, model="text-embedding-3-small"
        )
        query_embedding = response.data[0].embedding

        # Query the ChromaDB collection
        search_results = db_collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        return search_results['documents'][0], [result['filename'] for result in search_results['metadatas'][0]]
    except Exception as error:
        st.error(f"Error during the query: {str(error)}")
        return [], []

# Function to generate chatbot response using GPT model
def generate_chatbot_reply(user_query, document_context):
    initialize_openai_client()
    # Construct the prompt for the GPT model
    prompt_text = f"""You are an AI assistant with knowledge from specific documents. Use the following context to answer the user's question. If the information is not in the context, say you don't know based on the available information.

Context:
{document_context}

User Question: {user_query}

Answer:"""

    try:
        # Generate streaming response using OpenAI's chat completion
        response_stream = st.session_state.openai_api.chat.completions.create(
            model="gpt-4o",  # Using the GPT-4 model
            messages=[
                {"role": "system", "content": "You are an intelligent and helpful assistant."},
                {"role": "user", "content": prompt_text}
            ],
            stream=True  # Enable streaming
        )
        return response_stream
    except Exception as error:
        st.error(f"Error getting chatbot response: {str(error)}")
        return None

# Initialize session state for chat history and system readiness
if 'chat_log' not in st.session_state:
    st.session_state.chat_log = []
if 'is_system_ready' not in st.session_state:
    st.session_state.is_system_ready = False
if 'doc_collection' not in st.session_state:
    st.session_state.doc_collection = None

# Page content
st.title("Document Chatbot for Lab 4")

# Check if the system is ready, otherwise prepare it
if not st.session_state.is_system_ready:
    # Show a spinner while processing documents
    with st.spinner("Initializing system and processing documents..."):
        st.session_state.doc_collection = initialize_document_collection()
        if st.session_state.doc_collection:
            # Set the system as ready and show a success message
            st.session_state.is_system_ready = True
            st.success("AI Assistant is ready to chat!")
        else:
            st.error("Failed to load the document collection. Please check the file path and try again.")

# Only display chat interface if the system is ready
if st.session_state.is_system_ready and st.session_state.doc_collection:
    st.subheader("Ask the AI Assistant about the documents")

    # Display chat history
    for chat_message in st.session_state.chat_log:
        if isinstance(chat_message, dict):
            # New format (dictionary with 'role' and 'content' keys)
            with st.chat_message(chat_message["role"]):
                st.markdown(chat_message["content"])
        elif isinstance(chat_message, tuple):
            # Old format (tuple with role and content)
            sender, content = chat_message
            # Convert 'You' to 'user', and assume any other role is 'assistant'
            with st.chat_message("user" if sender == "You" else "assistant"):
                st.markdown(content)

    # User input
    user_query = st.chat_input("Ask a question about the uploaded documents:")

    if user_query:
        # Display user query
        with st.chat_message("user"):
            st.markdown(user_query)

        # Query the vector database
        matched_texts, matched_docs = search_vector_db(st.session_state.doc_collection, user_query)
        doc_context = "\n".join(matched_texts)

        # Get the chatbot response
        response_stream = generate_chatbot_reply(user_query, doc_context)

        # Display AI response
        with st.chat_message("assistant"):
            response_holder = st.empty()
            complete_response = ""
            for part in response_stream:
                if part.choices[0].delta.content is not None:
                    complete_response += part.choices[0].delta.content
                    response_holder.markdown(complete_response + "▌")
            response_holder.markdown(complete_response)

        # Add messages to chat history (new format)
        st.session_state.chat_log.append({"role": "user", "content": user_query})
        st.session_state.chat_log.append({"role": "assistant", "content": complete_response})

        # Display relevant documents
        with st.expander("Documents referenced for the answer"):
            for doc in matched_docs:
                st.write(f"- {doc}")

elif not st.session_state.is_system_ready:
    st.info("Please wait while the system is being set up...")
else:
    st.error("Could not load the document collection. Please check the setup and try again.")
