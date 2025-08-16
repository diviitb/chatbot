# app.py
import os
import streamlit as st
import tempfile
from dotenv import load_dotenv
import textwrap

# Import your custom modules
from parser import extract_pdf
from chunking import get_text_chunks
from embedding import EmbeddingHandler
from llm import get_answer_from_llm, get_suggested_questions
from utils import save_uploaded_file


# --- Setup ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="PDF RAG Chatbot", layout="wide")
st.title("📄 My Awesome PDF RAG Chatbot")

if not GOOGLE_API_KEY:
    st.error("Please set GOOGLE_API_KEY in your .env file.")
    st.stop()

# --- Main UI ---
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Initial Greeting
if "greeted" not in st.session_state:
    st.session_state.greeted = True
    st.success("👋 Hello! I'm your PDF RAG assistant. Upload a document to get started.")

if uploaded_file:
    # Save the file to a temporary location
    file_path = save_uploaded_file(uploaded_file)
    if not file_path:
        st.error("Failed to save the uploaded file.")
        st.stop()

    # Step 1: Parse the PDF
    with st.spinner("Extracting text, images, and tables..."):
        page_texts, images, tables = extract_pdf(file_path)

    # Check if text was extracted
    if not page_texts:
        st.error("Could not extract any text from the PDF. Please try a different document.")
        os.remove(file_path)
        st.stop()

    # Display first page overview
    st.subheader("Document Overview")
    if images and 1 in images:
        st.image(images[1][0], caption="First page overview", use_column_width=True)
    else:
        st.info("No images found on the first page to preview.")

    # Step 2: Chunk the text
    with st.spinner("Splitting text into chunks..."):
        all_chunks_with_metadata = get_text_chunks(page_texts)

    # Step 3: Create embeddings and vector store
    with st.spinner("Creating vector store and embeddings..."):
        embedding_handler = EmbeddingHandler()
        chunks_without_metadata = [chunk['text'] for chunk in all_chunks_with_metadata]
        try:
            embedding_handler.create_embeddings(chunks_without_metadata)
        except ValueError as e:
            st.error(str(e))
            os.remove(file_path)
            st.stop()

    st.success("✅ Document processed successfully!")

    # User options: Summarize or Ask Question
    st.markdown("---")
    option = st.radio(
        "What would you like to do?",
        ("Summarize the PDF", "Ask a question from the PDF"),
    )

    if option == "Summarize the PDF":
        if st.button("Generate Summary"):
            with st.spinner("Generating summary..."):
                summary_text = "\n\n".join(
                    [chunk["text"] for chunk in all_chunks_with_metadata[:5]]
                )
                summary_prompt = "Summarize the following document:\n\n" + summary_text
                summary_answer = get_answer_from_llm(summary_prompt, [])
                st.write("### Summary")
                st.write(summary_answer)

    elif option == "Ask a question from the PDF":
        user_query = st.text_input("Enter your question here:")

        if st.button("Get Answer") and user_query:
            with st.spinner("Retrieving relevant information..."):
                retrieved_chunks_info = embedding_handler.search(user_query, top_k=5)

                # Extract the text and get page numbers
                contexts = [chunk_text for chunk_text, _ in retrieved_chunks_info]
                chunk_map = {chunk["text"]: chunk["page"] for chunk in all_chunks_with_metadata}
                pages = [chunk_map[chunk_text] for chunk_text in contexts if chunk_text in chunk_map]

            with st.spinner("Generating response..."):
                try:
                    answer = get_answer_from_llm(user_query, contexts)

                    st.write("### Answer")
                    st.write(answer)

                    # Display related images if they exist on the pages found
                    for page in sorted(list(set(pages))):
                        if page in images:
                            st.write("---")
                            st.image(images[page][0], caption=f"Image from Page {page}", use_column_width=True)

                    # Display suggested questions
                    st.subheader("💡 People also ask")
                    suggestions = get_suggested_questions(user_query, contexts)
                    for s in suggestions:
                        st.markdown(f"- {s}")

                except Exception as e:
                    st.error(f"An error occurred during LLM generation: {e}")

    # Clean up the temporary file
    os.remove(file_path)