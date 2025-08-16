# utils.py
import tempfile
import os

def save_uploaded_file(uploaded_file):
    """
    Saves a Streamlit uploaded file to a temporary file on disk.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
            tf.write(uploaded_file.getbuffer())
            file_path = tf.name
        return file_path
    except Exception as e:
        print(f"Error saving file: {e}")
        return None