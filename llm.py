# llm.py
import os
from google.generativeai import configure, GenerativeModel

# Configure Gemini API
configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_answer_from_llm(query: str, contexts: list[str]) -> str:
    """
    Generates an answer from the Gemini LLM using the provided context.
    """
    model = GenerativeModel("gemini-1.5-pro")

    context_for_prompt = "\n\n---\n\n".join(contexts)
    prompt = (
        "You are a helpful assistant. Answer ONLY using the provided context. "
        "If the answer cannot be found in the context, say 'I don't know based on the document.'\n\n"
        f"Question: {query}\n\nContext:\n{context_for_prompt}\n\n"
        "Answer concisely and mention the page numbers where you found the information."
    )

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"LLM generation failed: {e}"


def get_suggested_questions(query: str, contexts: list[str]) -> list[str]:
    """
    Generates three related questions based on the query and context.
    """
    model = GenerativeModel("gemini-1.5-pro")
    context_for_prompt = "\n".join(contexts)

    prompt = (
        f"Given the question '{query}' and the short context below, suggest 3 follow-up questions "
        f"as a numbered list.\n\nContext preview:\n{context_for_prompt}\n\nList:"
    )

    try:
        response = model.generate_content(prompt)
        # Split the response into a list of questions
        suggestions = [s.strip() for s in response.text.splitlines() if s.strip()]
        return suggestions
    except Exception as e:
        print(f"Suggestion generation failed: {e}")
        return [
            "Could you clarify your question?",
            "Do you want a summary?",
            "Which page should I prioritize?",
        ]