import requests
from config import BACKEND_URL

def query_backend(question: str) -> str:
    """
    Send a query to the backend API and return the response.
    """
    try:
        response = requests.post(
            BACKEND_URL,
            json={"query": question}
        )
        response.raise_for_status()
        return response.json()["answer"]
    except Exception as e:
        return f"Error: {str(e)}"