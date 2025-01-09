from fastapi import FastAPI, HTTPException
from backend.models.retriever import DocumentRetriever
from pydantic import BaseModel 
from backend.models.generator import Generator  # Updated import

# Define request model
class QueryRequest(BaseModel):
    query: str


class QueryHandler:
    def __init__(self):
        """
        Initializes the retriever for the search pipeline.
        """
        self.retriever = DocumentRetriever()
        # We don't need the generator anymore since we're just doing retrieval

    def handle_query(self, query: str) -> str:
        """
        Processes the query using retrieval.
        """
        # Get top 3 relevant documents with their metadata and similarity scores
        results = self.retriever.retrieve(query, top_k=3)
        
        # Format the response
        response = "Here are the most relevant results:\n\n"
        for i, result in enumerate(results, 1):
            response += f"Result {i} (Similarity: {result['similarity']:.2f}):\n"
            metadata = result['metadata']
            response += f"Title: {metadata['title']}\n"
            response += f"Location: {metadata['location']}\n"
            if metadata['address']:
                response += f"Address: {metadata['address']}\n"
            response += f"\n{result['text']}\n\n"
            
        return response

app = FastAPI()
query_handler = QueryHandler()

@app.post("/query")
async def query(request: QueryRequest):
    """
    API endpoint to handle user queries.
    """
    if not request.query:
        raise HTTPException(status_code=400, detail="Query is required.")
    try:
        response = query_handler.handle_query(request.query)
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))