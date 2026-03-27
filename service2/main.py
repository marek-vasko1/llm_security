from fastapi import FastAPI, HTTPException 
from security import jailGuard 
from llm_client import MCPClient 
import logging 
from pydantic import BaseModel 
import nltk
nltk.download("punkt_tab")
nltk.download('stopwords')

class QueryRequest(BaseModel): 
    """
    API request model.

    Attributes:
        query (str): User query.
    """

    query: str 

app = FastAPI() 


@app.on_event("startup") 
async def startup_event():
    global client
    client = MCPClient("/app/mcp_server.py")
    await client.start()

@app.post("/query") 
async def query_endpoint(request: QueryRequest): 
    """
    Main API endpoint for processing queries.

    Workflow:
        1. Process query via MCP + LLM
        2. Validate input (via JailGuard)
        3. Process input and return response

    Args:
        request (QueryRequest): Incoming request.

    Returns:
        dict:
            query (str): Original query
            answer (str): Response or blocked message
    """

    try: #check JailGuard 
        if not await jailGuard(request.query, client):
            answer = await client.process_query(request.query) 
            if answer == None: 
                answer = "Model return None" 
        else: 
            answer="Blocked by JailGuard"
        
        return {"query": request.query, "answer": answer} 
    except Exception as e: 
        logging.info(f"tady exc {e}") 
        raise HTTPException(status_code=500, detail=str(e))
