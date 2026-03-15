from fastapi import FastAPI, HTTPException 
from security import jailGuard 
from llm_client import MCPClient 
import logging 
from pydantic import BaseModel 
import nltk
nltk.download("punkt_tab")
nltk.download('stopwords')

class QueryRequest(BaseModel): 
    query: str 
app = FastAPI() 
client = MCPClient("/app/mcp_server.py") 

@app.on_event("startup") 
async def startup_event(): 
    await client.start() 
@app.post("/query") 
async def query_endpoint(request: QueryRequest): 
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