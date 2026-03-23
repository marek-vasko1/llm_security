import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mcp.client.session import ClientSession
from typing import Optional
from contextlib import AsyncExitStack
import json
from security import sanitize_prompt
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os
from dotenv import load_dotenv
from typing import Optional, List

import logging
from openai import AsyncOpenAI


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

load_dotenv()

api_key_check = os.getenv("API_KEY")
url_check = os.getenv("BASE_URL")


client_ai = AsyncOpenAI(
    api_key=api_key_check,
    base_url=url_check
)

logging.basicConfig(level=logging.DEBUG)
MCP_SERVER_SCRIPT = "/app/mcp_server.py"
LLM_MODEL = "gpt-oss-120b"




#DEFINE API
app = FastAPI(title="MCP_Gateway_API")

class QueryRequest(BaseModel):
    query: str

class MCPClient:
    def __init__(self, server_script_path):
        self.server_script_path = server_script_path
        self.exit_stack = AsyncExitStack()
        self._lock = asyncio.Lock()
        self.formatted_tools: List[dict] = []

    async def start(self):
    #Connect to MCP server
    #Args: server_script_path: Path to the server script
    
        command = "python"
        server_params = StdioServerParameters(
            command=command,
            args=[self.server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()
        
        #recive tools
        response = await self.session.list_tools()

        # Conversion of tools into the correct JSON format for the e-infra API
        self.formatted_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                    }
                }
                for tool in response.tools
            ]

    async def process_query(self, query: str):
        """
        process current query using available tools
        """

        messages = [
            {
            "role": "system",
            "content": "You are a helpful assistant that can use external tools when necessary."
            },
            {
            "role":"user",
            "content":query
            }

        ]
              

        response = await client_ai.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=1000,
            messages=messages,
            tools=self.formatted_tools if self.formatted_tools else None
        )

        result=response.choices[0].message.content

        for i in range(10):

            message = response.choices[0].message

            if not message.tool_calls:
                break

            tool_call = message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args = tool_call.function.arguments
            tool_args = json.loads(tool_call.function.arguments)

            #call tool
            tool_result = await self.session.call_tool(tool_name, tool_args)
            
            if isinstance(tool_result.content, list) and len(tool_result.content) > 0:
                extracted_text = tool_result.content[0].text
            else:
                extracted_text = str(tool_result.content)
            
            messages.append(message)
            messages.append({
                "role":"tool",
                "tool_call_id": tool_call.id,
                "content": extracted_text
            })
             
            response = await client_ai.chat.completions.create(
                model=LLM_MODEL,
                max_tokens=1000,
                messages=messages,
                tools=self.formatted_tools if self.formatted_tools else None
            )
            
            
        return response.choices[0].message.content if response.choices else "Max tool iterations reached."


# INITIALIZE MCP CLIENT ON STARTUP

@app.on_event("startup")
async def startup_event():
    global client
    client = MCPClient(MCP_SERVER_SCRIPT)
    await client.start()

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    try:
        status = sanitize_prompt(request.query)
        logger.debug(status)
        logger.debug("INFO-------------------------------------")
        if (status > 0.6):
              return {"query": request.query, "answer": "I cannot answer this question, because it may try to bypass security guards"}
        else:
            answer = await client.process_query(request.query)

            if (answer is None):
                answer = "I cannot answer this question, because it may try to bypass security guards"
            

        return {"query": request.query, "answer": answer}
    except Exception as e:
        logging.info(f"tady exc {e}")
        raise HTTPException(status_code=500, detail=str(e))






