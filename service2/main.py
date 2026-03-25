import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
from contextlib import AsyncExitStack
from mcp.client.session import ClientSession
from typing import Optional
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os
from dotenv import load_dotenv
from typing import Optional, List 
from asgiref.sync import async_to_sync, sync_to_async

import logging
from openai import AsyncOpenAI

from llm_jailbreaking_defense import DefendedTargetLM, SelfReminderConfig, BacktranslationConfig, load_defense, TargetLM


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
        """
        Connect to MCP server
        Args: server_script_path: Path to the server script
        """
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


    async def send_message_to_llm(self, messages: list[dict]):
        """
        helper function for sending data to llm
        """
        logger.debug(f"-------------------------------------------Sending request to LLM with messages: {messages}")
        
        logger.debug(messages)

        response = await client_ai.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=1000,
            messages=messages,
            tools=self.formatted_tools if self.formatted_tools else None
        )

        return response

    async def process_query(self, query: str):
        """
        process current query using available tools
        """

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that can use external tools when necessary."
            }

        ]

        messages.append({"role":"user", "content": query})        
        
        response = await client.send_message_to_llm(messages)
        logger.debug(f"-------------------------------------------------\n{response}")

        for i in range(10):
            logger.debug(f"loop running {i}")
            message = response.choices[0].message
            
            messages.append(message) 
            if not message.tool_calls:
                break

            tool_call = message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            
            # call tool
            tool_result = await self.session.call_tool(tool_name, tool_args)
            logger.debug(f"*******************************tool call {tool_result.content} ***********************************")
            
            # result.content může být list objektů, převedeme na string
            if isinstance(tool_result.content, list) and len(tool_result.content) > 0:
                extracted_text = tool_result.content[0].text
            else:
                extracted_text = str(tool_result.content)
            
            messages.append(
                {
                    "role":"tool",
                    "tool_call_id": tool_call.id,
                    "content": extracted_text
                }
            )



            response = await client.send_message_to_llm(messages)

        return response.choices[0].message.content if response.choices else "Max tool iterations reached."            

#help class for defense library, but it is not used in code
class SimpleTemplate:
    system_message: str = "You are a helpful assistant."
    user_message: str = "{prompt}"

# MCP Adapter for DefendeTargetLM
class MCPAdapter(TargetLM):
    def __init__(self, mcp_client):
        self.mcp_client = mcp_client
        
        self.template = SimpleTemplate()

    def get_response(self, prompts_list, **kwargs):
    
        prompt = prompts_list[0] if isinstance(prompts_list, list) else prompts_list
        
    
        safe_sync_call = async_to_sync(self.mcp_client.process_query)
        
        response_text = safe_sync_call(prompt)
        
    
        return [response_text]

    def evaluate_log_likelihood(self, prompt, response):
        return 0

# INITIALIZE MCP CLIENT ON STARTUP
@app.on_event("startup")
async def startup_event():
    global client, defended_client
    client = MCPClient(MCP_SERVER_SCRIPT)
    await client.start()
    
    # Adapter for defence 
    mcp_adapter = MCPAdapter(client)

    config = BacktranslationConfig()
    defense = load_defense(config)

    defended_client = DefendedTargetLM(mcp_adapter, defense)

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    try:
         async_client = sync_to_async(defended_client.get-response)
         answer = await async_client([request.query])
         return {"query": request.query, "answer": answer[0]}
    except Exception as e:
        logger.exception("Error processing query")
        raise HTTPException(status_code=500, detail=str(e))




