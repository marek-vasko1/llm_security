import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
from contextlib import AsyncExitStack
from mcp.client.session import ClientSession
import asyncio
from typing import Optional
from contextlib import AsyncExitStack
import json
from security import sanitize_prompt
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os
from dotenv import load_dotenv

import logging
from openai import OpenAI

from llm_jailbreaking_defense import DefendedTargetLM, SelfReminderConfig, BacktranslationConfig, load_defense, TargetLM


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

load_dotenv()

api_key_check = os.getenv("API_KEY")
url_check = os.getenv("BASE_URL")


client_ai = OpenAI(
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
               
        # recive tools
        response = await self.session.list_tools()
        available_tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]
        
        # Conversion of tools into the correct JSON format for the e-infra API
        formatted_tools = [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool.get("input_schema", {"type": "object", "properties": {}})
            }
        }
        for tool in available_tools
        
        ]

       

        response = client_ai.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=1000,
            messages=messages,
            tools=formatted_tools
        )

        if (response.choices[0].message.content is not None and sanitize_prompt(response.choices[0].message.content) == None):
            return None

        result=response.choices[0].message.content

        while True:

            message = response.choices[0].message

            if not message.tool_calls:
                break

            tool_call = message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args = tool_call.function.arguments
            tool_args = json.loads(tool_call.function.arguments)
            
            # call tool
            result = await self.session.call_tool(tool_name, tool_args)

            
            messages.append(message)
            messages.append({
                "role":"user",
                "content": result.content
            })
             
            response = client_ai.chat.completions.create(
                model=LLM_MODEL,
                max_tokens=1000,
                messages=messages,
                tools=formatted_tools
            )
            
            
            result = response.choices[0].message.content if response.choices else "No response"

        return result


# INITIALIZE MCP CLIENT ON STARTUP


# MCP Adapter pro DefendedTargetLM

class MCPAdapter(TargetLM):
    def __init__(self, mcp_client):
        self.mcp_client = mcp_client
        self.loop = asyncio.get_event_loop()
        class SimpleTemplate:
            def __init__(self):
                self.system_message = "You are a helpful assistant."
                self.user_message = "{prompt}"
        
        self.template = SimpleTemplate()


    async def get_response(self, prompts_list, **kwargs):
        
        single_prompt = False
        if isinstance(prompts_list, str):
            prompts_list = [prompts_list]
            single_prompt = True

        results = []

        for prompt in prompts_list:
            response = await self.mcp_client.process_query(prompt)
            results.append(response)
        # If there was only one prompt originally, return only the answer
        return results[0] if single_prompt else results

    def evaluate_log_likelihood(self, prompt, response):
        return 0

@app.on_event("startup")
async def startup_event():
    global client, defended_client
    client = MCPClient(MCP_SERVER_SCRIPT)
    await client.start()
    
    # Adapter for defence 
    mcp_adapter = MCPAdapter(client)

    config = SelfReminderConfig()
    defense = load_defense(config)

    defended_client = DefendedTargetLM(mcp_adapter, defense)

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    try:
        if sanitize_prompt(request.query) is None:
            answer = "I cannot answer this question, because it may try to bypass security guards"
        else:
            answer = await defended_client.get_response(request.query)
            if answer is None or sanitize_prompt(answer) is None:
                answer = "I cannot answer this question, because it may try to bypass security guards"

        return {"query": request.query, "answer": answer}
    except Exception as e:
        logger.exception("Error processing query")
        raise HTTPException(status_code=500, detail=str(e))




