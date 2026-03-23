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
from openai import OpenAI, AsyncOpenAI

from llm_jailbreaking_defense import DefendedTargetLM, load_defense, TargetLM, ParaphraseDefenseConfig


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

    async def send_message_to_llm(self, messages: list[dict]):

        logger.debug("-------------------------------------------Sending request to LLM with messages: {prompt}")
        
        logger.debug(messages)

        response = await client_ai.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=1000,
            messages=messages,
            tools=formatted_tools
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
        paraphrased_query = await defense._paraphrase(query)
        messages.append({"role":"user", "content": paraphrased_query})        
        
        response = await client.send_message_to_llm(messages)
        logger.debug("-------------------------------------------------")
        logger.debug(response)

        for i in range(10):
            logger.debug("loop running")
            message = response.choices[0].message
            
            messages.append(message) 
            if not message.tool_calls:
                break

            tool_call = message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            
            # call tool
            result = await self.session.call_tool(tool_name, tool_args)
            logger.debug("*******************************tool call {result.content} ***********************************")
            
            # result.content může být list objektů, převedeme na string
            if isinstance(result.content, list) and len(result.content) > 0:
                extracted_text = result.content[0].text
            else:
                extracted_text = str(result.content)

            paraphrased_data = await defense._paraphrase(extracted_text)
            
            messages.append(
                {
                    "role":"tool",
                    "tool_call_id": response.id,
                    "content": paraphrased_data
                }
            )



            response = await client.send_message_to_llm(messages)

            result = response.choices[0].message.content if response.choices else "No response"
        
        logger.debug("RETRURNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN")
        return result


# INITIALIZE MCP CLIENT ON STARTUP

#PARAPHRASE LLM
class ParaLLM(TargetLM):
    def __init__(self, model_name: str, api_key: str, base_url: str):
        self.model_name = model_name
        
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )

    async def get_response(self, prompts_list, **kwargs):
        
        if isinstance(prompts_list, list):
            prompt = prompts_list[0]
        else:
            prompt = prompts_list

        messages = [
                {"role": "system", "content": prompt}
            ]

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7, 
                max_tokens=1024
            )

            content = response.choices[0].message.content
            
            
            return [content]

        except Exception as e:
            return [prompt] 

    def evaluate_log_likelihood(self, prompt, response):
        return 0

@app.on_event("startup")
async def startup_event():
    global client, defense, formatted_tools
    client = MCPClient(MCP_SERVER_SCRIPT)
    await client.start()
    
    para_llm = ParaLLM(model_name="gpt-oss-120b",api_key=api_key_check,base_url=url_check)
    
    config = ParaphraseDefenseConfig(paraphrase_model="custom-direct-model")
    defense = load_defense(config, preloaded_model=para_llm)


    response = await client.session.list_tools()
    available_tools = [
        {
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

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    try:
        if sanitize_prompt(request.query) is None:
            answer = "I cannot answer this question, because it may try to bypass security guards"
        else:
            answer = await client.process_query(request.query)
            if answer is None or sanitize_prompt(answer) is None:
                answer = "I cannot answer this question, because it may try to bypass security guards"

        return {"query": request.query, "answer": answer}
    except Exception as e:
        logger.exception("Error processing query")
        raise HTTPException(status_code=500, detail=str(e))




