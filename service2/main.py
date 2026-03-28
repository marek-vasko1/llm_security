import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import AsyncExitStack

from typing import List

import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os
from dotenv import load_dotenv

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
PARAPHRASE_LLM="qwen3-coder"

async def paraphrase(prompt: str):
    """
    Jailbreak defense layer. Paraphrases user or tool inputs using a separate 
    LLM to neutralize potential malicious instructions.
    
    Inspired by: https://github.com/YihanWang617/llm-jailbreaking-defense/tree/main 28.03.2026

    Args:
        prompt (str): The text to be paraphrased.

    Returns:
        str: The sanitized and paraphrased text.
    """
    logger.debug("------------------SENDING TO PARAPHRASE---------------")
    messages = [{
        "role":"system",
        "content":"You are security model made for parahrasing inputs"
        },
        {
            "role": "user",
            "content": f"paraphrase the following paragraph: \n'{prompt}'\n\n"
        }
    ]

    response = await client_ai.chat.completions.create(
        model=PARAPHRASE_LLM,
        max_tokens=1000,
        messages=messages,
    )
    output = response.choices[0].message.content

    return output.strip().strip(']').strip('[')


#DEFINE API
app = FastAPI(title="MCP_Gateway_API")

class QueryRequest(BaseModel):
    """
    API request model.

    Attributes:
        query (str): User query.
    """

    query: str

class MCPClient:
    def __init__(self, server_script_path):
        """
        Initialize MCP client.

        Args:
            server_script_path (str): Path to MCP server script.
        """

        self.server_script_path = server_script_path
        self.exit_stack = AsyncExitStack()
        self._lock = asyncio.Lock()
        self.formatted_tools: List[dict] = []

    async def start(self):
        """
        Start MCP client and connect to MCP server.

        - Launches MCP server via stdio
        - Initializes session
        - Retrieves available tools
        - Formats tools for LLM usage

        Returns:
            None
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
        Send messages to LLM and receive response.

        Args:
            messages (list): Conversation messages.

        Returns:
            Response object from LLM API.
        """
        logger.debug(f"--------------QUESTION TO SEND----------------------\n {messages}\n")
        response = await client_ai.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=1000,
            messages=messages,
            tools=self.formatted_tools
        )

        logger.debug(f"\n recieved response:\n {response}\n")
        return response

    async def process_query(self, query: str):
        """
        Process the user query in an "Agentic Loop". 
        If the LLM requests a tool, it executes it, paraphrases the output, and sends it back to the LLM.

        Args:
            query (str): The original user query.

        Returns:
            str: The final textual response from the model.
        """

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that can use external tools when necessary."
            }

        ]
        paraphrased_query = await paraphrase(query)
        messages.append({"role":"user", "content": paraphrased_query})        
        
        response = await self.send_message_to_llm(messages)
        logger.debug("-------------------------------------------------")
        logger.debug(response)

        for i in range(10):
            logger.debug(f"loop running {i}")
            message = response.choices[0].message
            
             
            if not message.tool_calls:
                break
            
            messages.append(message)

            tool_call = message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            
            # call tool
            result = await self.session.call_tool(tool_name, tool_args)
            logger.debug("*******************************tool call {result.content} ***********************************")
            
            # result.content can be list of objects so --> string
            if isinstance(result.content, list) and len(result.content) > 0:
                extracted_text = result.content[0].text
            else:
                extracted_text = str(result.content)

            paraphrased_data = await paraphrase(extracted_text)
            
            messages.append(
                {
                    "role":"tool",
                    "tool_call_id": tool_call.id,
                    "content": paraphrased_data
                }
            )



            response = await self.send_message_to_llm(messages)

            logger.debug(f"-------output--------------\n{response}\n")
        return response.choices[0].message.content if response.choices else "Max tool iterations reached."


# INITIALIZE MCP CLIENT ON STARTUP

@app.on_event("startup")
async def startup_event():
    """Initialize the MCP client when the FastAPI server starts."""

    global client
    client = MCPClient(MCP_SERVER_SCRIPT)
    await client.start()

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    """
    Main endpoint for receiving queries.

    Args:
        request (QueryRequest): The POST request body containing the 'query'.

    Returns:
        dict: A dictionary containing the original query and the final answer.
    """

    try:
       
        answer = await client.process_query(request.query)
        if answer is None:
            answer = "Answer was None"

        return {"query": request.query, "answer": answer}
    except Exception as e:
        logger.exception("Error processing query")
        raise HTTPException(status_code=500, detail=str(e))




