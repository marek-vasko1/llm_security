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
import copy

import logging
from openai import AsyncOpenAI

#granite
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

"""code of llm defense is inspirated by https://huggingface.co/ibm-granite/granite-guardian-3.1-2b 24.03.2026"""

# ====== Settings  ======
model_path = "ibm-granite/granite-guardian-3.1-2b"
safe_token = "No"
unsafe_token = "Yes"
nlogprobs = 20
threshold = 0.5


# ====== Load model ======
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

"""code from https://huggingface.co/ibm-granite/granite-guardian-3.1-2b 24.03.2026"""
def parse_output(output, input_len):
    label, prob_of_risk = None, None

    if nlogprobs > 0:

        list_index_logprobs_i = [torch.topk(token_i, k=nlogprobs, largest=True, sorted=True)
                                 for token_i in list(output.scores)[:-1]]
        if list_index_logprobs_i is not None:
            prob = get_probabilities(list_index_logprobs_i)
            prob_of_risk = prob[1]

    res = tokenizer.decode(output.sequences[:,input_len:][0],skip_special_tokens=True).strip()
    if unsafe_token.lower() == res.lower():
        label = unsafe_token
    elif safe_token.lower() == res.lower():
        label = safe_token
    else:
        label = "Failed"

    return label, prob_of_risk.item()

"""code from https://huggingface.co/ibm-granite/granite-guardian-3.1-2b 24.03.2026"""
def get_probabilities(logprobs):
    safe_token_prob = 1e-50
    unsafe_token_prob = 1e-50
    for gen_token_i in logprobs:
        for logprob, index in zip(gen_token_i.values.tolist()[0], gen_token_i.indices.tolist()[0]):
            decoded_token = tokenizer.convert_ids_to_tokens(index)
            if decoded_token.strip().lower() == safe_token.lower():
                safe_token_prob += math.exp(logprob)
            if decoded_token.strip().lower() == unsafe_token.lower():
                unsafe_token_prob += math.exp(logprob)

    probabilities = torch.softmax(
        torch.tensor([math.log(safe_token_prob), math.log(unsafe_token_prob)]), dim=0
    )

    return probabilities



def check_input(user_text, risk_name="harm"):
    """
    Validate user input using Granite Guardian model.

    Args:
        user_text (str): Input text from user.
        risk_name (str): Type of risk to evaluate.

    Returns:
        tuple:
            label (str): Safety classification
            prob_of_risk (float): Probability of unsafe content

    Inspirated by https://huggingface.co/ibm-granite/granite-guardian-3.1-2b from 24.03.2026
    """
    
    guardian_config = {"risk_name": risk_name}
    messages = [{"role": "user", "content": user_text}]
    #logging.debug(messages)
    #logging.debug("INPUT----------------------------------------------")

    input_ids = tokenizer.apply_chat_template(
        messages,
        guardian_config=guardian_config,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    input_len = input_ids.shape[1]

    model.eval()

    with torch.no_grad():
        output = model.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=20,
            return_dict_in_generate=True,
            output_scores=True
        )

    label, prob_of_risk = parse_output(output, input_len)
    logging.debug("INPUT CHECK DONE")
    return label, prob_of_risk

def check_output(messages, risk_name="harm"):
    """
    Validate conversation output using Granite Guardian model.

    Args:
        messages (list): Conversation history.
        risk_name (str): Type of risk to evaluate.

    Returns:
        tuple:
            label (str): Safety classification
            prob_of_risk (float): Probability of unsafe content

    Inspirated by https://huggingface.co/ibm-granite/granite-guardian-3.1-2b from 24.03.2026
    """

    #logging.debug(f"----------{messages}------------\n")
    guardian_config = {"risk_name": risk_name}
    
  
    #logging.debug("ALL----------------------------------------------")

    input_ids = tokenizer.apply_chat_template(
        messages,
        guardian_config=guardian_config,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    input_len = input_ids.shape[1]

    model.eval()

    with torch.no_grad():
        output = model.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=20,
            return_dict_in_generate=True,
            output_scores=True
        )

    label, prob_of_risk = parse_output(output, input_len)

    return label, prob_of_risk


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
        #logging.debug(f"\nthere are recived tools:\n self.formatted_tools\n")

    async def send_message_to_llm(self, messages: list[dict]):
        """
        Send messages to LLM and receive response.

        Args:
            messages (list): Conversation messages.

        Returns:
            Response object from LLM API.
        """
        #logger.debug(f"\nSending request to LLM with messages:\n {messages} \n")

        response = await client_ai.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=1000,
            messages=messages,
            tools=self.formatted_tools if self.formatted_tools else None
        )
        #logger.debug(f"\n recieved response:\n {response}\n")
        return response
    
    async def process_query(self, query: str):
        """
        Main processing function for handling user queries.

        Workflow:
            1. Send query to LLM
            2. If LLM requests tool usage:
                - Call MCP tool
                - Append tool result to messages
            3. Validate output after each step
            4. Repeat up to 10 iterations

        Args:
            query (str): User query.

        Returns:
            tuple:
                response (str): Final answer or block message
                messages (list): Conversation history
        """

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that can use external tools when necessary."
            }

        ]
        messages.append({"role":"user", "content": query})        
        
        #logger.debug("\nSENDING QUERY\n")
        response = await self.send_message_to_llm(messages)
    
        messages_for_guardian = copy.deepcopy(messages)

        for i in range(10):
            #logger.debug(f"loop running{i}")
            message = response.choices[0].message
            #logger.debug(f"\nRESPONSE PRO KONTROLU V PRIPADE DRUHEHO KOLA\n {response}")
             
            if not message.tool_calls:
                #logger.debug("\nNO TOOL CALL\n")
                break

            messages.append(message.model_dump(exclude_none=True))
            
            tool_call = message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            messages_for_guardian.append({
                 "role": "assistant", 
                 "content": f"[Calling tool {tool_name} with arguments: {tool_args}]"
            })

            # call tool
            tool_result = await self.session.call_tool(tool_name, tool_args)
            #logger.debug(f"*******************************tool call result {tool_result.content} ***********************************")
            
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
            messages_for_guardian.append({"role":"user", "content": extracted_text})
            #logger.debug(f"\nMESSAGES:\n {messages}")
            #logger.debug(f"\nMESSAGES FOR GUARDIAN:\n{messages_for_guardian}")

            label, probability = await asyncio.to_thread(check_output,messages_for_guardian)
            if label == safe_token:
                if probability > threshold:
                    return "Blocked by Granite Guardian", messages      
            elif label == unsafe_token or label == "Failed":
                return "Blocked by Granite Guardian", messages

            response = await self.send_message_to_llm(messages)

        return response.choices[0].message.content if response.choices else "Max tool iterations reached."      
# INITIALIZE MCP CLIENT ON STARTUP

@app.on_event("startup")
async def startup_event():
    global client
    client = MCPClient(MCP_SERVER_SCRIPT)
    await client.start()

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    """
    Main API endpoint for processing queries.

    Workflow:
        1. Validate input (check_input)
        2. Block if unsafe
        3. Process query via MCP + LLM
        4. Validate output (check_output)
        5. Return final response

    Args:
        request (QueryRequest): Incoming request.

    Returns:
        dict:
            query (str): Original query
            answer (str): Response or blocked message
    """

    try:
        #check input
        label, probability = await asyncio.to_thread(check_input, request.query) #maby move to process querry function
        if label == safe_token:
            if probability > threshold:
                return {"query": request.query, "answer": "Blocked by Granite Guardian"}
        if label == unsafe_token or label == "Failed":
                return {"query": request.query, "answer": "Blocked by Granite Guardian"}
        
        #logger.debug(f"\nCHECK INPUT DONE-------------------{label}{probability}")
        answer = await client.process_query(request.query)
           
        if answer == "Blocked by Granite Guardian":
            return {"query": request.query, "answer": answer}

        try:
            #check output
            #logger.debug("\nSTARTING OUTPUT CHECK\n")
            messages = [{"role": "user", "content": request.query}, {"role" : "assistant", "content" : answer}]
            #logger.debug(f"\nMESSAGES:\n {messages}")
            label, probability = await asyncio.to_thread(check_output,messages)
            if label == safe_token:
                if probability > threshold:
                     return {"query": request.query, "answer": "Blocked by Granite Guardian"}
            elif label == unsafe_token or label == "Failed":
                return {"query": request.query, "answer": "Blocked by Granite Guardian"}
                
            if (answer is None):
                answer = "Something happend..."

        except AttributeError as e:
            logging.exception("CRASH IN ENDPOINT:")

        return {"query": request.query, "answer": answer}
    except Exception as e:
        logging.info(f"tady exc {e}")
        raise HTTPException(status_code=500, detail=str(e))






