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

#granite
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

#v2
#from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
#import torch
#from torch.nn.functional import softmax
#import jinja2, json
#from vllm import LLM, SamplingParams
#import math
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
"""code of llm defense is inspirated by https://huggingface.co/ibm-granite/granite-guardian-3.1-2b"""

# ====== Nastavení ======
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
"""
tokenizer = AutoTokenizer.from_pretrained(model_path)
sampling_params = SamplingParams(temperature=0.0, logprobs=nlogprobs)
model = LLM(model=model_path, tensor_parallel_size=1)

def parse_output(output):
    label, prob_of_risk = None, None

    if nlogprobs > 0:
        logprobs = next(iter(output.outputs)).logprobs
        if logprobs is not None:
            prob = get_probabilities(logprobs)
            prob_of_risk = prob[1]

    res = next(iter(output.outputs)).text.strip()
    if unsafe_token.lower() == res.lower():
        label = unsafe_token
    elif safe_token.lower() == res.lower():
        label = safe_token
    else:
        label = "Failed"

    return label, prob_of_risk.item()

def get_probabilities(logprobs):
    safe_token_prob = 1e-50
    risky_token_prob = 1e-50
    for gen_token_i in logprobs:
        for token_prob in gen_token_i.values():
            decoded_token = token_prob.decoded_token
            if decoded_token.strip().lower() == safe_token.lower():
                safe_token_prob += math.exp(token_prob.logprob)
            if decoded_token.strip().lower() == unsafe_token.lower():
                risky_token_prob += math.exp(token_prob.logprob)

    probabilities = torch.softmax(
        torch.tensor([math.log(safe_token_prob), math.log(risky_token_prob)]), dim=0
    )

    return probabilities


def check_input(user_text, risk_name="harm"):
    messages = [{"role": "user", "content": user_text}]
    guardian_config = {"risk_name": risk_name}
    chat = tokenizer.apply_chat_template(messages, guardian_config = guardian_config, tokenize=False, add_generation_prompt=True)

    output = model.generate(chat, sampling_params, use_tqdm=False)
    predicted_label = output[0].outputs[0].text.strip()

    label, prob_of_risk = parse_output(output[0])

    return label, prob_of_risk

def check_output(messages, risk_name="harm"):
    guardian_config = {"risk_name": risk_name}
    chat = tokenizer.apply_chat_template(messages, guardian_config = guardian_config, tokenize=False, add_generation_prompt=True)

    output = model.generate(chat, sampling_params, use_tqdm=False)
    predicted_label = output[0].outputs[0].text.strip()

    label, prob_of_risk = parse_output(output[0])
    
    return label, prob_of_risk
*/"""

"""code from https://huggingface.co/ibm-granite/granite-guardian-3.1-2b"""
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

"""code from https://huggingface.co/ibm-granite/granite-guardian-3.1-2b"""
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
    guardian_config = {"risk_name": risk_name}
    messages = [{"role": "user", "content": user_text}]
    logging.debug(messages)
    logging.debug("INPUT----------------------------------------------")

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

def check_output(messages, risk_name="harm"):
    guardian_config = {"risk_name": risk_name}
    
    logging.debug(messages)
    logging.debug("ALL----------------------------------------------")

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
        messages_for_guardian = [
            {
            "role": "system",
            "content": "You are a helpful assistant that can use external tools when necessary."
            },
            {
            "role":"user",
            "content":query
            }

        ]
               
        #recive tools
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
            return "Sanitize check failed", messages_for_guardian

        result=response.choices[0].message.content

        while True:

            message = response.choices[0].message

            if not message.tool_calls:
                break

            tool_call = message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args = tool_call.function.arguments
            tool_args = json.loads(tool_call.function.arguments)

            #call tool
            result = await self.session.call_tool(tool_name, tool_args)
            

            messages.append(message)
            messages.append({
                "role":"user",
                "content": result.content
            })
            messages_for_guardian.append({"role": "assistant", "content" : str(message.reasoning_content or "")})
            messages_for_guardian.append({"role": "user", "content": result.content[0].text})
            response = client_ai.chat.completions.create(
                model=LLM_MODEL,
                max_tokens=1000,
                messages=messages,
                tools=formatted_tools
            )
            
            
            result = response.choices[0].message.content if response.choices else "No response"
            #logging.debug("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa")
            #logging.debug(result)
        return result, messages_for_guardian


# INITIALIZE MCP CLIENT ON STARTUP

@app.on_event("startup")
async def startup_event():
    global client
    client = MCPClient(MCP_SERVER_SCRIPT)
    await client.start()

@app.post("/query")
async def query_endpoint(request: QueryRequest):

    try:
        label, probability = await asyncio.to_thread(check_input, request.query) #maby move to process querry function
        if label == safe_token:
            if probability > threshold:
                return {"query": request.query, "answer": "Blocked by Granite Guardian"}
            elif label == unsafe_token or label == "Failed":
                return {"query": request.query, "answer": "Blocked by Granite Guardian"}
                
        if (sanitize_prompt(request.query) is None):
              answer = "I cannot answer this question, because it may try to bypass security guards"
        else:
            logger.debug("DONE CHECK INPUT-------------------")
            answer, messages = await client.process_query(request.query)
            try:

                message = {"role": "assistant", "content": answer}
                messages.append(message)
                label, probability = await asyncio.to_thread(check_output,messages)
                if label == safe_token:
                    if probability > threshold:
                        return {"query": request.query, "answer": "Blocked by Granite Guardian"}
                elif label == unsafe_token or label == "Failed":
                    return {"query": request.query, "answer": "Blocked by Granite Guardian"}
                
                if (answer is None or sanitize_prompt(answer) is None):
                    answer = "I cannot answer this question, because it may try to bypass security guards"

            except AttributeError as e:
                logging.info(f"tady uvod exc {e}")

        return {"query": request.query, "answer": answer}
    except Exception as e:
        logging.info(f"tady exc {e}")
        raise HTTPException(status_code=500, detail=str(e))






