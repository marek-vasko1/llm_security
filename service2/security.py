from llm_client import MCPClient
import asyncio
import argparse 
import os 
os.environ["CUDA_VISIBLE_DEVICES"]="1" 
import sys 
sys.path.append('./utils') 
from utils import * 
import numpy as np 
from mask_utils import * 
from augmentations import * 
import spacy 
from similarity import *
from help_method import change_prompt
import logging

metric = spacy.load("en_core_web_md")
#client = MCPClient("/app/mcp_server.py")
logger = logging.getLogger(__name__) 
logging.basicConfig(level=logging.DEBUG) 

def update_divergence(output_list, simialrity_eval='spacy', metric=None, top_string=None): 
    all_block = determine_blocked(output_list) 
    number = len(output_list) 
    similarity_matrix = np.zeros((number, number)) 
    divergence_matrix = np.zeros((number, number)) 
    if top_string != None: 
        output_list = [_str[:top_string] for _str in output_list] 
    if simialrity_eval in ['spacy', 'transformer']: 
        for i in range(number): 
            for j in range(number): 
                similarity_matrix[i,j] = get_similarity(output_list[i], output_list[j], method=simialrity_eval, misc=metric)
    else: 
        print('not a valid similarity metric') 
        
    similarity_matrix = np.clip(similarity_matrix, 0.01, None) 

    for i in range(number): 
        for j in range(number): 
            if i != j: 
                divergence_matrix[i, j] = get_divergence(similarity_matrix, i, j) 

    divergence_matrix = np.clip(divergence_matrix, None, 100) 
    return divergence_matrix.max(), all_block 
    
'''Check if input is attack or not 
@return True if attack otherwise False'''
async def jailGuard(prompt: str, client):
    generated_variants = []
    responses = []

    for _ in range(4):
        variant = change_prompt(prompt)
        generated_variants.append(variant)

    for variant in generated_variants:
        prompt_text = variant[0] if isinstance(variant, list) else variant

        try:
            response = await client.process_query(prompt_text)
            logging.debug("RESPONSE JAIGUARD----------------------")
            logging.debug(response)
            responses.append(response or "")
            
        except Exception as e:
            responses.append("No response!")
            print(f"Chyba: {e}")

    max_div, jailbreak_keywords = update_divergence(
        output_list=responses,
        metric=metric
    )

    threshold = 0.3
    detect_value = detect_attack(max_div, jailbreak_keywords, threshold)
    logging.debug("JAILGUARD RESPONSEEEEEEEEEEEEEEEEEEEEEEEEEE")
    logging.debug(detect_value)
    return detect_value