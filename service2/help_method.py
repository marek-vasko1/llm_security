import os
import sys
sys.path.append('./utils') 
from mask_utils import * 
from augmentations import * 
from similarity import *
import time
import logging

logger = logging.getLogger(__name__) 
logging.basicConfig(level=logging.DEBUG) 
def get_method(method_name): 
    try: 
        method = text_aug_dict[method_name] 
    except: 
        print('Check your method!!') 
        os._exit(0) 
    return method 


'''Added because of error in translation -- ConnectTimeout(TimeoutError())
default set on PL'''
def change_prompt(prompt, mode="PL", retries=3):
    tmp_method=get_method(mode)
    for i in range(retries):
        try:
            return tmp_method(text_list=[prompt])
        except Exception as e:
            logger.warning(f"Chyba při augmentaci (pokus {i+1}): {e}")
            time.sleep(4) 
    time.sleep(4)
    return [prompt]