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
    """
    Retrieve a text augmentation method by its name.

    Workflow:
        1. Look up the method in the predefined `text_aug_dict`.
        2. Return the method if found.
        3. Raise a ValueError if the method does not exist to prevent silent failures.

    Args:
        method_name (str): The abbreviation or name of the augmentation method (e.g., 'PL', 'RR', 'SR').

    Returns:
        Callable: The function that performs the requested text augmentation.

    Raises:
        ValueError: If the provided method_name is not found in the dictionary.
    """

    try: 
        method = text_aug_dict[method_name] 
    except: 
        print('Check your method!!') 
        os._exit(0) 
    return method 


def change_prompt(prompt, mode="PL", retries=3):
    """
    Apply an augmentation method to modify the input prompt.

    Workflow:
        1. Fetch the target augmentation method based on the `mode`.
        2. Attempt to generate the augmented prompt.
        3. If an exception occurs (e.g., translation API timeout), log the warning and sleep.
        4. Retry up to the specified maximum number of `retries`.
        5. If all retries fail, return the original prompt as a safe fallback.

    Args:
        prompt (str): The original user input string to be augmented.
        mode (str, optional): The augmentation method to apply. Defaults to "PL" (Policy).
        retries (int, optional): The number of attempts before failing gracefully. Defaults to 3.

    Returns:
        list: A list containing the augmented prompt string. Returns the original prompt if augmentation fails.
    """

    tmp_method=get_method(mode)
    for i in range(retries):
        try:
            return tmp_method(text_list=[prompt])
        except Exception as e:
            logger.warning(f"Chyba při augmentaci (pokus {i+1}): {e}")
            time.sleep(4) 
    time.sleep(4)
    return [prompt]
