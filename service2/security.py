import pandas as pd
import joblib
import logging
from asgiref.sync import sync_to_async

from perplexity_calc import compute_perplexity

logger = logging.getLogger(__name__)

# Global env for model

detector_model = None
detector_threshold = None
class SecurityDetector:
    """
    A security handler that encapsulates the LightGBM classifier
    and the logic for detecting jailbreak attacks using Perplexity and Length.
    """
    def __init__(self, model_path: str):
        """
        Load the pre-trained LightGBM model and its optimal decision threshold.
        This should be called once during the application startup event.
        """
        self.model_path = model_path
               
        logger.info("loading LightGBM...")
        data = joblib.load('jailbreak_detector.pkl')
        self.model = data['model']
        self.threshold = data['threshold']
        logger.info(f"LightGBM loaded (Treshold: {self.hreshold:.4f})")
        

    async def check_perplexity(query_text: str):
        """
        Perform a security audit on the incoming prompt using Perplexity and Length features.
    
        Args:
            query_text (str): The raw user input string.
        
        Returns:
            bool: True if the prompt is considered safe, False if a jailbreak attack is detected.
        """
        ppl, seq_len = await sync_to_async(compute_perplexity)(query_text)
    
        features = pd.DataFrame([{'length': seq_len, 'perplexity': ppl}])
        probability = self.model.predict_proba(features)[0, 1]
    
        is_safe = bool(probability < self.threshold)
        logger.debug(f"\n is_safe: {is_safe} | probability: {probability} | perplexity: {ppl} | length: {seq_len} \n")
        return is_safe

