#!/usr/bin/env python3
"""
Grammar Correction Module.
This module handles the correction of grammatical errors in the translated text.
"""

import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

class GrammarCorrector:
    """Class for correcting grammatical errors in text."""
    def __init__(self, model_name="prithivida/grammar_error_correcter_v1", device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
        # Load the model and tokenizer
        self.load_model()
    
    def load_model(self):
        """Load the T5 model and tokenizer for grammar correction."""
        try:
            print(f"Loading grammar correction model: {self.model_name}")
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
            self.model.to(self.device)
            print("Grammar correction model loaded successfully.")
        except Exception as e:
            print(f"Error loading grammar correction model: {e}")
            print("Using fallback grammar correction.")
    
    def correct(self, text):
        """Correct grammatical errors in the text."""
        if self.model is None or self.tokenizer is None:
            return self._fallback_correction(text)
        
        try:
            input_text = f"grammar: {text}"
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
            
            outputs = self.model.generate(input_ids, max_length=128)
            corrected_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return corrected_text
        except Exception as e:
            print(f"Error during grammar correction: {e}")
            return self._fallback_correction(text)
    
    def _fallback_correction(self, text):
        """Fallback method for grammar correction when the model is not available."""
        print("Using fallback grammar correction.")
        
        # Simple rule-based corrections for common ASL to English grammar issues
        words = text.split()
        
        # Add articles if missing
        if len(words) > 0 and words[0].lower() not in ["the", "a", "an", "i", "you", "he", "she", "we", "they"]:
            if words[0][0].lower() in "aeiou":
                words.insert(0, "An")
            else:
                words.insert(0, "A")
        
        # Capitalize first word
        if len(words) > 0:
            words[0] = words[0].capitalize()
        
        # Add period at the end if missing
        if len(words) > 0 and not words[-1].endswith((".", "!", "?")):
            words[-1] = words[-1] + "."
        
        corrected_text = " ".join(words)
        
        # Handle common ASL patterns
        corrected_text = corrected_text.replace("me go", "I am going")
        corrected_text = corrected_text.replace("me want", "I want")
        corrected_text = corrected_text.replace("me have", "I have")
        corrected_text = corrected_text.replace("you go", "you are going")
        
        return corrected_text

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Grammar Correction')
    parser.add_argument('--text', type=str, required=True, help='Text to correct')
    parser.add_argument('--model', type=str, default="prithivida/grammar_error_correcter_v1", 
                        help='Model name or path')
    args = parser.parse_args()
    
    corrector = GrammarCorrector(model_name=args.model)
    corrected_text = corrector.correct(args.text)
    
    print(f"Original text: {args.text}")
    print(f"Corrected text: {corrected_text}")