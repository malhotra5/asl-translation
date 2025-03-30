#!/usr/bin/env python3
"""
Text to Speech Module.
This module handles the conversion of text to speech using ESPnet.
"""

import os
import torch
import soundfile as sf
import numpy as np
from tqdm import tqdm

class TextToSpeechConverter:
    """Class for converting text to speech."""
    def __init__(self, model_name="kan-bayashi/ljspeech_tacotron2", device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.tts_model = None
        self.fs = 22050  # Default sampling rate
        
        # Load the model
        self.load_model()
    
    def load_model(self):
        """Load the TTS model."""
        try:
            print(f"Loading TTS model: {self.model_name}")
            
            # Import ESPnet libraries here to avoid loading them if not needed
            from espnet2.bin.tts_inference import Text2Speech
            from espnet_model_zoo.downloader import ModelDownloader
            
            # Download and load the model
            d = ModelDownloader()
            model_config = d.download_and_unpack(self.model_name)
            self.tts_model = Text2Speech(**model_config["tts_train_args"])
            self.tts_model.load_state_dict(d.download_and_unpack(model_config["tts_model_file"]))
            self.tts_model.to(self.device)
            self.fs = self.tts_model.fs
            
            print("TTS model loaded successfully.")
        except Exception as e:
            print(f"Error loading TTS model: {e}")
            print("Using fallback TTS method.")
    
    def synthesize(self, text, output_path="output.wav"):
        """Convert text to speech and save to a file."""
        if self.tts_model is None:
            return self._fallback_synthesis(text, output_path)
        
        try:
            with torch.no_grad():
                wav = self.tts_model(text)["wav"]
            
            sf.write(output_path, wav.numpy(), self.fs, "PCM_16")
            print(f"Speech saved to {output_path}")
            return output_path
        except Exception as e:
            print(f"Error during speech synthesis: {e}")
            return self._fallback_synthesis(text, output_path)
    
    def _fallback_synthesis(self, text, output_path="output.wav"):
        """Fallback method for TTS when the model is not available."""
        print("Using fallback TTS method.")
        
        try:
            # Try using pyttsx3 as a fallback
            import pyttsx3
            engine = pyttsx3.init()
            engine.save_to_file(text, output_path)
            engine.runAndWait()
            print(f"Speech saved to {output_path} using fallback method.")
            return output_path
        except Exception as e:
            print(f"Error during fallback speech synthesis: {e}")
            print(f"Could not synthesize speech. Text: {text}")
            
            # Create a silent audio file as a last resort
            sample_rate = 22050
            duration = 1  # seconds
            samples = np.zeros(int(sample_rate * duration))
            sf.write(output_path, samples, sample_rate, "PCM_16")
            
            return output_path

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Text to Speech Conversion')
    parser.add_argument('--text', type=str, required=True, help='Text to convert to speech')
    parser.add_argument('--output', type=str, default="output.wav", help='Output audio file path')
    parser.add_argument('--model', type=str, default="kan-bayashi/ljspeech_tacotron2", 
                        help='Model name or path')
    args = parser.parse_args()
    
    tts = TextToSpeechConverter(model_name=args.model)
    output_path = tts.synthesize(args.text, args.output)
    
    print(f"Text: {args.text}")
    print(f"Audio saved to: {output_path}")