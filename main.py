#!/usr/bin/env python3
"""
Main script for the Sign Language to Speech Translation System.
This script orchestrates the complete pipeline.
"""

import argparse
import os
import cv2
import numpy as np
import torch
from sign_recognition import SignRecognizer
from grammar_correction import GrammarCorrector
from text_to_speech import TextToSpeechConverter

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Sign Language to Speech Translation System')
    parser.add_argument('--video', type=str, required=True, help='Path to the sign language video')
    parser.add_argument('--output', type=str, default='output.wav', help='Path to save the output audio')
    parser.add_argument('--model', type=str, default='models/sign_language_model.pth', 
                        help='Path to the sign language recognition model')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for inference if available')
    parser.add_argument('--max_frames', type=int, default=30, 
                        help='Maximum number of frames to process from the video')
    parser.add_argument('--verbose', action='store_true', help='Print detailed information')
    return parser.parse_args()

def main():
    """Main function to run the complete pipeline."""
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    if args.verbose:
        print(f"Using device: {device}")
    
    # Step 1: Initialize the sign language recognizer
    sign_recognizer = SignRecognizer(model_path=args.model, device=device, max_frames=args.max_frames)
    
    # Step 2: Initialize the grammar corrector
    grammar_corrector = GrammarCorrector()
    
    # Step 3: Initialize the text-to-speech converter
    tts_converter = TextToSpeechConverter()
    
    # Step 4: Run the complete pipeline
    if args.verbose:
        print(f"Processing video: {args.video}")
        print("Step 1: Translating sign language to text...")
    
    # Recognize sign from video
    recognized_text = sign_recognizer.recognize(args.video)
    
    if args.verbose:
        print(f"Recognized text: {recognized_text}")
        print("Step 2: Correcting grammar...")
    
    # Correct grammar
    corrected_text = grammar_corrector.correct(recognized_text)
    
    if args.verbose:
        print(f"Corrected text: {corrected_text}")
        print("Step 3: Converting text to speech...")
    
    # Convert text to speech
    tts_converter.synthesize(corrected_text, args.output)
    
    if args.verbose:
        print(f"Speech saved to: {args.output}")
        print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()