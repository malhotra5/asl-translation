#!/usr/bin/env python3
"""
Demo Script for Sign Language to Speech Translation System.
This script demonstrates the complete pipeline with a sample video.
"""

import os
import argparse
import cv2
import numpy as np
import torch
from sign_recognition import SignRecognizer
from grammar_correction import GrammarCorrector
from text_to_speech import TextToSpeechConverter

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Sign Language to Speech Translation Demo')
    parser.add_argument('--video', type=str, default=None, 
                        help='Path to the sign language video (if not provided, will use webcam)')
    parser.add_argument('--model', type=str, default='models/sign_language_model.pth', 
                        help='Path to the sign language recognition model')
    parser.add_argument('--output', type=str, default='output.wav', 
                        help='Path to save the output audio')
    parser.add_argument('--use_gpu', action='store_true', 
                        help='Use GPU for inference if available')
    parser.add_argument('--max_frames', type=int, default=30, 
                        help='Maximum number of frames to process from the video')
    parser.add_argument('--webcam_duration', type=int, default=5, 
                        help='Duration in seconds to record from webcam')
    parser.add_argument('--webcam_device', type=int, default=0, 
                        help='Webcam device index')
    return parser.parse_args()

def record_from_webcam(output_path, duration=5, device_index=0):
    """Record a video from the webcam."""
    cap = cv2.VideoCapture(device_index)
    
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30  # Default to 30 fps if not detected
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Calculate total frames to capture
    total_frames = duration * fps
    
    print(f"Recording from webcam for {duration} seconds...")
    
    # Record video
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Display countdown
        remaining = duration - int(i / fps)
        cv2.putText(frame, f"Recording: {remaining}s", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Write frame to video file
        out.write(frame)
        
        # Display frame
        cv2.imshow('Recording', frame)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Video saved to {output_path}")
    return output_path

def main():
    """Main function to run the demo."""
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Record from webcam if no video is provided
    if args.video is None:
        webcam_output = "webcam_recording.mp4"
        args.video = record_from_webcam(webcam_output, args.webcam_duration, args.webcam_device)
        if args.video is None:
            print("Error: Failed to record from webcam.")
            return
    
    # Step 1: Initialize the sign language recognizer
    print("Initializing sign language recognizer...")
    sign_recognizer = SignRecognizer(model_path=args.model, device=device, max_frames=args.max_frames)
    
    # Step 2: Initialize the grammar corrector
    print("Initializing grammar corrector...")
    grammar_corrector = GrammarCorrector()
    
    # Step 3: Initialize the text-to-speech converter
    print("Initializing text-to-speech converter...")
    tts_converter = TextToSpeechConverter()
    
    # Step 4: Run the complete pipeline
    print(f"Processing video: {args.video}")
    
    # Visualize landmarks
    print("Visualizing landmarks...")
    vis_output = "landmarks.jpg"
    sign_recognizer.visualize_landmarks(args.video, vis_output)
    
    # Recognize sign from video
    print("Recognizing sign language...")
    recognized_text = sign_recognizer.recognize(args.video)
    print(f"Recognized text: {recognized_text}")
    
    # Correct grammar
    print("Correcting grammar...")
    corrected_text = grammar_corrector.correct(recognized_text)
    print(f"Corrected text: {corrected_text}")
    
    # Convert text to speech
    print("Converting text to speech...")
    tts_converter.synthesize(corrected_text, args.output)
    print(f"Speech saved to: {args.output}")
    
    print("Demo completed successfully!")
    
    # Try to play the audio
    try:
        import platform
        import subprocess
        
        system = platform.system()
        if system == 'Darwin':  # macOS
            subprocess.call(["afplay", args.output])
        elif system == 'Linux':
            subprocess.call(["aplay", args.output])
        elif system == 'Windows':
            subprocess.call(["start", args.output], shell=True)
        else:
            print(f"Audio playback not supported on {system}.")
    except Exception as e:
        print(f"Error playing audio: {e}")

if __name__ == "__main__":
    main()