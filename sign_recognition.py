#!/usr/bin/env python3
"""
Sign Language Recognition Module.
This module handles the extraction of landmarks from sign language videos
and the recognition of signs using a pre-trained model.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from tqdm import tqdm

class SignLanguageModel(nn.Module):
    """LSTM-based model for sign language recognition."""
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(SignLanguageModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)  # *2 for bidirectional
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        
        # LSTM output: (batch_size, sequence_length, hidden_dim * 2)
        lstm_out, _ = self.lstm(x)
        
        # Take the output from the last time step
        lstm_out = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.fc1(lstm_out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

class SignRecognizer:
    """Class for sign language recognition from videos."""
    def __init__(self, model_path=None, device="cpu", max_frames=30):
        self.device = device
        self.max_frames = max_frames
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Load the model if a path is provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.model = None
            self.label_to_gloss = {}
            print("Warning: No model loaded. Only landmark extraction will be available.")
    
    def load_model(self, model_path):
        """Load a pre-trained sign language recognition model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract model parameters
        input_dim = checkpoint.get('input_dim', 543)
        hidden_dim = checkpoint.get('hidden_dim', 128)
        num_layers = checkpoint.get('num_layers', 2)
        num_classes = checkpoint.get('num_classes', 10)
        
        # Initialize the model
        self.model = SignLanguageModel(input_dim, hidden_dim, num_layers, num_classes).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load label to gloss mapping
        self.label_to_gloss = checkpoint.get('label_to_gloss', {})
        
        print(f"Model loaded successfully with {num_classes} classes.")
    
    def extract_landmarks(self, video_path):
        """Extract pose, face, and hand landmarks from a video using MediaPipe."""
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps
        
        # Calculate frames to sample
        if frame_count <= self.max_frames:
            frames_to_sample = list(range(frame_count))
        else:
            frames_to_sample = np.linspace(0, frame_count-1, self.max_frames, dtype=int)
        
        landmarks_sequence = []
        
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            for frame_idx in frames_to_sample:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                success, image = cap.read()
                
                if not success:
                    # If frame read failed, append zeros
                    landmarks_sequence.append(np.zeros(543))
                    continue
                    
                # Convert the BGR image to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Process the image and detect landmarks
                results = holistic.process(image)
                
                # Extract landmarks
                frame_landmarks = []
                
                # Pose landmarks (33 landmarks x 3 coordinates)
                if results.pose_landmarks:
                    pose = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
                    frame_landmarks.extend(np.array(pose).flatten())
                else:
                    frame_landmarks.extend(np.zeros(33*3))
                    
                # Left hand landmarks (21 landmarks x 3 coordinates)
                if results.left_hand_landmarks:
                    left_hand = [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
                    frame_landmarks.extend(np.array(left_hand).flatten())
                else:
                    frame_landmarks.extend(np.zeros(21*3))
                    
                # Right hand landmarks (21 landmarks x 3 coordinates)
                if results.right_hand_landmarks:
                    right_hand = [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]
                    frame_landmarks.extend(np.array(right_hand).flatten())
                else:
                    frame_landmarks.extend(np.zeros(21*3))
                    
                # Face landmarks (we'll use a subset of 10 landmarks for simplicity)
                if results.face_landmarks:
                    # Select a subset of face landmarks (e.g., eyes, nose, mouth)
                    face_indices = [0, 4, 6, 8, 10, 152, 234, 454, 10, 338]  # Example indices
                    face = [[results.face_landmarks.landmark[idx].x,
                             results.face_landmarks.landmark[idx].y,
                             results.face_landmarks.landmark[idx].z] for idx in face_indices]
                    frame_landmarks.extend(np.array(face).flatten())
                else:
                    frame_landmarks.extend(np.zeros(10*3))
                    
                landmarks_sequence.append(frame_landmarks)
        
        cap.release()
        
        # Pad or truncate to ensure all sequences have the same length
        if len(landmarks_sequence) < self.max_frames:
            # Pad with zeros
            pad_length = self.max_frames - len(landmarks_sequence)
            landmarks_sequence.extend([np.zeros(543)] * pad_length)
        
        return np.array(landmarks_sequence)
    
    def recognize(self, video_path):
        """Recognize sign language from a video."""
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        # Extract landmarks
        landmarks = self.extract_landmarks(video_path)
        landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(landmarks_tensor)
            _, predicted = torch.max(outputs.data, 1)
            predicted_label = predicted.item()
        
        # Convert to text
        predicted_text = self.label_to_gloss.get(predicted_label, f"Unknown_{predicted_label}")
        
        return predicted_text
    
    def visualize_landmarks(self, video_path, output_path=None):
        """Visualize the landmarks on a frame from the video."""
        cap = cv2.VideoCapture(video_path)
        
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            success, image = cap.read()
            
            if not success:
                print("Failed to read video")
                return
                
            # Convert the BGR image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image and detect landmarks
            results = holistic.process(image_rgb)
            
            # Draw landmarks
            image_copy = image.copy()
            
            # Draw pose landmarks
            self.mp_drawing.draw_landmarks(
                image_copy, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
            
            # Draw left hand landmarks
            self.mp_drawing.draw_landmarks(
                image_copy, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
            
            # Draw right hand landmarks
            self.mp_drawing.draw_landmarks(
                image_copy, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
            
            # Draw face landmarks
            self.mp_drawing.draw_landmarks(
                image_copy, results.face_landmarks)
            
            # Save or display the image
            if output_path:
                cv2.imwrite(output_path, image_copy)
                print(f"Visualization saved to {output_path}")
            else:
                # Convert back to RGB for display
                image_rgb_vis = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
                return image_rgb_vis
        
        cap.release()

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Sign Language Recognition')
    parser.add_argument('--video', type=str, required=True, help='Path to the sign language video')
    parser.add_argument('--model', type=str, default=None, help='Path to the sign language recognition model')
    parser.add_argument('--visualize', action='store_true', help='Visualize landmarks')
    parser.add_argument('--output', type=str, default=None, help='Path to save visualization')
    args = parser.parse_args()
    
    recognizer = SignRecognizer(model_path=args.model)
    
    if args.visualize:
        recognizer.visualize_landmarks(args.video, args.output)
    
    if args.model:
        text = recognizer.recognize(args.video)
        print(f"Recognized sign: {text}")