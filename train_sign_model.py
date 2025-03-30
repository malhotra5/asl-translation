#!/usr/bin/env python3
"""
Training Script for Sign Language Recognition Model.
This script handles the training of the sign language recognition model.
"""

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import requests
from sign_recognition import SignLanguageModel, SignRecognizer

# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SignLanguageDataset(Dataset):
    """Dataset for sign language videos."""
    def __init__(self, video_paths, labels, transform=None, max_frames=30):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform
        self.max_frames = max_frames
        self.recognizer = SignRecognizer(max_frames=max_frames)
        
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        try:
            # Extract landmarks
            landmarks = self.recognizer.extract_landmarks(video_path)
            
            # Convert to tensor
            landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32)
            label_tensor = torch.tensor(label, dtype=torch.long)
            
            return landmarks_tensor, label_tensor
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            # Return empty landmarks in case of error
            empty_landmarks = np.zeros((self.max_frames, 543))
            return torch.tensor(empty_landmarks, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def download_wlasl_videos(data, output_dir='data/wlasl/videos', limit=10):
    """Download videos from the WLASL dataset."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Keep track of videos and their labels
    video_paths = []
    video_labels = []
    label_to_gloss = {}
    
    for label_idx, entry in enumerate(tqdm(data[:limit], desc="Downloading videos")):
        gloss = entry['gloss']
        label_to_gloss[label_idx] = gloss
        
        for instance in entry['instances']:
            video_id = instance['video_id']
            video_url = instance.get('url', '')
            
            # Skip if no URL is provided
            if not video_url:
                continue
                
            # Define output path
            output_path = os.path.join(output_dir, f"{video_id}.mp4")
            
            # Skip if already downloaded
            if os.path.exists(output_path):
                # Verify the video file is valid
                try:
                    cap = cv2.VideoCapture(output_path)
                    if not cap.isOpened():
                        print(f"Warning: Existing video file {output_path} is corrupted. Re-downloading...")
                        os.remove(output_path)
                    else:
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        if frame_count <= 0:
                            print(f"Warning: Existing video file {output_path} has no frames. Re-downloading...")
                            os.remove(output_path)
                            cap.release()
                        else:
                            cap.release()
                            video_paths.append(output_path)
                            video_labels.append(label_idx)
                            continue
                except Exception as e:
                    print(f"Error verifying video {output_path}: {e}")
                    os.remove(output_path)
                
            try:
                # Download the video
                response = requests.get(video_url, stream=True, timeout=10)
                if response.status_code == 200:
                    with open(output_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=1024):
                            if chunk:
                                f.write(chunk)
                    
                    # Verify the downloaded video
                    cap = cv2.VideoCapture(output_path)
                    if not cap.isOpened():
                        print(f"Warning: Downloaded video {output_path} is corrupted. Skipping...")
                        os.remove(output_path)
                        continue
                    
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if frame_count <= 0:
                        print(f"Warning: Downloaded video {output_path} has no frames. Skipping...")
                        os.remove(output_path)
                        cap.release()
                        continue
                    
                    cap.release()
                    video_paths.append(output_path)
                    video_labels.append(label_idx)
                else:
                    print(f"Failed to download {video_id}: HTTP {response.status_code}")
            except Exception as e:
                print(f"Error downloading {video_id}: {e}")
                # Remove partially downloaded file if it exists
                if os.path.exists(output_path):
                    os.remove(output_path)
    
    return video_paths, video_labels, label_to_gloss

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    """Train the sign language recognition model."""
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
    
    return model

def evaluate_model(model, test_loader, device):
    """Evaluate the sign language recognition model."""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    return accuracy, all_preds, all_labels

def main():
    """Main function to train the sign language recognition model."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Sign Language Recognition Model')
    parser.add_argument('--data', type=str, default='data/wlasl/WLASL_v0.3.json', 
                        help='Path to the WLASL dataset JSON file')
    parser.add_argument('--output_dir', type=str, default='data/wlasl/videos', 
                        help='Directory to save downloaded videos')
    parser.add_argument('--model_output', type=str, default='models/sign_language_model.pth', 
                        help='Path to save the trained model')
    parser.add_argument('--num_classes', type=int, default=10, 
                        help='Number of classes (words) to use for training')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for training if available')
    parser.add_argument('--max_frames', type=int, default=30, 
                        help='Maximum number of frames to process from each video')
    args = parser.parse_args()
    
    # Set random seed
    set_seed(42)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(os.path.dirname(args.model_output), exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the WLASL dataset
    if not os.path.exists(args.data):
        os.makedirs(os.path.dirname(args.data), exist_ok=True)
        print(f"Downloading WLASL dataset to {args.data}")
        url = "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"
        response = requests.get(url)
        with open(args.data, 'w') as f:
            f.write(response.text)
    
    with open(args.data, 'r') as f:
        wlasl_data = json.load(f)
    
    print(f"Total number of glosses (words): {len(wlasl_data)}")
    print(f"Using a subset of {args.num_classes} words")
    
    # Download videos
    video_paths, video_labels, label_to_gloss = download_wlasl_videos(
        wlasl_data, args.output_dir, limit=args.num_classes)
    
    print(f"Downloaded {len(video_paths)} videos")
    print(f"Label to gloss mapping: {label_to_gloss}")
    
    # Check if we have enough videos
    if len(video_paths) < 2:
        print("Error: Not enough valid videos found. Please check your internet connection and try again.")
        return
    
    # Check if we have at least one video for each class
    unique_labels = set(video_labels)
    if len(unique_labels) < 2:
        print(f"Error: Only found videos for {len(unique_labels)} classes. Need at least 2 classes for training.")
        return
    
    # Split the data into train and test sets
    try:
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            video_paths, video_labels, test_size=0.2, random_state=42, stratify=video_labels
        )
    except ValueError as e:
        print(f"Error splitting data: {e}")
        print("Trying without stratification...")
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            video_paths, video_labels, test_size=0.2, random_state=42
        )
    
    print(f"Training set: {len(train_paths)} videos")
    print(f"Test set: {len(test_paths)} videos")
    
    # Create datasets
    train_dataset = SignLanguageDataset(train_paths, train_labels, max_frames=args.max_frames)
    test_dataset = SignLanguageDataset(test_paths, test_labels, max_frames=args.max_frames)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize the model
    input_dim = 543  # Total number of features per frame (pose + hands + face landmarks)
    hidden_dim = 128
    num_layers = 2
    num_classes = len(label_to_gloss)
    
    model = SignLanguageModel(input_dim, hidden_dim, num_layers, num_classes).to(device)
    print(model)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Train the model
    model = train_model(model, train_loader, criterion, optimizer, device, num_epochs=args.epochs)
    
    # Evaluate the model
    accuracy, all_preds, all_labels = evaluate_model(model, test_loader, device)
    
    # Save the model
    os.makedirs(os.path.dirname(args.model_output), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'label_to_gloss': label_to_gloss,
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'num_classes': num_classes
    }, args.model_output)
    
    print(f"Model saved to {args.model_output}")

if __name__ == "__main__":
    main()