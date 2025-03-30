# Sign Language to Speech Translation System

This repository contains a complete pipeline for translating sign language to speech, consisting of three main components:

1. **Sign Language to Text Translation**: Uses computer vision and deep learning to recognize signs from videos.
2. **Grammatical Error Correction**: Employs a language model to correct any grammatical errors in the translated text.
3. **Text to Speech Conversion**: Converts the corrected text to natural-sounding speech.

## Features

- **MediaPipe Integration**: Extracts hand, face, and body landmarks from sign language videos.
- **LSTM-based Recognition**: Uses bidirectional LSTM networks to recognize signs from landmark sequences.
- **T5 Grammar Correction**: Corrects grammatical errors specific to sign language translation.
- **ESPnet TTS**: Generates high-quality speech from the corrected text.
- **Complete Pipeline**: End-to-end system from sign language video to speech output.
- **Modular Design**: Each component can be used independently or as part of the complete pipeline.
- **Command-line Interface**: Easy-to-use command-line tools for training, evaluation, and inference.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/malhotra5/asl-translation.git
   cd asl-translation
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Training the Sign Language Recognition Model

```bash
python train_sign_model.py --data data/wlasl/WLASL_v0.3.json --output_dir data/wlasl/videos --model_output models/sign_language_model.pth --num_classes 10 --epochs 10 --batch_size 4 --use_gpu
```

### Running the Complete Pipeline

```bash
python main.py --video path/to/sign_video.mp4 --output output.wav --model models/sign_language_model.pth --use_gpu --verbose
```

### Running the Demo with Webcam

```bash
python demo.py --webcam_duration 5 --output output.wav --model models/sign_language_model.pth
```

### Using Individual Components

#### Sign Language Recognition

```bash
python sign_recognition.py --video path/to/sign_video.mp4 --model models/sign_language_model.pth --visualize --output landmarks.jpg
```

#### Grammar Correction

```bash
python grammar_correction.py --text "me go store" --model "prithivida/grammar_error_correcter_v1"
```

#### Text to Speech

```bash
python text_to_speech.py --text "I am going to the store." --output speech.wav --model "kan-bayashi/ljspeech_tacotron2"
```

## Dataset

The system is trained and evaluated on the WLASL (Word-Level American Sign Language) dataset, which contains videos of ASL signs for various words.

## Project Structure

```
asl-translation/
├── main.py                  # Main script for the complete pipeline
├── demo.py                  # Demo script with webcam support
├── sign_recognition.py      # Sign language recognition module
├── grammar_correction.py    # Grammar correction module
├── text_to_speech.py        # Text to speech module
├── train_sign_model.py      # Training script for the sign language model
├── requirements.txt         # Required dependencies
├── models/                  # Directory for trained models
├── data/                    # Directory for datasets
└── README.md                # Project documentation
```

## Limitations and Future Work

- Currently limited to isolated sign recognition (single words).
- Does not handle continuous sign language.
- Uses a small subset of the WLASL dataset for demonstration.
- Future work will focus on extending to continuous sign language recognition and improving the grammar correction for sign language specific patterns.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The WLASL dataset creators
- MediaPipe team for their pose estimation tools
- Hugging Face for their transformer models
- ESPnet team for their TTS implementation
