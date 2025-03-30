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

## Dataset

The system is trained and evaluated on the WLASL (Word-Level American Sign Language) dataset, which contains videos of ASL signs for various words.

## Requirements

- Python 3.8+
- PyTorch
- OpenCV
- MediaPipe
- Transformers
- ESPnet
- SoundFile

## Usage

The repository includes a Jupyter notebook that demonstrates the complete pipeline:

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/sign-language-translation-system.git
   cd sign-language-translation-system
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Open the Jupyter notebook:
   ```
   jupyter notebook sign_language_translation_system.ipynb
   ```

4. Follow the instructions in the notebook to run the complete pipeline.

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