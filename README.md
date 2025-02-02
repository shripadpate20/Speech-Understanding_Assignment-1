# Speech Understanding and Emotion Recognition System

A comprehensive system for speech emotion recognition (SER), environmental sound classification, and music genre analysis using deep learning techniques.

## Features

- Speech Emotion Recognition (SER) with multiple windowing techniques
- Environmental sound classification using UrbanSound8K dataset
- Music genre spectrogram analysis
- Custom STFT implementation with different windowing methods
- ResNet50-based audio classification

## Dataset Requirements

- UrbanSound8K dataset
- Music genre samples (Rock, Classical, Jazz, Hip-Hop)

## Installation

```bash
pip install -r requirements.txt
```

Required packages:
- torch
- torchaudio
- librosa
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- tqdm

## Project Structure

```
├── models/
│   └── ResNetAudioClassifier.py
├── utils/
│   ├── windowing.py
│   └── datasets.py
├── experiments/
│   └── WindowingExperiment.py
└── analysis/
    └── spectrogram_analysis.py
```

## Usage

### 1. Data Preprocessing

```python
from utils.datasets import UrbanSoundDataset

dataset = UrbanSoundDataset(
    metadata_path='path/to/metadata.csv',
    dataset_path='path/to/audio',
    window_type='hann'
)
```

### 2. Training the Model

```python
from experiments.WindowingExperiment import WindowingExperiment

experiment = WindowingExperiment(DATASET_PATH, METADATA_PATH)
metrics = experiment.evaluate_all(num_epochs=5)
experiment.plot_results()
```

### 3. Music Genre Analysis

```python
from analysis.spectrogram_analysis import analyze_genres
analyze_genres(files_dict)
```

## Key Components

### Windowing Techniques
- Hann Window
- Hamming Window
- Rectangular Window

### Model Architecture
- Modified ResNet50 for audio classification
- Custom input layer for spectrogram processing
- Dropout layers for regularization
- AdamW optimizer with OneCycleLR scheduler

## Results

The system achieves competitive performance across different tasks:
- Environmental sound classification using UrbanSound8K dataset
- Comparative analysis of different windowing techniques
- Visualization of spectral characteristics across music genres

## Applications

1. **Healthcare & Mental Health**
   - Mental health monitoring
   - Stress analysis
   - Therapeutic support

2. **Customer Service**
   - Sentiment analysis
   - Call center quality assurance
   - Emotion-adaptive chatbots

3. **Security & Surveillance**
   - Emergency call analysis
   - Threat detection

4. **Low-Resource Applications**
   - Edge computing
   - Mobile devices
   - IoT applications

## Future Work

- Implementation of Wav2Vec2 and WavLM models
- Integration of multi-modal emotion recognition
- Dataset distillation techniques
- Robustness testing and optimization

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details

## References

1. Task2 audio link
2. [Wav2Small: Distilling Wav2Vec2 to 72K parameters](https://arxiv.org/abs/2408.13920)
3. [Hugging Face Wav2Small](https://huggingface.co/dkounadis/wav2small)
4. [Urban Sound Classification](https://github.com/smitkiri/urban-sound-classification)
5. [Librosa Documentation](https://librosa.org/doc/latest/index.html)
6. [PyTorch Documentation](https://pytorch.org/)

## Acknowledgments

- Department of Mathematics, Indian Institute of Technology, Jodhpur
- Data and Computational Sciences (DCS) program
