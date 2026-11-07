# RNN and LSTM Implementation for Speech Recognition

## Project Overview
This project implements Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) networks for speech command classification.

## Objectives
- Implement a simple RNN using NumPy.
- Understand Backpropagation Through Time (BPTT).
- Compare NumPy RNN calculations with PyTorch.
- Implement an LSTM forward pass using NumPy.
- Perform speech command classification using RNN and LSTM.

## Dataset
The project uses a speech commands dataset containing five classes:
- One
- Two
- Three
- Right
- Left

Total audio samples: 1000.

## Preprocessing
- Audio sampling rate: 16 kHz
- Feature extraction: 13 MFCC coefficients
- Frame length: 25 ms
- Hop length: 10 ms
- Stratified train/validation/test split: 70% / 15% / 15%
- Feature normalization and sequence padding.

## Models
### RNN
- Input features: 13
- Hidden units: 64
- Number of layers: 1
- Optimizer: Adam
- Learning rate: 0.001
- Epochs: 10

### LSTM
- Input projection: 13 to 64
- Hidden units: 64
- Number of layers: 1
- Optimizer: Adam
- Learning rate: 0.001
- Epochs: 10

## Results
Training accuracy, validation accuracy, test accuracy, and confusion matrices are included in the results folder/archive.

## Files
- `Lab51_RNN.ipynb` – Implementation and experiments.
- `Lab5_Results.zip` – Generated plots and trained models.
- `requirements.txt` – Python dependencies.

## How to Run
Open the notebook using Google Colab or Jupyter Notebook, install the required libraries, and provide the speech commands dataset in the expected directory structure.
