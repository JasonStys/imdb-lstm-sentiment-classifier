# IMDB LSTM Sentiment Classifier

This repository contains a CS478 deep learning notebook that builds and evaluates Long Short Term Memory models for binary sentiment classification using the IMDB movie review dataset. The project uses Keras to process tokenized review sequences, train an LSTM based neural network, evaluate predictions with validation accuracy and ROC AUC, and compare a baseline model with an improved model.

## Project Overview

The project demonstrates how LSTM networks can be used for natural language sentiment classification. The notebook loads tokenized IMDB movie reviews, pads each review to a fixed sequence length, uses an embedding layer to create trainable word vectors, applies an LSTM layer to learn sequential text patterns, and predicts whether each movie review is positive or negative.

The notebook includes a baseline LSTM model and an improved LSTM model with adjusted embedding size, LSTM size, dropout, learning rate, batch size, checkpointing, and early stopping.

## Dataset

The project uses the Keras IMDB movie review dataset.

```text
Training reviews: 25,000
Validation reviews: 25,000
Task: Binary sentiment classification
Labels: 0 for negative review, 1 for positive review
Vocabulary size used: 10,000 words
Maximum review length: 100 tokens
Padding and truncation: Pre
```

## Preprocessing

The notebook prepares the IMDB review data by:

- Loading tokenized movie reviews from Keras
- Restricting the vocabulary to the 10,000 most common words
- Padding shorter reviews to 100 tokens
- Truncating longer reviews to 100 tokens
- Using binary labels for sentiment classification

## Baseline LSTM Model

The baseline model uses an embedding layer, spatial dropout, an LSTM layer, and a sigmoid output layer.

```text
Embedding dimension: 64
Vocabulary size: 10,000
Maximum review length: 100
Spatial dropout: 0.2
LSTM units: 256
LSTM dropout: 0.2
Output activation: Sigmoid
Loss function: Binary cross entropy
Optimizer: Adam
Epochs: 4
Batch size: 128
```

## Baseline Model Summary

```text
Embedding parameters: 640,000
LSTM parameters: 328,704
Output parameters: 257
Total parameters: 968,961
Trainable parameters: 968,961
```

## Baseline Results

The baseline model reached its best validation accuracy at epoch 4.

```text
Base model best validation accuracy: 84.67 percent
Base model best epoch: 4
Base model ROC AUC: 92.62
```

The training curves show that training accuracy continued rising while validation accuracy improved more slowly, suggesting the model was beginning to overfit.

## Improved LSTM Model

The improved model uses a larger embedding layer, fewer LSTM units, higher dropout, a smaller learning rate, a smaller batch size, model checkpointing, and early stopping.

```text
Embedding dimension: 128
Vocabulary size: 10,000
Maximum review length: 100
Spatial dropout: 0.3
LSTM units: 128
LSTM dropout: 0.3
Learning rate: 0.0005
Epochs: 8
Batch size: 64
Model checkpointing: Enabled
Early stopping: Enabled
```

## Improved Model Summary

```text
Embedding parameters: 1,280,000
LSTM parameters: 131,584
Output parameters: 129
Total parameters: 1,411,713
Trainable parameters: 1,411,713
```

## Improved Model Results

The improved model performed better than the baseline model.

```text
Improved model best validation accuracy: 85.50 percent
Improved model best epoch: 2
Improved model ROC AUC: 93.38
```

## Model Comparison

| Model | Best Validation Accuracy | Best Epoch | ROC AUC |
| --- | ---: | ---: | ---: |
| Baseline LSTM | 84.67 percent | 4 | 92.62 |
| Improved LSTM | 85.50 percent | 2 | 93.38 |

The improved model increased validation accuracy by 0.83 percentage points and improved ROC AUC by 0.76 points.

## Key Takeaways

- LSTM networks can model review text as sequences rather than treating words as independent features.
- Embedding layers create trainable vector representations for tokenized words.
- The improved LSTM model outperformed the baseline model on both validation accuracy and ROC AUC.
- Early stopping helped preserve the best validation model before later epochs could overfit.
- ROC AUC provides a useful evaluation metric beyond accuracy for binary sentiment classification.

## Visualizations

The notebook includes:

- Baseline training accuracy versus validation accuracy
- Baseline training loss versus validation loss
- Baseline prediction probability histogram
- Baseline ROC curve
- Improved training accuracy versus validation accuracy
- Improved training loss versus validation loss
- Improved prediction probability histogram
- Improved ROC curve

## Technologies Used

- Python
- Keras
- TensorFlow backend
- scikit-learn
- NumPy
- Matplotlib
- Jupyter Notebook
- Google Colab

## Repository Structure

```text
.
├── Jason_Stys_CS478_01_CA14_NLP_LSTM.ipynb
├── Jason Stys CS478-01 CA14 NLP LSTM.pdf
├── README.md
└── requirements.txt
```

Optional generated folders after running the notebook:

```text
model_output/
├── LSTM_base/
└── LSTM_improved/
```

## How to Run

Clone the repository.

```bash
git clone https://github.com/your-username/imdb-lstm-sentiment-classifier.git
cd imdb-lstm-sentiment-classifier
```

Install the required packages.

```bash
pip install tensorflow keras numpy matplotlib scikit-learn notebook
```

Launch Jupyter Notebook.

```bash
jupyter notebook
```

Open the notebook file and run all cells from top to bottom.

## Skills Demonstrated

- Natural language processing with tokenized review sequences
- IMDB sentiment classification
- Keras dataset loading
- Sequence padding and truncation
- Embedding layer design
- Spatial dropout
- LSTM sequence modeling
- Binary classification with sigmoid output
- Model checkpointing
- Early stopping
- ROC AUC evaluation
- ROC curve visualization
- Training and validation curve analysis
- Jupyter Notebook workflow

## Possible Future Improvements

- Compare LSTM results against the dense, Conv1D, and SimpleRNN sentiment models in one summary table
- Add confusion matrix and precision recall evaluation
- Add bidirectional LSTM layers
- Increase maximum review length
- Tune vocabulary size and embedding dimension
- Use pretrained word embeddings
- Save the best trained model for reuse
