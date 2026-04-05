# SMS Spam Classifier

A machine learning project that classifies SMS messages as **spam** or **ham** (not spam) using Natural Language Processing (NLP) techniques and a Naive Bayes classifier.

---

## Overview

This project demonstrates a complete NLP pipeline — from raw text preprocessing to model training and real-time prediction. It is built using Python and trained on a real-world SMS spam dataset.

---

## Features

- Text preprocessing with tokenization, stopword removal, stemming, and lemmatization
- TF-IDF vectorization for feature extraction
- Multinomial Naive Bayes classification
- Model evaluation with accuracy, precision, recall, F1 score, and confusion matrix
- Interactive command-line prediction for custom messages

---

## Project Structure

```
mdm_assignment/
├── main.py          # Main script — preprocessing, training, evaluation, prediction
├── dataset.csv      # SMS spam dataset (tab-separated)
├── requirements.txt # Python dependencies
└── README.md
```

---

## NLP Pipeline

```
Raw Text
   |
   v
Lowercase  -->  Remove Punctuation  -->  Tokenize
   |
   v
Remove Stopwords  -->  Stemming  -->  Lemmatization
   |
   v
TF-IDF Vectorization  -->  Naive Bayes Model  -->  Prediction
```

---

## Dataset

The dataset is the [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset), containing 5,572 SMS messages labelled as `ham` or `spam`.

| Label | Description        |
|-------|--------------------|
| ham   | Legitimate message |
| spam  | Spam message       |

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/your-username/mdm_assignment.git
cd mdm_assignment
```

**2. Install dependencies**
```bash
pip install pandas numpy nltk scikit-learn
```

**3. Run the script**
```bash
python main.py
```

---

## Usage

After training completes, the program enters an interactive mode:

```
Enter message (or 'exit'): Congratulations! You have won a free prize, call now!
Spam Message

Enter message (or 'exit'): Hey, are you coming to the meeting tomorrow?
Ham Message

Enter message (or 'exit'): exit
```

---

## Model Evaluation

The model is evaluated on a 70/30 train-test split using the following metrics:

| Metric    | Description                              |
|-----------|------------------------------------------|
| Accuracy  | Overall correct predictions              |
| Precision | Spam messages correctly identified       |
| Recall    | Actual spam messages caught              |
| F1 Score  | Harmonic mean of precision and recall    |

---

## Technologies Used

| Library      | Purpose                        |
|--------------|--------------------------------|
| pandas       | Data loading and manipulation  |
| numpy        | Numerical operations           |
| nltk         | NLP preprocessing              |
| scikit-learn | Vectorization and ML model     |

---

## License

This project is for academic/assignment purposes.
