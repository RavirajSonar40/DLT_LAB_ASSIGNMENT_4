# SMS Spam Classification using NLP

## Project Overview

This project implements a **Text Classification system** to automatically classify SMS messages as **Ham** (legitimate) or **Spam** using Natural Language Processing (NLP) techniques and Machine Learning. The model uses a **Multinomial Naive Bayes** classifier with **TF-IDF vectorization** to achieve high accuracy in spam detection.

---

## Dataset Details

### Dataset Statistics

| Property | Value |
|----------|-------|
| Total Messages | 5,572 |
| Ham (Legitimate) | 4,825 messages (86.6%) |
| Spam | 747 messages (13.4%) |
| Format | Tab-separated CSV with two columns |

- **Column 1 (label):** Label indicating whether the message is `ham` (legitimate) or `spam`
- **Column 2 (text):** The raw text of the SMS message

### Sample Dataset Rows

| Label | Message |
|-------|---------|
| ham | "Go until jurong point, crazy.. Available only in bugis n great world la e buffet..." |
| ham | "Ok lar... Joking wif u oni..." |
| spam | "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121..." |
| ham | "U dun say so early hor... U c already then say..." |
| ham | "Nah I don't think he goes to usf, he lives around here though" |

---

### Dataset Sources & Corpus Information

This SMS Spam Collection dataset has been compiled from multiple free or research-available sources:

**1. Grumbletext Web Forum SMS Spam**
- Size: 425 SMS spam messages
- Source: Grumbletext Web site — a UK forum where cell phone users report SMS spam
- Collection Method: Manually extracted from public complaints

**2. NUS SMS Corpus (NSC) — Ham Messages**
- Size: 3,375 SMS randomly selected legitimate messages
- Source: National University of Singapore, Department of Computer Science
- Origin: Primarily from Singaporeans and University students
- Collection Method: Volunteers were informed their contributions would be made publicly available

**3. Caroline Tag's PhD Thesis SMS Collection**
- Size: 450 SMS ham (legitimate) messages
- Source: Academic research collection from PhD thesis

**4. SMS Spam Corpus v.0.1 Big**
- Ham Messages: 1,002 SMS
- Spam Messages: 322 SMS
- Total: 1,324 SMS messages

---

## Project Workflow

### Step 1: Install Requirements
```bash
pip install -r requirements.txt
```

### Step 2: Data Loading and Cleaning
- Load dataset from `dataset.csv`
- Filter valid labels (`ham` / `spam`) and drop rows with missing values
- Columns mapped to `label` and `text` for clarity

### Step 3: NLP Preprocessing

The following preprocessing techniques are applied to normalize and clean the text data:

| Step | Description |
|------|-------------|
| Lowercase Conversion | Convert all text to lowercase |
| Punctuation Removal | Remove special characters and punctuation |
| Tokenization | Break text into individual words |
| Stopword Removal | Eliminate common English words (e.g., "the", "a", "is") |
| Stemming | Reduce words to root form using Porter Stemmer (e.g., "running" → "run") |
| Lemmatization | Convert words to dictionary form (e.g., "walked" → "walk") |

### Step 4: Feature Extraction
- **TF-IDF Vectorization:** Convert processed text into numerical features
- **Maximum Features:** 3,000 terms
- **Output Shape:** (5572, 3000)

### Step 5: Train-Test Split
- **Training Data:** 70% (3,900 messages)
- **Testing Data:** 30% (1,672 messages)
- **Random State:** 42 (for reproducibility)

### Step 6: Model Training
- **Algorithm:** Multinomial Naive Bayes
- **Training:** Fitted on training data with TF-IDF features

### Step 7: Model Evaluation

---

## Performance Metrics

| Metric | Score |
|--------|-------|
| Accuracy | 96.76% |
| Precision (Spam) | 100% |
| Recall (Spam) | 74.88% |
| F1-Score (Spam) | 85.89% |

### Detailed Classification Report

```
              precision    recall  f1-score   support

           0       0.96      0.98      0.97      1443
           1       1.00      0.75      0.86       229

    accuracy                           0.97      1672
   macro avg       0.98      0.86      0.91      1672
weighted avg       0.97      0.97      0.97      1672
```

### Confusion Matrix

```
                 Predicted Ham    Predicted Spam
Actual Ham           1413              30
Actual Spam            58             171
```

| Term | Value | Description |
|------|-------|-------------|
| True Negatives (TN) | 1,413 | Correctly classified as ham |
| True Positives (TP) | 171 | Correctly classified as spam |
| False Negatives (FN) | 58 | Spam classified as ham |
| False Positives (FP) | 30 | Ham classified as spam |

---

## Key Findings

- **High Accuracy:** The model achieves **96.76% accuracy**, demonstrating excellent overall performance
- **Perfect Precision:** 100% precision for spam detection means no legitimate messages are incorrectly flagged as spam
- **Good Recall:** 74.88% recall indicates the model catches approximately 75% of actual spam messages
- **Balanced Performance:** The model shows good balance between sensitivity and specificity

---

## How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/mdm_assignment.git
cd mdm_assignment
```

### 2. Install Dependencies
```bash
pip install pandas numpy nltk scikit-learn
```

### 3. Run the Script
```bash
python main.py
```

### 4. Interactive Prediction
After training completes, the program enters interactive mode:
```
Enter message (or 'exit'): Congratulations! You have won a free prize, call now!
Spam Message

Enter message (or 'exit'): Hey, are you coming to the meeting tomorrow?
Ham Message

Enter message (or 'exit'): exit
```

---

## Technologies Used

| Library | Purpose |
|---------|---------|
| Python 3.x | Core programming language |
| pandas | Data manipulation and analysis |
| numpy | Numerical computations |
| nltk | NLP preprocessing (tokenization, stemming, lemmatization) |
| scikit-learn | TF-IDF vectorization, Naive Bayes, evaluation metrics |

---

## Project Structure

```
mdm_assignment/
├── main.py              # Main script with complete pipeline
├── dataset.csv          # SMS spam dataset (5,572 messages)
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## Future Improvements

- **Advanced Models:** Implement SVM, Random Forest, or Deep Learning models (LSTM/CNN)
- **Hyperparameter Tuning:** Optimize TF-IDF and model parameters
- **Class Balancing:** Address the class imbalance (ham: 86.6%, spam: 13.4%)
- **Cross-Validation:** Implement k-fold cross-validation for robust evaluation
- **Feature Engineering:** Extract domain-specific features (phone numbers, URLs, etc.)
- **Ensemble Methods:** Combine multiple models for better predictions

---

## References

- [NLTK Documentation](https://www.nltk.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [SMS Spam Collection Dataset — Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- NUS SMS Corpus — National University of Singapore
- Grumbletext Forum — SMS spam complaints repository

---

## License

This dataset and project are available for educational and research purposes. Please refer to the original dataset sources for specific licensing information.
