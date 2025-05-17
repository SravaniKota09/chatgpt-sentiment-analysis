# chatgpt-sentiment-analysis
chatgpt sentiment analysis using tf-idf
This Jupyter notebook performs sentiment analysis on tweets about ChatGPT using machine learning techniques.
dataset download from - 'https://www.kaggle.com/datasets/charunisa/chatgpt-sentiment-analysis'


ChatGPT Sentiment Analysis Notebook Description

This Jupyter notebook performs sentiment analysis on tweets about ChatGPT using machine learning techniques. Here's a detailed explanation of how the code works:

Overview
The code analyzes tweet data labeled with sentiments (good, bad, neutral) to build a classification model that can predict sentiment from text content.

Key Components
1. Data Loading and Initial Exploration
- Imports necessary libraries (matplotlib, seaborn, numpy, pandas)
- Loads the dataset from 'chatgpt.csv'
- Displays the first few rows showing tweet text and sentiment labels

2. Data Preprocessing
- Uses LabelEncoder to convert text labels ('good', 'bad', 'neutral') to numerical values (1, 0, 2)
- Filters the dataset to keep only 'good' (1) and 'bad' (0) sentiment tweets, removing 'neutral'
- Adds a 'length' feature showing character count of each tweet
- Removes punctuation from tweets and stores in new 'nopunc' column

3. Exploratory Data Analysis
- Creates a countplot showing distribution of sentiment labels
- Visual analysis reveals more negative (0) than positive (1) tweets in the dataset

4. Feature Engineering
- Uses CountVectorizer to convert text into numerical feature vectors (word counts)
- Splits data into training (70%) and testing (30%) sets

5. Model Building
- Implements a Multinomial Naive Bayes classifier (well-suited for text classification)
- First approach: Basic CountVectorizer + Naive Bayes
- Second approach: Pipeline with CountVectorizer → TF-IDF Transformer → Naive Bayes

6. Model Evaluation
- Predicts on test set and prints classification report showing:
    - Precision, recall, and F1-score for each class
    - Overall accuracy and weighted averages

How the Code Works
1. Data Preparation: The raw tweet data is cleaned by removing punctuation and converting text labels to numbers that the model can process.
2. Feature Extraction: The text data is transformed into numerical features using:
- CountVectorizer: Counts word frequencies
- TF-IDF Transformer: Adjusts word counts by importance (rare words get more weight)
3. Model Training: The Naive Bayes algorithm learns patterns in the word frequencies that correlate with positive or negative sentiment.
4. Evaluation: The model's performance is measured on unseen test data using standard classification metrics.

Results
The basic CountVectorizer + Naive Bayes approach achieved:
- 91% overall accuracy
- Good performance on both classes (F1-scores: 0.93 for negative, 0.87 for positive)
The pipeline with TF-IDF transformation performed slightly worse (84% accuracy), suggesting in this case that simple word counts may be more effective than TF-IDF weighting for sentiment analysis of these tweets.

