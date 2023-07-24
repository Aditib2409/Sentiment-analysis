# Sentiment-analysis
This project focuses on the hands-on experience with text representations and using text classification for sentiment analysis. \\
Here, sentiment analysis is extensively used to study customer behaviors using reviews and survey responses, online and social media, and healthcare materials for marketing and customer service applications. 

## Dataset 
Amazon reviews dataset - real reviews for jewelry products sold on Amazon. The dataset can be downloaded @ <a href="https://www.google.com/](https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_ us_Beauty_v1_00.tsv.gz" target="_blank">Here</a>. We create a 3-class classification problem according to the ratings. since the original dataset is large, I selected 20000 random reviews from each rating class to avoid any computational burden and create a balanced dataset to perform the required tasks on the downsized dataset. The dataset is split into 80% training and 20% testing datasets. 

## Data Preprocessing
NLTK package is used to remove stop words and perform lemmatization. 

## Feature Extraction
TF-IDF features are extracted for effective feature selection.

## Models
Trained a Perceptron model, SVM model, Multinomial Naive Bayes, and Logistic Regression on the training dataset

## Metrics
Precision, Recall, f1-score

