__version__ = "3.8.10"
import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('omw-1.4')
import re
import contractions
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from bs4 import BeautifulSoup

data = pd.read_table('https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Beauty_v1_00.tsv.gz', sep='\t', on_bad_lines='skip')

# Keep reviews and ratings
columns = ['star_rating', 'review_body']
new_data = data[columns]

# assigning labels based on ratings
new_data.insert(2, "label", np.ones((new_data.shape[0], 1)), True)
# class 3
new_data.loc[new_data['star_rating'] == '5', 'label'] = int(3)
new_data.loc[new_data['star_rating'] == 5, 'label'] = int(3)
new_data.loc[new_data['star_rating'] == '4', 'label'] = int(3)
new_data.loc[new_data['star_rating'] == 4, 'label'] = int(3)

# class 2 
new_data.loc[new_data['star_rating'] == '3', 'label'] = int(2)
new_data.loc[new_data['star_rating'] == 3, 'label'] = int(2)

# class 1
new_data.loc[new_data['star_rating'] == '1', 'label'] = int(1)
new_data.loc[new_data['star_rating'] == 1, 'label'] = int(1)
new_data.loc[new_data['star_rating'] == '2', 'label'] = int(1)
new_data.loc[new_data['star_rating'] == 2, 'label'] = int(1)

new_data['label'] = new_data['label'].astype(np.int64)

# randomly sampling 20,000 reviews per class
new_data_1 = new_data[new_data['label'] == 1].sample(n=20000)

new_data_2 = new_data[new_data['label'] == 2].sample(n=20000)

new_data_3 = new_data[new_data['label'] == 3].sample(n=20000)


df = pd.concat([new_data_1, new_data_2, new_data_3])
df.reset_index(drop=True)

"""
    DATA CLEANING
"""
# average character length of reviews before cleaning
before_cleaning = np.mean(df['review_body'].str.len())

# convert all reviews into lower case
df['review_body'] = df['review_body'].str.replace('\n', ' ').str.lower()

# Remove extra espaces
df['review_body'] = df['review_body'].apply(lambda x : re.sub(' +', ' ', str(x)))

# Performing contractions
df['review_body'] = df['review_body'].apply(lambda x: contractions.fix(str(x)))

print(f'Average length before and after data cleaning: -')
after_cleaning = np.mean(df['review_body'].str.len())
print(f'{before_cleaning}, {after_cleaning}')

"""
    PREPROCESSING
"""
# average character length of reviews before preprocessing
before_preprocessing = np.mean(df['review_body'].str.len())

## Removing stop words
english_stopwords = stopwords.words('english')
# convert text to lower case and split to a list of words
df['review_body'] = df['review_body'].apply(lambda x: ' '.join([word for word in x.split() if word not in (english_stopwords)]))

## Lemmetization
lemmatizer = WordNetLemmatizer()
df['review_body'] = df['review_body'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split() if word not in (english_stopwords)]))

# print the average length of the reviews in terms of character length in your dataset after cleaning
print(f'Before and After Preprocessing: -')
after_preprocessing = np.mean(df['review_body'].str.len())
print(f'{before_preprocessing}, {after_preprocessing}')

"""
    SPLITTING THE DATASET INTO TRAIN AND TEST
"""
# split the dataframe into train and test dataframes
train_df, test_df = train_test_split(df, test_size=0.2)

"""
    FEATURE EXTRACTION - TF IDF
"""

tfidfvectorizer = TfidfVectorizer(analyzer='word', stop_words='english', max_features=1000)

# feature extraction on the training set
tfidf_tr = tfidfvectorizer.fit_transform(train_df['review_body'])
tokens_tfidf = tfidfvectorizer.get_feature_names()
df_tfidf_tr = pd.DataFrame(data = tfidf_tr.toarray(),columns = tokens_tfidf)

# feature extraction on the testing set
tfidf_ts = tfidfvectorizer.transform(test_df['review_body'])
df_tfidf_ts = pd.DataFrame(data = tfidf_ts.toarray(), columns = tokens_tfidf)

# adjusted training dataframe
train_ratings = np.array(train_df['star_rating'])
train_labels = np.array(train_df['label'])
df_tfidf_tr['star_rating'] = train_ratings
df_tfidf_tr.insert(1001, "Target label", train_labels, True)

# adjusted testing dataframe
test_ratings = np.array(test_df['star_rating'])
test_labels = np.array(test_df['label'])
df_tfidf_ts['star_rating'] = test_ratings
df_tfidf_ts.insert(1001, "target label", test_labels, True)

"""
    MODEL TRAINING
"""

## PERCEPTRON

perceptron = Perceptron(tol=1e-3, random_state=0, max_iter=10)
perceptron.fit(df_tfidf_tr.iloc[:,:-2], df_tfidf_tr['Target label'])

y_pred = perceptron.predict(df_tfidf_tr.iloc[:,:-2]) 

print('Perceptron:')
print('Scores on Training set')

# precision scores per class
pr = precision_score(df_tfidf_tr['Target label'], y_pred, average=None)
for p in pr:
    print(p)

# recall score per class
rc = recall_score(df_tfidf_tr['Target label'], y_pred, average=None)
for r in rc:
    print(r)

# F-1 score per class
f1 = f1_score(df_tfidf_tr['Target label'], y_pred, average=None)
for f in f1:
    print(f)

print('Average Scores on Test set')
# averages for test set
y_ts_pred = perceptron.predict(df_tfidf_ts.iloc[:,:-2]) 

# precision score
pr_ts = precision_score(df_tfidf_ts['target label'], y_ts_pred, average=None)
print(np.mean(pr_ts))

# recall score
rc_ts = recall_score(df_tfidf_ts['target label'], y_ts_pred, average=None)
print(np.mean(rc_ts))

# F-1 score per class
f1_ts = f1_score(df_tfidf_ts['target label'], y_ts_pred, average=None)
print(np.mean(f1_ts))

## SVM

svm = make_pipeline(StandardScaler(), SVC(kernel='linear'))
svm.fit(df_tfidf_tr.iloc[:,:-2], df_tfidf_tr['Target label'])
y_pred = svm.predict(df_tfidf_tr.iloc[:,:-2]) 

print('Scores on Training set')
# precision score
pr = precision_score(df_tfidf_tr['Target label'], y_pred, average=None)
for p in pr:
    print(p)

# recall score
rc = recall_score(df_tfidf_tr['Target label'], y_pred, average=None)
for r in rc:
    print(r)

# F-1 score per class
f1 = f1_score(df_tfidf_tr['Target label'], y_pred, average=None)
for f in f1:
    print(f)

print('Average Scores on Test set')
# averages for test set
y_ts_pred = svm.predict(df_tfidf_ts.iloc[:,:-2]) 

# precision score
pr_ts = precision_score(df_tfidf_ts['target label'], y_ts_pred, average=None)
print(np.mean(pr_ts))

# recall score
rc_ts = recall_score(df_tfidf_ts['target label'], y_ts_pred, average=None)
print(np.mean(rc_ts))

# F-1 score per class
f1_ts = f1_score(df_tfidf_ts['target label'], y_ts_pred, average=None)
print(np.mean(f1_ts))

## LOGISTIC REGRESSION

lr = LogisticRegression(random_state=0, max_iter=200).fit(df_tfidf_tr.iloc[:,:-2], df_tfidf_tr['Target label'])
y_pred = lr.predict(df_tfidf_tr.iloc[:,:-2]) 

print('Logistic Regression:')
print('Scores on Training set')
# precision score
pr = precision_score(df_tfidf_tr['Target label'], y_pred, average=None)
for p in pr:
    print(p)

# recall score
rc = recall_score(df_tfidf_tr['Target label'], y_pred, average=None)
for r in rc:
    print(r)

# F-1 score per class
f1 = f1_score(df_tfidf_tr['Target label'], y_pred, average=None)
for f in f1:
    print(f)

print('Average Scores on Test set')
# averages for test set
y_ts_pred = lr.predict(df_tfidf_ts.iloc[:,:-2]) 

# precision score
pr_ts = precision_score(df_tfidf_ts['target label'], y_ts_pred, average=None)
print(np.mean(pr_ts))

# recall score
rc_ts = recall_score(df_tfidf_ts['target label'], y_ts_pred, average=None)
print(np.mean(rc_ts))

# F-1 score per class
f1_ts = f1_score(df_tfidf_ts['target label'], y_ts_pred, average=None)
print(np.mean(f1_ts))

## MULTINOMIAL NAIVE BAYES
nb = MultinomialNB()
nb.fit(df_tfidf_tr.iloc[:,:-2], df_tfidf_tr['Target label'])
y_pred = nb.predict(df_tfidf_tr.iloc[:,:-2]) 

print('Multinomial Naive Bayes: ')
print('Scores on Training set')
# precision score
pr = precision_score(df_tfidf_tr['Target label'], y_pred, average=None)
for p in pr:
    print(p)

# recall score
rc = recall_score(df_tfidf_tr['Target label'], y_pred, average=None)
for r in rc:
    print(r)

# F-1 score per class
f1 = f1_score(df_tfidf_tr['Target label'], y_pred, average=None)
for f in f1:
    print(f)

# averages for test set
y_ts_pred = lr.predict(df_tfidf_ts.iloc[:,:-2]) 

print('Average Scores on Test set')

# precision score
pr_ts = precision_score(df_tfidf_ts['target label'], y_ts_pred, average=None)
print(np.mean(pr_ts))

# recall score
rc_ts = recall_score(df_tfidf_ts['target label'], y_ts_pred, average=None)
print(np.mean(rc_ts))

# F-1 score per class
f1_ts = f1_score(df_tfidf_ts['target label'], y_ts_pred, average=None)
print(np.mean(f1_ts))