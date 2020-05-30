# NLP

# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

# download stop words
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

classifier = GaussianNB()
cv = CountVectorizer(max_features=1500)
# loading dataset
dataset = pd.read_csv('Python/Restaurant_Reviews.tsv', delimiter='\t',
                      quoting=3)
corpus = []

for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word=word) for word in review if not word in
                                                           set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)

# creating the bags of words
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# creating data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0,
                                                    test_size=0.20)

# training the naive base model
classifier.fit(X_train, y_train)

# predictions
y_pred = classifier.predict(X_test)

# check accuracy
cm = confusion_matrix(y_test, y_pred)
print(cm)

acc = accuracy_score(y_test, y_pred)
print(acc)
