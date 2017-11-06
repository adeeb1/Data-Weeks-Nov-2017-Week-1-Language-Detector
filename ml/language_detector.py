"""Build a language detector model

The goal of this exercise is to train a linear classifier on text features
that represent sequences of up to N consecutive characters so as to be
recognize natural languages by using the frequencies of short character
sequences as 'fingerprints'.

The script saves the trained model to disk for later use
"""
# Author: Olivier Grisel <olivier.grisel@ensta.org>
# License: Simplified BSD
# Adapted by: Francesco Mosconi

import numpy as np
import pandas as pd
import pdb
import gzip
import dill
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import confusion_matrix, classification_report

# The training data folder must be passed as first argument
try:
    dataset = load_files('./wikidata/short_paragraphs')
except OSError as ex:
    print(ex)
    print("Couldn't import the data, did you unzip the wikidata.zip folder?")
    exit(-1)


# TASK: Split the dataset in training and test set
# (use 20% of the data for test):

docs = dataset['data']
targets = dataset['target']

target_names = dataset['target_names']

x_train, x_test, y_train, y_test = train_test_split(docs, targets, test_size=0.20, random_state=0)

# TASK: Build a an vectorizer that splits
# strings into sequence of 1 to 3
# characters instead of word tokens
# using the class TfidfVectorizer

# The vectorize splits the characters into ngrams
vectorizer = TfidfVectorizer(ngram_range=(1, 7), analyzer="char", use_idf=True)

# TASK: Use the function make_pipeline to build a
#       vectorizer / classifier pipeline
#       using the previous analyzer
#       and a classifier of choice.
#       The pipeline instance should be
#       stored in a variable named model

model = make_pipeline(vectorizer, LinearSVC(random_state=0))

# TASK: Fit the pipeline on the training set

model.fit(x_train, y_train)

# TASK: Predict the outcome on the testing set.
# Store the result in a variable named y_predicted

y_predicted = model.predict(x_test)

# TASK: Print the classification report

print("Classification report: ", classification_report(y_test, y_predicted, target_names=target_names))

# TASK: Print the confusion matrix. Bonus points if you make it pretty.

cm = confusion_matrix(y_test, y_predicted)

predicted_names = map(lambda name: "p_" + name, target_names)
df_matrix = pd.DataFrame(cm, columns=predicted_names, index=target_names)

print("\n=========== Confusion Matrix ==============")
print("")
print(df_matrix)
print("\n============== End Matrix =================")

# TASK: Is the score good? Can you improve it changing
#       the parameters or the classifier?
#       Try using cross validation and grid search

# Score with LinearSVC: 0.971
# Yes, the score is great

# TASK: Use dill and gzip to persist the trained model in memory.
#       1) gzip.open a file called my_model.dill.gz
#       2) dump to the file both your trained classifier
#          and the target_names of the dataset (for later use)
#    They should be passed as a list [model, dataset.target_names]

with gzip.open('my_model.dill.gz', 'wb') as f:
    file_contents = dill.dumps([model, target_names])
    f.write(file_contents)
