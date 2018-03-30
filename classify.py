from word_features import extract

import pickle
import sys

try:
    f = open('classifier.pickle', 'rb')
except FileNotFoundError:
    print('Classifier not found')
    exit()

classifier = pickle.load(f)
f.close()

try:
    features = extract(sys.argv[1])
except IndexError:
    print('No text supplied to classify')

print(classifier.classify(features))
