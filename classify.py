from word_features import extract

import pickle


f = open('classifier.pickle', 'rb')
classifier = pickle.load(f)
f.close()

features = extract('I am using this amazing classifier')
print(classifier.classify(features))