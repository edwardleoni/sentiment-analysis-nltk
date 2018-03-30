import nltk
nltk.download('movie_reviews')

import pickle

import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews


negids = movie_reviews.fileids('neg')
posids = movie_reviews.fileids('pos')

negfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
posfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'pos') for f in posids]

negcutoff = int(len(negfeats) * 3 / 4)
poscutoff = int(len(posfeats) * 3 / 4)

trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
print
'train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats))

classifier = NaiveBayesClassifier.train(trainfeats)
print
'accuracy:', nltk.classify.util.accuracy(classifier, testfeats)
classifier.show_most_informative_features()


f = open('my_classifier.pickle', 'wb')
pickle.dump(classifier, f)
f.close()

f = open('my_classifier.pickle', 'rb')
classifier = pickle.load(f)
f.close()

features = word_feats('I am using this amazing classifier')
print(classifier.classify(features))