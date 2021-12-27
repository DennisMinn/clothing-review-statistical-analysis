import re
import numpy as np
import pandas as pd

from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score

from matplotlib import pyplot as plt


class BagOfWords(object):
    """
    Class for implementing Bag of Words
    """
    def __init__(self, vocabulary_size):
        """
        Initialize the BagOfWords model
        """
        self.vocabulary_size = vocabulary_size
        self.vocab = []

    def preprocess(self, text):
        """
        Preprocessing of one Review Text
            - convert to lowercase
            - remove punctuation
            - empty spaces
            - remove 1-letter words
            - split the sentence into words

        Return the split words
        """
        text = text.lower()
        word_list = re.split(r'[\W\s]+', text)
        word_list = filter(lambda word: len(word) > 1, word_list)
        return list(word_list)

    def fit(self, X_train):
        """
        Building the vocabulary using X_train
        """
        for row in X_train:
            self.vocab.extend(self.preprocess(row))

        self.vocab = sorted(Counter(self.vocab).items(), key = lambda item: item[1], reverse = True)[: self.vocabulary_size]
        self.vocab = [word[0] for word in self.vocab]
        self.vocab.sort()
        return self.vocab

    def transform(self, X):
        """
        Transform the texts into word count vectors (representation matrix)
            using the fitted vocabulary
        """
        X = [Counter(self.preprocess(row)) for row in X]
        rep_matrix = np.array([list(map(lambda word: row[word], self.vocab)) for row in X])
        return rep_matrix


class NaiveBayes(object):
    def __init__(self, beta=1, n_classes=2):
        """
        Initialize the Naive Bayes model
            w/ beta and n_classes
        """
        self.beta = beta
        self.C = n_classes
        self.priors = np.zeros(n_classes)
        self.likelihoods = None

    def fit(self, X_train, y_train):
        """
        Fit the model to X_train, y_train
            - build the conditional probabilities
            - and the prior probabilities
        """

        N, V = X_train.shape
        self.likelihoods = np.zeros((self.C, V))

        for c in range(self.C):
            class_mask = y_train == c
            self.priors[c] = (class_mask.sum() + self.beta)/ (N + (self.C * self.beta))
            self.likelihoods[c] = (X_train[class_mask].sum(0) + self.beta) / (X_train[class_mask].sum() + (V * self.beta))

        return self.priors, self.likelihoods

    def predict(self, X_test):
        """
        Predict the X_test with the fitted model
        """
        # posterior probabilities per class
        posteriors = np.array([np.sum(row.toarray() * np.log(c))
                             for row in X_test
                             for c in self.likelihoods])

        posteriors = posteriors.reshape(X_test.shape[0], self.C) + np.log(self.priors)
        y_pred = np.argmax(posteriors, axis = 1)
        return posteriors, y_pred


def confusion_matrix(y_true, y_pred):
    """
    Calculate the confusion matrix of the
        predictions with true labels
    """
    res = np.concatenate([y_true.reshape(-1, 1),
                               y_pred.reshape(-1, 1)], axis = 1)

    res = np.array([confusion_matrix_helper(row) for row in res])

    conf_mat = np.zeros((2,2))
    conf_mat[0, 0] = np.sum(res == 0)
    conf_mat[1, 0] = np.sum(res == 1)
    conf_mat[0, 1] = np.sum(res == 2)
    conf_mat[1, 1] = np.sum(res == 3)

    return conf_mat

def confusion_matrix_helper(row):
    if row[0] == 0 and row[1] == 0:
        return 0
    elif row[0] == 1 and row[1] == 0:
        return 1
    elif row[0] == 0 and row[1] == 1:
        return 2
    else:
        return 3

def load_data(path = 'Data', return_numpy=False, debug=False):
    """
    Load data

    Params
    ------
    return_numpy:   when true return the representation of Review Text
                    using the CountVectorizer or BagOfWords
                    when false return the Review Text

    Return
    ------
    X_train
    y_train
    X_valid
    y_valid
    X_test
    """
    X_train = pd.read_csv(path + "/X_train.csv")['Review Text'].values
    X_valid = pd.read_csv(path + "/X_val.csv")['Review Text'].values
    X_test  = pd.read_csv(path + "/X_test.csv")['Review Text'].values
    y_train = (pd.read_csv(path + "/Y_train.csv")['Sentiment'] == 'Positive').astype(int).values
    y_valid = (pd.read_csv(path + "/Y_val.csv")['Sentiment'] == 'Positive').astype(int).values

    # both BoW representation and non-representation are numpy objects
    # this variable name is confusing and VERY misleading
    # also where is this part mentioned in the PDF how are students
    # suppose to know when to work on this part. This HW i.m.o is very
    # mismanaged. 
    if return_numpy:

        # To do (not for Q1.1, used in Q1.3)
        # transform the Review Text into bag of word representation using vectorizer
        # process X_train, X_valid, X_test

        X_train = X_train.tolist()

        bow = CountVectorizer(token_pattern = '[\w]{2,}') if not debug else CountVectorizer(X_train, token_pattern = '[\w]{2,}', max_features = 10)
        X_train = bow.fit_transform(X_train)
        X_valid = bow.transform(X_valid)
        X_test = bow.transform(X_test)

        if debug:
            print(bow.get_feature_names())

    return X_train, y_train, X_valid, y_valid, X_test



def main():
    # Load in data
    X_train, y_train, X_valid, y_valid, X_test = load_data(path='../../Data', return_numpy=False)
    # Fit the Bag of Words model for Q1.1
    bow = BagOfWords(vocabulary_size=10)
    vocab = bow.fit(X_train[:100])
    representation = bow.transform(X_train[100:200])
    counts = representation.sum(0)

    for word, count in zip(vocab, counts):
        print(f'{word}: {count}')

    # Load in data
    X_train, y_train, X_valid, y_valid, X_test = load_data(path='../../Data', return_numpy=True, debug=False)

    # Fit the Naive Bayes model for Q1.3
    print('Q1.3')
    nb = NaiveBayes(beta=1)
    nb.fit(X_train, y_train)
    _, y_pred = nb.predict(X_valid)
    print(f'roc auc: {roc_auc_score(y_valid, y_pred)}')
    print(f'f1: {f1_score(y_valid, y_pred)}')
    print(f'acc: {accuracy_score(y_valid, y_pred)}')
    print(f'conf:\n {confusion_matrix(y_valid, y_pred)}')

    print('Q1.4')
    betas = []
    scores = []
    for i in range(1, 10):
        model = NaiveBayes(beta = i)
        NB_priors, NB_likelihoods = model.fit(X_train, y_train)
        NB_posteriors, NB_y_pred = model.predict(X_valid)
        score = roc_auc_score(y_valid, NB_y_pred)

        betas.append(i)
        scores.append(score)
        print(f'Beta:{i}, Score:{score}')

    plt.xlabel('Betas')
    plt.ylabel('ROC AUC scores')
    plt.plot(betas, scores)
    # plt.savefig('ROC_AUC_Scores')

if __name__ == '__main__':
    main()
