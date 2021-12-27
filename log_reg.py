from __future__ import print_function

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

def getauc(y_true, probs):
    """
    Use sklearn roc_auc_score to get the auc given predicted probabilities and the true labels
    
    Args:
        - y_true: The true labels for the data
        - probs: predicted probabilities
    """
    #TODO: return auc using sklearn roc_auc_score
    roc_auc_score(y_true, probs)

def conf_mat(y_true, y_pred):
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

class LogisticRegression(object):
    def __init__(self, input_size, reg=0.0, std=1e-4):
        """
        Initializing the weight vector
        
        Input:
        - input_size: the number of features in dataset, for bag of words it would be number of unique words in your training data
        - reg: the l2 regularization weight
        - std: the variance of initialized weights
        """
        self.W = (std * np.random.randn(input_size)).reshape(-1, 1) # column vector
        self.reg = reg
        
    def sigmoid(self,Z):
        """
        Compute sigmoid of x
        
        Input:
        - x: Input data of shape (N,)
        
        Returns:
        - sigmoid of each value in x with the same shape as x (N,)
        """
        #TODO: write sigmoid function
        return 1 / (1 + np.exp(Z))

    def loss(self, X, y):
        """
        Compute the loss and gradients for your logistic regression.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: A numpy array f shape (N,) giving training labels.

        Returns:
        - loss: Loss (data loss and regularization loss) for the entire training samples
        - dLdW: gradient of loss with respect to W
        """
        N, D = X.shape
        reg = self.reg

        y = y.reshape(-1, 1)
        #TODO: Compute scores
        Z = X.dot(self.W)
        A = 1 / (1 + np.exp(Z))
        #TODO: Compute the loss
        loss = -(y * np.log(A) + (1 - y) * np.log(1 - A)).sum() / N
        loss += self.reg * (self.W**2).sum()

        dW = (X.T).dot(A - y) / N + (reg * 2 * self.W)
        #TODO: Compute gradients
        # Calculate dLdW meaning the gradient of loss function according to W 
        # you can use chain rule here with calculating each step to make your job easier
        return loss, dW

    def gradDescent(self,X, y, learning_rate, num_epochs, verbose = False):
        """
        We will use Gradient Descent for updating our weights, here we used gradient descent instead of stochastic gradient descent for easier implementation
        so you will use the whole training set for each update of weights instead of using batches of data.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - num_epochs: integer giving the number of epochs to train

        Returns:
        - loss_hist: numpy array of size (num_epochs,) giving the loss value over epochs
        """
        N, D = X.shape
        loss_hist = np.zeros(num_epochs)
        for i in range(num_epochs):
            #TODO: implement steps of gradient descent
            loss, dW = self.loss(X, y)
            self.W += learning_rate * dW
            # printing loss, you can also print accuracy here after few iterations to see how your model is doing
            loss_hist[i] = loss

            if verbose and (i+1) % 100 == 0:
                print("Epoch : ", i+1, " loss : ", loss)
          
        return loss_hist

    def predict(self, X):
        """
        Use the trained weights to predict labels for data given as X

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
            - probs: A numpy array of shape (N,) giving predicted probabilities for each of the elements of X.
            - y_pred: A numpy array of shape (N,) giving predicted labels for each of the elements of X. You can get this by putting a threshold of 0.5 on probs
        """
        #TODO: get the scores (probabilities) for input data X and calculate the labels (0 and 1 for each input data) and return them
        
        Z = X.dot(self.W)
        probs = self.sigmoid(Z)
        y_pred = probs > 0.5
        return probs, y_pred

def score(y_true, y_pred):
    return f1_score(y_true, y_pred)

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
    X_train, y_train, X_valid, y_valid, X_test = load_data(path = '../../Data', return_numpy = True)
    # e.g.  LR = LogisticRegression(X_train.shape[1], reg=0)
    #       LR.gradDescent(X_train, y_train, 1.0, 100)
    #       LR.predict(X_valid)

    # Write a for loop for each hyper parameter here each time initialize logistic regression train it on the train data and get auc on validation data and get confusion matrix using the best hyper params 

    for i in range(3):
        print(f'Test {i}: ')
        y_pred = None
        reg = np.random.uniform(0, 0.2)
        lr = np.random.uniform(1e-4, 1e-2)
        iterations = int(np.random.uniform(10, 500))

        print(f'regularization: {reg}')
        print(f'learning rate: {lr}')
        print(f'iterations: {iterations}')
        print('\n')
        model = LogisticRegression(X_train.shape[1], reg = reg)
        loss_hist = model.gradDescent(X_train, y_train, lr, iterations, verbose = False)
        probs, y_pred = model.predict(X_valid)

        if y_pred is None:
            print('-' * 40)
            continue

        print(f'f1 scores: {score(y_valid, y_pred)}')
        print(f'roc auc score: {roc_auc_score(y_valid, y_pred)}')
        print('-' * 40)

    print('Q3.1')
    model = LogisticRegression(X_train.shape[1], reg = 1e-3)
    loss_hist = model.gradDescent(X_train, y_train, 1, 200, verbose = True)

    _, y_pred = model.predict(X_train)
    print(f'train roc auc: {roc_auc_score(y_train, y_pred)}')
    _, y_pred = model.predict(X_valid)
    print(f'valid roc auc: {roc_auc_score(y_valid, y_pred)}')

    print('Q3.3')
    print(conf_mat(y_valid, y_pred))

    test_probs, test_predictions = model.predict(X_test)
    # pd.DataFrame(test_predictions).to_csv('../Predictions/best.csv', index = True, header = False)
if __name__ == '__main__':
    main()



