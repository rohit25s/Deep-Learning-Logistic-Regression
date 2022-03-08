#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 12:00:48 2019

@author: 
"""

import numpy as np
import sys

"""This script implements a two-class logistic regression model.
"""

class logistic_regression_multiclass(object):
	
    def __init__(self, learning_rate, max_iter, k):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.k = k 
        
    def fit_BGD(self, X, labels, batch_size):
        """Train perceptron model on data (X,y) with BGD.

        Args:
            X: An array of shape [n_samples, n_features].
            labels: An array of shape [n_samples,].  Only contains 0,..,k-1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.

        Hint: the labels should be converted to one-hot vectors, for example: 1----> [0,1,0]; 2---->[0,0,1].
        """

		    ### YOUR CODE HERE
        n_samples, n_features = X.shape
        w = np.random.rand(n_features, self.k)
        self.W = w
        y_one_hot = np.zeros((n_samples, self.k))
        for sample in range(n_samples):
          y_one_hot[sample][int(labels[sample])] = 1

        for iter in range(self.max_iter):
          sample = 0
          while(sample < n_samples):
            gradient = 0
            for i in range(sample, sample + batch_size):
              gradient += self._gradient(X[i], y_one_hot[i])
            gradient /= batch_size
            w = self.get_params() - self.learning_rate * gradient
            self.W = w  
            sample += batch_size

          print("Iteration:" + str(iter) + ", score:" +str(self.score(X,labels)))

        
	    	### END YOUR CODE
    

    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: One_hot vector. 

        Returns:
            _g: An array of shape [n_features,]. The gradient of
                cross-entropy with respect to self.W.
        """
		    ### YOUR CODE HERE

        sm = self.softmax(_x)
        gradient = np.matmul(np.array([sm - _y]).T, np.array([_x]))
        return gradient
		    ### END YOUR CODE
    
    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        ### You must implement softmax by youself, otherwise you will not get credits for this part.

		    ### YOUR CODE HERE
        soft = np.zeros(self.k)
        weight = self.get_params()
        for i in range(self.k):
          top = np.exp(np.matmul(weight[i].T,x))
          soft[i] = top
        soft /= np.sum(soft)
          
        return soft
		    ### END YOUR CODE
    
    def get_params(self):
        """Get parameters for this perceptron model.

        Returns:
            W: An array of shape [n_features,].
        """
        if self.W is None:
            print("Run fit first!")
            sys.exit(-1)
        return self.W


    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 0,..,k-1.
        """
		    ### YOUR CODE HERE
        n_samples, n_features = X.shape
        w=self.get_params()
        predictions = np.zeros(n_samples)
        for i in range(n_samples):
            predictions[i] = np.argmax(self.softmax(X[i]), axis=0)
        return predictions
		    ### END YOUR CODE


    def score(self, X, labels):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            labels: An array of shape [n_samples,]. Only contains 0,..,k-1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. labels.
        """
		    ### YOUR CODE HERE
        n_samples, n_features = X.shape
        predictions=self.predict(X)
        count=0
        for i,prediction in enumerate(predictions):
          if prediction==labels[i]:
            count=count+1
        return count/n_samples
		    ### END YOUR CODE
