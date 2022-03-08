#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import sys
import random

"""This script implements a two-class logistic regression model.
"""

class logistic_regression(object):
	
    def __init__(self, learning_rate, max_iter):
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def fit_GD(self, X, y):
        """Train perceptron model on data (X,y) with GD.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
        n_samples, n_features = X.shape

		    ### YOUR CODE HERE
        w = np.random.rand(n_features)
        self.assign_weights(w)
        for iter in range(self.max_iter):
          gradient = 0
          for sample in range(n_samples):
            gradient += self._gradient(X[sample], y[sample])

          gradient /= n_samples
          w = self.get_params() - self.learning_rate * gradient
          self.assign_weights(w) 
          print("Iteration:" + str(iter) + ", score:" +str(self.score(X,y)))  
        
		    ### END YOUR CODE
        return self

    def fit_BGD(self, X, y, batch_size):
        """Train perceptron model on data (X,y) with BGD.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.
        """
		    ### YOUR CODE HERE
        n_samples, n_features = X.shape
        w = np.random.rand(n_features)
        self.assign_weights(w)

        for iter in range(self.max_iter):
          sample = 0
          while(sample < n_samples):
            gradient = 0
            for i in range(sample, sample + batch_size):
              gradient += self._gradient(X[i], y[i])
            gradient /= batch_size
            w = self.get_params() - self.learning_rate * gradient
            self.assign_weights(w)  
            sample += batch_size

          print("Iteration:" + str(iter) + ", score:" +str(self.score(X,y)))     
		    ### END YOUR CODE
        return self    

    def fit_SGD(self, X, y):
        """Train perceptron model on data (X,y) with SGD.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
		    ### YOUR CODE HERE
        n_samples, n_features = X.shape
        w = np.random.rand(n_features)
        self.assign_weights(w)

        for iter in range(self.max_iter):
          random_i = random.randint(0, n_samples-1)
          w = self.get_params() - self.learning_rate * self._gradient(X[random_i], y[random_i])
          self.assign_weights(w)
          print("Iteration:" + str(iter) + ", score:" +str(self.score(X,y)))
		    ### END YOUR CODE
        return self   

    
    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: An integer. 1 or -1.

        Returns:
            _g: An array of shape [n_features,]. The gradient of
                cross-entropy with respect to self.W.
        """
        ### YOUR CODE HERE
        temp = np.exp((-1)*_y*np.matmul(self.get_params().T , _x))
        return (-1)*_x*_y*temp/(1+temp)
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

    def predict_proba(self, X):
        """Predict class probabilities for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds_proba: An array of shape [n_samples, 2].
                Only contains floats between [0,1].
        """
		    ### YOUR CODE HERE
        n_samples, n_features = X.shape
        ans = np.zeros((n_samples,2))
        for i in range(n_samples):
          z = np.sum(np.matmul(self.get_params().T, X[i]))
          sigmoid = 1/(1+ np.exp(-z))
          ans[i] = [sigmoid, 1-sigmoid]
        return ans    
		    ### END YOUR CODE


    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].
        
        Returns:
            preds: An array of shape [n_samples,]. Only contains 1 or -1.
        """
		    ### YOUR CODE HERE
        n_samples, n_features = X.shape
        predictions=np.zeros(n_samples)
        for i,_x in enumerate(X):
          sigmoid = 1/(1+np.exp(np.sum(-np.matmul(self.get_params().T,_x))))
          if sigmoid >= 0.5:
            predictions[i] = 1
          else:
            predictions[i] = -1

        return predictions
		    ### END YOUR CODE

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.


        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. y.
        """
		    ### YOUR CODE HERE
        yhat = self.predict(X)
        mean = 0
        n_samples, n_features = X.shape
        for i in range(n_samples):
          if yhat[i] == y[i]:
            mean += 1
        return mean/n_samples   
		    ### END YOUR CODE
    
    def assign_weights(self, weights):
        self.W = weights
        return self

