import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  

  #############################################################################
  # TODO:  
      #updating the column of W that corresponds to correct class
        dW[:,y[i]] = dW[:,y[i]] -  X[i,:].T
        #updating the rest
        dW[:,j] = dW[:,j] +  X[i,:].T                                                                   #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.    
  dW /= num_train

  dW+=reg*W                                   #
  #############################################################################
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                     
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  scores= np.dot(X,W)
#  print scores.shape
#  print y.shape
  corr=scores[range(0,num_train),y].reshape(num_train,1)
  corr =corr 
  margins=np.maximum(0, scores-corr+1)
  margins[np.arange(num_train), y] = 0
  loss = np.sum(margins) / num_train
  loss += 0.5 * reg * np.sum(W * W)                                      #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO: 
                                                                      #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.   
  m = margins
  m[m > 0] = 1
  #print m.shape

  c_sum = np.sum(m, axis=1)
  #print c_sum.shape
  #y=y.reshape(500,1)
  #print y.shape
 
  a = range(num_train)
  
 #other things to try 
 #  m[a,y] = -c_sum[range(num_train)]
 #  m[a,:]= b[range(num_train)]
  
  #dot product takes into account all the cases where y[i] not equal to j and the summation of the cases where j=y[i] is
  #done in the following statement
  m[a,y] =-c_sum[range(num_train)]
  
  #now multiplying the corrected margin matrix with the X gives us the gradient with the correct sizes.
  dW = np.dot(X.T, m)
  dW /= num_train
  
  #reg
  dW+=reg*W
                                                                    #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
