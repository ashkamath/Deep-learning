import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!     
  num_classes = W.shape[1]
  num_train = X.shape[0] 
  for i in xrange(num_train):
    scores = X[i].dot(W)
    #for numerical stability
    scores-= np.max(scores)
 #   correct_class_score = scores[y[i]]
    #print scores.shape
    v=0.0
    #print X[i].shape
    #print dW[i].shape
    for j in xrange(num_classes):
        v=v+np.exp(scores[j])
        
    for j in range(num_classes):
      common = np.exp(scores[j])/v
      #for when j==y[i] multiply by X[i,:]
      dW[:, j] += (common-(j == y[i])) * X[i, :]    
      
    loss+= -1*(scores[y[i]])+np.log(v)
      
  dW=dW/num_train
  dW+=reg*W
  loss=loss/num_train
  loss += 0.5 * reg * np.sum(W * W)                                                     #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!    
  num_train=X.shape[0]
  loss = 0.0
  dW = np.zeros_like(W)
  scores=X.dot(W)
  #print scores.shape
  #for numerical stablity, 
  scores-=np.max(scores)
  
  e=np.exp(scores)
  
  denominator=np.sum(e,axis=1)
  
  #print denominator.shape
  numerator=e[range(num_train),y]
  l=(-1)*np.log(numerator/denominator)
  #print l.shape
  loss=np.sum(l)
  loss=loss/num_train
  loss += 0.5 * reg * np.sum(W * W)

  
  denominator=np.reshape(denominator,(-1,1))  
  common = e/denominator
  idx = np.zeros(common.shape)
  idx[range(num_train),y] = 1
  dW = np.dot((common-idx).T, X)
  dW=dW.T
  dW /= num_train
  dW+=reg*W                                                       #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

