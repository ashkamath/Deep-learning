import numpy as np

from asgn2.layers import *
from asgn2.fast_layers import *
from asgn2.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:

  conv - relu - 2x2 max pool - affine - relu - affine - softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    self.params['W1']=weight_scale* np.random.randn(num_filters,input_dim[0],filter_size,filter_size)
    self.params['b1']=np.zeros(num_filters)    
    self.params['W2']=weight_scale* np.random.randn((num_filters*input_dim[1]*input_dim[2])/4,hidden_dim)
    self.params['b2']=np.zeros(hidden_dim)
    self.params['W3']=weight_scale* np.random.randn(hidden_dim,num_classes)
    self.params['b3']=np.zeros(num_classes)
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
#    out, cache=conv_forward_naive(X, W1, b1, conv_param)
#    out,relu1_cache=relu_forward(out)
#    out, max_pool_cache=max_pool_forward_naive(out, pool_param)
#    out, affine1_cache=affine_forward(out, W2, b2)
#    out,relu2_cache=relu_forward(out)
#    out, affine2_cache=affine_forward(out, W3, b3)
#    scores=out
    
    out, cache=conv_forward_im2col(X, W1, b1, conv_param)
    out,relu1_cache=relu_forward(out)
    out, max_pool_cache=max_pool_forward_reshape(out, pool_param)
    out, affine1_cache=affine_forward(out, W2, b2)
    out,relu2_cache=relu_forward(out)
    out, affine2_cache=affine_forward(out, W3, b3)
    scores=out
    
    
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    sloss,sgrads=softmax_loss(scores,y)
    loss=sloss+0.5*self.reg*np.sum(self.params['W1']*self.params['W1'])+0.5*self.reg*np.sum(self.params['W2']*self.params['W2'])+0.5*self.reg*np.sum(self.params['W3']*self.params['W3'])
    
#    dX3,grads['W3'],grads['b3']=affine_backward(sgrads, affine2_cache)
#    dX3=relu_backward(dX3, relu2_cache)
#    dX2,grads['W2'],grads['b2']=affine_backward(dX3, affine1_cache)
#    dX2=max_pool_backward_naive(dX2, max_pool_cache)
#    dX2=relu_backward(dX2, relu1_cache)
#    dX,grads['W1'],grads['b1']=conv_backward_naive(dX2, cache)
    
    dX3,grads['W3'],grads['b3']=affine_backward(sgrads, affine2_cache)
    dX3=relu_backward(dX3, relu2_cache)
    dX2,grads['W2'],grads['b2']=affine_backward(dX3, affine1_cache)
    dX2=max_pool_backward_reshape(dX2, max_pool_cache)
    dX2=relu_backward(dX2, relu1_cache)
    dX,grads['W1'],grads['b1']=conv_backward_im2col(dX2, cache)
    
    grads['W1']+=self.params['W1']*self.reg
    grads['W2']+=self.params['W2']*self.reg
    grads['W3']+=self.params['W3']*self.reg
    
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


pass


class ThreeLayerConvNetWithNorm(object):
  """
  A three-layer convolutional network with the following architecture:

  conv - relu - 2x2 max pool - affine - relu - affine - softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, use_batchnorm=False,num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.use_batchnorm = use_batchnorm
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    self.params['W1']=weight_scale* np.random.randn(num_filters,input_dim[0],filter_size,filter_size)
    self.params['b1']=np.zeros(num_filters)    
    self.params['W2']=weight_scale* np.random.randn((num_filters*input_dim[1]*input_dim[2])/4,hidden_dim)
    self.params['b2']=np.zeros(hidden_dim)
    self.params['W3']=weight_scale* np.random.randn(hidden_dim,num_classes)
    self.params['b3']=np.zeros(num_classes)
    
    if self.use_batchnorm:    
        self.params['gamma1']=np.ones(num_filters)
        self.params['beta1']=np.zeros(num_filters)
        self.params['gamma2']=np.ones(hidden_dim)
        self.params['beta2']=np.zeros(hidden_dim)
        
        
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(2)]

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
#    out, cache=conv_forward_naive(X, W1, b1, conv_param)
#    out,relu1_cache=relu_forward(out)
#    out, max_pool_cache=max_pool_forward_naive(out, pool_param)
#    out, affine1_cache=affine_forward(out, W2, b2)
#    out,relu2_cache=relu_forward(out)
#    out, affine2_cache=affine_forward(out, W3, b3)
#    scores=out
    
    
    
    out, cache=conv_forward_im2col(X, W1, b1, conv_param)
    if self.use_batchnorm:
        out, bn1_cache=spatial_batchnorm_forward(out,self.params['gamma1'],self.params['beta1'],self.bn_params[0])
    out,relu1_cache=relu_forward(out)
    out, max_pool_cache=max_pool_forward_reshape(out, pool_param)
    out, affine1_cache=affine_forward(out, W2, b2)
    if self.use_batchnorm:
        out, bn2_cache=batchnorm_forward(out, self.params['gamma2'],self.params['beta2'],self.bn_params[1])
    out,relu2_cache=relu_forward(out)
    out, affine2_cache=affine_forward(out, W3, b3)
    scores=out
    
    
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    sloss,sgrads=softmax_loss(scores,y)
    loss=sloss+0.5*self.reg*np.sum(self.params['W1']*self.params['W1'])+0.5*self.reg*np.sum(self.params['W2']*self.params['W2'])+0.5*self.reg*np.sum(self.params['W3']*self.params['W3'])
    
#    dX3,grads['W3'],grads['b3']=affine_backward(sgrads, affine2_cache)
#    dX3=relu_backward(dX3, relu2_cache)
#    dX2,grads['W2'],grads['b2']=affine_backward(dX3, affine1_cache)
#    dX2=max_pool_backward_naive(dX2, max_pool_cache)
#    dX2=relu_backward(dX2, relu1_cache)
#    dX,grads['W1'],grads['b1']=conv_backward_naive(dX2, cache)
    
    dX3,grads['W3'],grads['b3']=affine_backward(sgrads, affine2_cache)
    dX3=relu_backward(dX3, relu2_cache)
    if self.use_batchnorm:
        dX3,grads['gamma2'],grads['beta2']=batchnorm_backward(dX3, bn2_cache)
    dX2,grads['W2'],grads['b2']=affine_backward(dX3, affine1_cache)
    dX2=max_pool_backward_reshape(dX2, max_pool_cache)
    dX2=relu_backward(dX2, relu1_cache)
    if self.use_batchnorm:
        dX2,grads['gamma1'],grads['beta1']=spatial_batchnorm_backward(dX2, bn1_cache)        
    dX,grads['W1'],grads['b1']=conv_backward_im2col(dX2, cache)
    
    grads['W1']+=self.params['W1']*self.reg
    grads['W2']+=self.params['W2']*self.reg
    grads['W3']+=self.params['W3']*self.reg
    
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


pass


class DeeperConvNet1(object):
  """
  A three-layer convolutional network with the following architecture:

  conv - relu - pool - conv - relu - pool  - affine - relu - affine - softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=3,
               hidden_dim=100, use_batchnorm=False,num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.use_batchnorm = use_batchnorm
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    self.params['W1']=weight_scale* np.random.randn(num_filters,input_dim[0],filter_size,filter_size)
    self.params['b1']=np.zeros(num_filters)   
    self.params['W1b']=weight_scale*np.random.randn(num_filters,self.params['W1'].shape[0],filter_size,filter_size)
    self.params['b1b']=np.zeros(num_filters)
    
    
    self.params['W2']=weight_scale* np.random.randn((num_filters*input_dim[1]*input_dim[2])/16,hidden_dim)
    self.params['b2']=np.zeros(hidden_dim)
    self.params['W3']=weight_scale* np.random.randn(hidden_dim,num_classes)
    self.params['b3']=np.zeros(num_classes)
    
    
    
    if self.use_batchnorm:    
        self.params['gamma1']=np.ones(num_filters)
        self.params['beta1']=np.zeros(num_filters)
        self.params['gamma1b']=np.ones(num_filters)
        self.params['beta1b']=np.zeros(num_filters)
        self.params['gamma2']=np.ones(hidden_dim)
        self.params['beta2']=np.zeros(hidden_dim)
        
        
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(2)]

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W1b,b1b= self.params['W1b'], self.params['b1b']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
#    out, cache=conv_forward_naive(X, W1, b1, conv_param)
#    out,relu1_cache=relu_forward(out)
#    out, max_pool_cache=max_pool_forward_naive(out, pool_param)
#    out, affine1_cache=affine_forward(out, W2, b2)
#    out,relu2_cache=relu_forward(out)
#    out, affine2_cache=affine_forward(out, W3, b3)
#    scores=out
    
    #print X.shape
    
    out, cache=conv_forward_im2col(X, W1, b1, conv_param)
   
    if self.use_batchnorm:
        out, bn1_cache=spatial_batchnorm_forward(out,self.params['gamma1'],self.params['beta1'],self.bn_params[0])
    
    out,relu1_cache=relu_forward(out)
       
    #print out.shape 
    
    out, max_pool_cache=max_pool_forward_reshape(out, pool_param)
    
    #print out.shape
    
    out, cache2=conv_forward_im2col(out, W1b, b1b, conv_param) 
    
    #print out.shape
    
    if self.use_batchnorm:
        out, bn1b_cache=spatial_batchnorm_forward(out,self.params['gamma1b'],self.params['beta1b'],self.bn_params[0])
    out,relu1b_cache=relu_forward(out)
    
    out, max_pool_cache_b=max_pool_forward_reshape(out, pool_param)
    
    #print out.shape
    #print W2.shape
    out, affine1_cache=affine_forward(out, W2, b2)
    if self.use_batchnorm:
        out, bn2_cache=batchnorm_forward(out, self.params['gamma2'],self.params['beta2'],self.bn_params[1])
    out,relu2_cache=relu_forward(out)
    out, affine2_cache=affine_forward(out, W3, b3)
    scores=out
    
    
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    sloss,sgrads=softmax_loss(scores,y)
    loss=sloss+0.5*self.reg*np.sum(self.params['W1']*self.params['W1'])+0.5*self.reg*np.sum(self.params['W2']*self.params['W2'])+0.5*self.reg*np.sum(self.params['W3']*self.params['W3']+0.5*self.reg*np.sum(self.params['W1b']*self.params['W1b']))
    
#    dX3,grads['W3'],grads['b3']=affine_backward(sgrads, affine2_cache)
#    dX3=relu_backward(dX3, relu2_cache)
#    dX2,grads['W2'],grads['b2']=affine_backward(dX3, affine1_cache)
#    dX2=max_pool_backward_naive(dX2, max_pool_cache)
#    dX2=relu_backward(dX2, relu1_cache)
#    dX,grads['W1'],grads['b1']=conv_backward_naive(dX2, cache)
    
    dX3,grads['W3'],grads['b3']=affine_backward(sgrads, affine2_cache)
    dX3=relu_backward(dX3, relu2_cache)
    if self.use_batchnorm:
        dX3,grads['gamma2'],grads['beta2']=batchnorm_backward(dX3, bn2_cache)
    dX2,grads['W2'],grads['b2']=affine_backward(dX3, affine1_cache)
   
    dX2=max_pool_backward_reshape(dX2, max_pool_cache_b)
    dX2=relu_backward(dX2, relu1b_cache)
    if self.use_batchnorm:
        dX2,grads['gamma1b'],grads['beta1b']=spatial_batchnorm_backward(dX2, bn1b_cache)        
    dX,grads['W1b'],grads['b1b']=conv_backward_im2col(dX2, cache2)
    
    dX=max_pool_backward_reshape(dX, max_pool_cache)
    dX=relu_backward(dX, relu1_cache)
    if self.use_batchnorm:
        dX,grads['gamma1'],grads['beta1']=spatial_batchnorm_backward(dX, bn1_cache)        
    dX,grads['W1'],grads['b1']=conv_backward_im2col(dX, cache)
    
    
    
    grads['W1']+=self.params['W1']*self.reg
    grads['W1b']+=self.params['W1b']*self.reg
    grads['W2']+=self.params['W2']*self.reg
    grads['W3']+=self.params['W3']*self.reg
    
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


pass
    
class DeeperConvNet2(object):
  """
  A three-layer convolutional network with the following architecture:

  conv - relu - conv- relu - 2x2 max pool - affine - relu - affine - softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=3,
               hidden_dim=100, use_batchnorm=False,num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.use_batchnorm = use_batchnorm
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    self.params['W1']=weight_scale* np.random.randn(num_filters,input_dim[0],filter_size,filter_size)
    self.params['b1']=np.zeros(num_filters)   
    self.params['W1b']=weight_scale*np.random.randn(num_filters,self.params['W1'].shape[0],filter_size,filter_size)
    self.params['b1b']=np.zeros(input_dim[0]/2)
    
    
    self.params['W2']=weight_scale* np.random.randn((num_filters*input_dim[1]*input_dim[2])/16,hidden_dim)
    self.params['b2']=np.zeros(hidden_dim)
    self.params['W3']=weight_scale* np.random.randn(hidden_dim,num_classes)
    self.params['b3']=np.zeros(num_classes)
    
    
    
    if self.use_batchnorm:    
        self.params['gamma1']=np.ones(num_filters)
        self.params['beta1']=np.zeros(num_filters)
        self.params['gamma1b']=np.ones(num_filters)
        self.params['beta1b']=np.zeros(num_filters)
        self.params['gamma2']=np.ones(hidden_dim)
        self.params['beta2']=np.zeros(hidden_dim)
        
        
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(2)]

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W1b,b1b= self.params['W1b'], self.params['b1b']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
#    out, cache=conv_forward_naive(X, W1, b1, conv_param)
#    out,relu1_cache=relu_forward(out)
#    out, max_pool_cache=max_pool_forward_naive(out, pool_param)
#    out, affine1_cache=affine_forward(out, W2, b2)
#    out,relu2_cache=relu_forward(out)
#    out, affine2_cache=affine_forward(out, W3, b3)
#    scores=out
    
    #print X.shape
    
    out, cache=conv_forward_im2col(X, W1, b1, conv_param)
   
    if self.use_batchnorm:
        out, bn1_cache=spatial_batchnorm_forward(out,self.params['gamma1'],self.params['beta1'],self.bn_params[0])
    
    out,relu1_cache=relu_forward(out)
    
    print out.shape
    
    out, cache2=conv_forward_im2col(out, W1b, b1b, conv_param)
    
    
    if self.use_batchnorm:
        out, bn1b_cache=spatial_batchnorm_forward(out,self.params['gamma1b'],self.params['beta1b'],self.bn_params[0])
   
    out,relu1b_cache=relu_forward(out)
       
    #print out.shape 
    
    out, max_pool_cache=max_pool_forward_reshape(out, pool_param)
    
    
    #print out.shape
    #print W2.shape
    out, affine1_cache=affine_forward(out, W2, b2)
    if self.use_batchnorm:
        out, bn2_cache=batchnorm_forward(out, self.params['gamma2'],self.params['beta2'],self.bn_params[1])
    out,relu2_cache=relu_forward(out)
    out, affine2_cache=affine_forward(out, W3, b3)
    scores=out
    
    
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    sloss,sgrads=softmax_loss(scores,y)
    loss=sloss+0.5*self.reg*np.sum(self.params['W1']*self.params['W1'])+0.5*self.reg*np.sum(self.params['W2']*self.params['W2'])+0.5*self.reg*np.sum(self.params['W3']*self.params['W3']+0.5*self.reg*np.sum(self.params['W1b']*self.params['W1b']))
    
#    dX3,grads['W3'],grads['b3']=affine_backward(sgrads, affine2_cache)
#    dX3=relu_backward(dX3, relu2_cache)
#    dX2,grads['W2'],grads['b2']=affine_backward(dX3, affine1_cache)
#    dX2=max_pool_backward_naive(dX2, max_pool_cache)
#    dX2=relu_backward(dX2, relu1_cache)
#    dX,grads['W1'],grads['b1']=conv_backward_naive(dX2, cache)
    
    dX3,grads['W3'],grads['b3']=affine_backward(sgrads, affine2_cache)
    dX3=relu_backward(dX3, relu2_cache)
    if self.use_batchnorm:
        dX3,grads['gamma2'],grads['beta2']=batchnorm_backward(dX3, bn2_cache)
    dX2,grads['W2'],grads['b2']=affine_backward(dX3, affine1_cache)
   
    dX2=max_pool_backward_reshape(dX2, max_pool_cache)
    
    dX2=relu_backward(dX2, relu1b_cache)
    if self.use_batchnorm:
        dX2,grads['gamma1b'],grads['beta1b']=spatial_batchnorm_backward(dX2, bn1b_cache)        
    dX,grads['W1b'],grads['b1b']=conv_backward_im2col(dX2, cache2)
    
    
    dX=relu_backward(dX, relu1_cache)
    if self.use_batchnorm:
        dX,grads['gamma1'],grads['beta1']=spatial_batchnorm_backward(dX, bn1_cache)        
    dX,grads['W1'],grads['b1']=conv_backward_im2col(dX, cache)
    
    
    
    grads['W1']+=self.params['W1']*self.reg
    grads['W1b']+=self.params['W1b']*self.reg
    grads['W2']+=self.params['W2']*self.reg
    grads['W3']+=self.params['W3']*self.reg
    
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


pass
    
class DeeperConvNet1LeakyRelu(object):
  """
  A three-layer convolutional network with the following architecture:

  conv - relu - pool - conv - relu - pool  - affine - relu - affine - softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=3,
               hidden_dim=100, use_batchnorm=False,num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.use_batchnorm = use_batchnorm
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    self.params['W1']=weight_scale* np.random.randn(num_filters,input_dim[0],filter_size,filter_size)
    self.params['b1']=np.zeros(num_filters)   
    self.params['W1b']=weight_scale*np.random.randn(num_filters,self.params['W1'].shape[0],filter_size,filter_size)
    self.params['b1b']=np.zeros(num_filters)
    
    
    self.params['W2']=weight_scale* np.random.randn((num_filters*input_dim[1]*input_dim[2])/16,hidden_dim)
    self.params['b2']=np.zeros(hidden_dim)
    self.params['W3']=weight_scale* np.random.randn(hidden_dim,num_classes)
    self.params['b3']=np.zeros(num_classes)
    
    
    
    if self.use_batchnorm:    
        self.params['gamma1']=np.ones(num_filters)
        self.params['beta1']=np.zeros(num_filters)
        self.params['gamma1b']=np.ones(num_filters)
        self.params['beta1b']=np.zeros(num_filters)
        self.params['gamma2']=np.ones(hidden_dim)
        self.params['beta2']=np.zeros(hidden_dim)
        
        
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(2)]

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W1b,b1b= self.params['W1b'], self.params['b1b']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
#    out, cache=conv_forward_naive(X, W1, b1, conv_param)
#    out,relu1_cache=relu_forward(out)
#    out, max_pool_cache=max_pool_forward_naive(out, pool_param)
#    out, affine1_cache=affine_forward(out, W2, b2)
#    out,relu2_cache=relu_forward(out)
#    out, affine2_cache=affine_forward(out, W3, b3)
#    scores=out
    
    #print X.shape
    
    out, cache=conv_forward_im2col(X, W1, b1, conv_param)
   
    if self.use_batchnorm:
        out, bn1_cache=spatial_batchnorm_forward(out,self.params['gamma1'],self.params['beta1'],self.bn_params[0])
    
    out,relu1_cache=leaky_relu_forward(out)
       
    #print out.shape 
    
    out, max_pool_cache=max_pool_forward_reshape(out, pool_param)
    
    #print out.shape
    
    out, cache2=conv_forward_im2col(out, W1b, b1b, conv_param) 
    
    #print out.shape
    
    if self.use_batchnorm:
        out, bn1b_cache=spatial_batchnorm_forward(out,self.params['gamma1b'],self.params['beta1b'],self.bn_params[0])
    out,relu1b_cache=relu_forward(out)
    
    out, max_pool_cache_b=max_pool_forward_reshape(out, pool_param)
    
    #print out.shape
    #print W2.shape
    out, affine1_cache=affine_forward(out, W2, b2)
    if self.use_batchnorm:
        out, bn2_cache=batchnorm_forward(out, self.params['gamma2'],self.params['beta2'],self.bn_params[1])
    out,relu2_cache=relu_forward(out)
    out, affine2_cache=affine_forward(out, W3, b3)
    scores=out
    
    
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    sloss,sgrads=softmax_loss(scores,y)
    loss=sloss+0.5*self.reg*np.sum(self.params['W1']*self.params['W1'])+0.5*self.reg*np.sum(self.params['W2']*self.params['W2'])+0.5*self.reg*np.sum(self.params['W3']*self.params['W3']+0.5*self.reg*np.sum(self.params['W1b']*self.params['W1b']))
    
#    dX3,grads['W3'],grads['b3']=affine_backward(sgrads, affine2_cache)
#    dX3=relu_backward(dX3, relu2_cache)
#    dX2,grads['W2'],grads['b2']=affine_backward(dX3, affine1_cache)
#    dX2=max_pool_backward_naive(dX2, max_pool_cache)
#    dX2=relu_backward(dX2, relu1_cache)
#    dX,grads['W1'],grads['b1']=conv_backward_naive(dX2, cache)
    
    dX3,grads['W3'],grads['b3']=affine_backward(sgrads, affine2_cache)
    dX3=relu_backward(dX3, relu2_cache)
    if self.use_batchnorm:
        dX3,grads['gamma2'],grads['beta2']=batchnorm_backward(dX3, bn2_cache)
    dX2,grads['W2'],grads['b2']=affine_backward(dX3, affine1_cache)
   
    dX2=max_pool_backward_reshape(dX2, max_pool_cache_b)
    dX2=relu_backward(dX2, relu1b_cache)
    if self.use_batchnorm:
        dX2,grads['gamma1b'],grads['beta1b']=spatial_batchnorm_backward(dX2, bn1b_cache)        
    dX,grads['W1b'],grads['b1b']=conv_backward_im2col(dX2, cache2)
    
    dX=max_pool_backward_reshape(dX, max_pool_cache)
    dX=relu_backward(dX, relu1_cache)
    if self.use_batchnorm:
        dX,grads['gamma1'],grads['beta1']=spatial_batchnorm_backward(dX, bn1_cache)        
    dX,grads['W1'],grads['b1']=conv_backward_im2col(dX, cache)
    
    
    
    grads['W1']+=self.params['W1']*self.reg
    grads['W1b']+=self.params['W1b']*self.reg
    grads['W2']+=self.params['W2']*self.reg
    grads['W3']+=self.params['W3']*self.reg
    
    pass
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


pass