from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


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
        self.int_params = {}
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        
        C,H,W = input_dim
        input_dim2 = input_dim[0]*input_dim[1]*input_dim[2]
        F = filter_size
        

        self.int_params['conv_pad'] = int((F - 1) // 2)
        self.int_params['conv_stride'] = 1
        self.int_params['pool_height'] = 2
        self.int_params['pool_width'] = 2
        self.int_params['pool_stride'] = 2

        P = self.int_params['conv_pad'] 
        S = self.int_params['conv_stride'] 

        conv_out_dim = (int((H-F+2*P)/1+1),int((W-F+2*P)/1+1))
        print('conv_out_dim = ',conv_out_dim)
        #N, F, H',W'
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
        self.params['W1'] = weight_scale * np.random.randn(num_filters, C,filter_size,filter_size)
        self.params['b1'] = np.zeros(num_filters)
        w2_dim = int(num_filters*conv_out_dim[0]/2*conv_out_dim[1]/2) #conv relu and pool(size decrease to 1/2*1/2)
        print('w2_dim = ',w2_dim)
        self.params['W2'] = weight_scale * np.random.randn(w2_dim, hidden_dim)
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b3'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
       
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        filter_size = W1.shape[2]
        self.int_params['conv_pad'] = (filter_size - 1) // 2
        self.int_params['conv_stride'] = 1
        self.int_params['pool_height'] = 2
        self.int_params['pool_width'] = 2
        self.int_params['pool_stride'] = 2
        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': self.int_params['conv_stride'], 'pad': self.int_params['conv_pad']}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height':  self.int_params['pool_height'], 'pool_width': self.int_params['pool_width'], 'stride': self.int_params['pool_stride']}
        
        
        scores = None
        X = X.astype(self.dtype)
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################

        crp_out, crp_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        #print('crp_out.shape = ',crp_out.shape)
        crp_out2 = crp_out.reshape(crp_out.shape[0],-1)
        ar_out,ar_cache = affine_relu_forward(crp_out2, W2, b2)
        scores, a_cache = affine_forward(ar_out, W3, b3)
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
        loss,ds = softmax_loss(scores, y)
        reg = self.reg
        
        loss += 0.5 * reg * np.sum(W1 * W1) 
        loss += 0.5 * reg * np.sum(W2 * W2)
        loss += 0.5 * reg * np.sum(W3 * W3)
        #dout = 1
        a_dx, a_dw, a_db = affine_backward(ds, a_cache)
        a_dw += reg * W3
        ar_dx, ar_dw, ar_db = affine_relu_backward(a_dx, ar_cache)
        ar_dw += reg * W2
        ar_dx2 = ar_dx.reshape(crp_out.shape)
        #print('ar_dx shape = ', ar_dx.shape)
        #print('ar_dx2 shape = ', ar_dx2.shape)
        crp_dx, crp_dw, crp_db = conv_relu_pool_backward(ar_dx2, crp_cache)
        crp_dw += reg *W1
        grads['W1'] = crp_dw
        grads['b1'] = crp_db
        grads['W2'] = ar_dw
        grads['b2'] = ar_db
        grads['W3'] = a_dw
        grads['b3'] = a_db
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
