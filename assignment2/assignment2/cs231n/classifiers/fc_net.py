from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.
        A two-layer fully-connected neural network. The net has an input dimension of
        N, a hidden layer dimension of H, and performs classification over C classes.
        We train the network with a softmax loss function and L2 regularization on the
        weight matrices. The network uses a ReLU nonlinearity after the first fully
        connected layer
        
        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        
            W1: First layer weights; has shape (D, H)
            b1: First layer biases; has shape (H,)
            W2: Second layer weights; has shape (H, C)
            b2: Second layer biases; has shape (C,)

            Inputs:
            - input_dim: The dimension D of the input data.
            - hidden_dim: The number of neurons H in the hidden layer.
            - output_size: The number of classes C.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
         
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # Unpack variables from the params dictionary
        
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N = X.shape[0]
        X = X.reshape(N,-1)
        #print('X shape = ',X.shape)
        N, D = X.shape
        h1,cache1= affine_forward(X,W1,b1)
        #h1_t = X.dot(W1) + b1 #(NxD)*(DxH)
        h2,cache2 = relu_forward(h1)
        #h1 = np.maximum(0,h1_t)
        #print "W2 shape = ", W2.shape
        scores,cache3 = affine_forward(h2,W2,b2)
        #scores = h1.dot(W2) + b2#(NxH)*(HxC) 
        #print "scores = " % scores
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        #pass
        loss,ds = softmax_loss(scores, y)
        reg = self.reg
        #print('reg =',reg)
        #loss += (0.5 * reg * np.sum(W1 * W1) + 0.5 * reg * np.sum(b1*b1))
        #loss += (0.5 * reg * np.sum(W2 * W2) + 0.5 * reg * np.sum(b2*b2))
        loss += 0.5 * reg * np.sum(W1 * W1) 
        loss += 0.5 * reg * np.sum(W2 * W2)
        
        dout = 1
       
        dx3, dw3, db3 = affine_backward(ds,cache3)
        dw3 += reg * W2
        #db3 += reg * b2
        dx2 = relu_backward(dx3, cache2)
        dx1, dw1, db1 = affine_backward(dx2,cache1)
        dw1 += reg * W1
        #db1 += reg * b1
        grads['W1'] = dw1
        grads['b1'] = db1
        grads['W2'] = dw3
        grads['b2'] = db3
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        self.caches = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        '''
        num of layer == num of weight ==num of hidden layer +1
        x|-W1-|H1|-W2-|H2|-W3-|s
        '''
        W_names = []
        b_names = []
        gamma_names = []
        beta_names = []
        h_input_dim = input_dim;
        L = self.num_layers
        for ind in range(self.num_layers - 1):
        #for ind in range(L):
            hidden_dim = hidden_dims[ind]
            W_name = 'W' + str(ind)
            b_name = 'b' + str(ind)

            self.params[W_name] =  weight_scale * np.random.randn(h_input_dim, hidden_dim)
            self.params[b_name] = np.zeros(hidden_dim)
            if self.use_batchnorm:
                gamma_name = 'gamma' + str(ind)
                
                beta_name = 'beta' + str(ind)
                self.params[gamma_name] = np.ones(hidden_dim)
                self.params[beta_name] = np.zeros(hidden_dim)
                
                gamma_names.append(gamma_name)
                beta_names.append(beta_name)
            h_input_dim = hidden_dim
            
            W_names.append(W_name)
            b_names.append(b_name)
        W_name = 'W' + str(self.num_layers - 1)
        b_name = 'b' + str(self.num_layers - 1)
        W_names.append(W_name)
        b_names.append(b_name)
        self.params[W_name] =  weight_scale * np.random.randn(h_input_dim,num_classes )
        self.params[b_name] = np.zeros(num_classes) 

            

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        

        H = X
        L = self.num_layers
      
        for ind in range(L - 1):
            w_name = 'W' + str(ind)
            b_name = 'b' + str(ind)

            W = self.params[w_name]
            b = self.params[b_name]
            
            if self.use_batchnorm:
                gamma_name = 'gamma' + str(ind)
                beta_name = 'beta' + str(ind)
                gamma = self.params[gamma_name]
                beta = self.params[beta_name]
         
            out_aff, cache_aff = affine_forward(H, W, b)
            cache_name = "aff_" + str(ind)
            self.caches[cache_name] = cache_aff
            if self.use_batchnorm:
                #print('ind = ',ind,'bn_params lenght = ',len(self.bn_params))
                out_bat, cache_bat = batchnorm_forward(out_aff, gamma, beta,self.bn_params[ind] )
                cache_name = "bat_" + str(ind)
                self.caches[cache_name] = cache_bat

                out_relu, cache_relu = relu_forward(out_bat)
            else:
                out_relu, cache_relu = relu_forward(out_aff)
            cache_name = "relu_" + str(ind)
            self.caches[cache_name] = cache_relu
            if self.use_dropout:
                out_drop, cache_drop = dropout_forward(out_relu, self.dropout_param)
                cache_name = "drop_" + str(ind)
                self.caches[cache_name] = cache_drop
                H = out_drop
            else:
                H = out_relu
        w_name = 'W' + str(L-1)
        b_name = 'b' + str(L-1)
        W = self.params[w_name]
        b = self.params[b_name]
        scores,cache_aff2 = affine_forward(H,W,b)
        cache_name = "aff_" + str(L-1)
        self.caches[cache_name] = cache_aff2
        #print("scores dim is ",scores.shape)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores
        
        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, ds = softmax_loss(scores, y)
        reg = self.reg
        for ind in range(L):
            w_name = 'W' + str(ind)
            W = self.params[w_name]
            loss += 0.5 * reg * np.sum(W * W) 
        
        dout = 1
        '''
        {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
        '''
        cache_name = "aff_"+str(L-1) 
        dx_L, dw_L, db_L = affine_backward(ds,self.caches[cache_name])
        x_L, w_L, b_L = self.caches[cache_name]
        grad_w_name = 'W' + str(L-1)
        dw_L += reg * w_L
        grads[grad_w_name] = dw_L
        grad_b_name = 'b' + str(L-1)
        grads[grad_b_name] = db_L
        
        dH = dx_L
        for ind in range( L - 1 ): # L = 3 => ind = 0,1, 
            r_ind = ( L-1 ) - ind -1  # r_ind = 2-0-1 = 1 2-1-1 = 0
            if self.use_dropout:
                dx_dp = dropout_backward(dH,self.caches[("drop_"+str(r_ind))])
                dx_rl = relu_backward(dx_dp,self.caches[("relu_"+str(r_ind))])
            else:
                dx_rl = relu_backward(dH,self.caches[("relu_"+str(r_ind))])
            if self.use_batchnorm:
                dx_bn, dgamma_bn, dbeta_bn = batchnorm_backward(dx_rl,self.caches[("bat_"+str(r_ind))])
                grad_gamma_name = 'gamma'+str(r_ind)
                grad_beta_name = 'beta'+str(r_ind)
                grads[grad_gamma_name] = dgamma_bn
                grads[grad_beta_name] = dbeta_bn
                caches_aff = self.caches[("aff_"+str(r_ind))]
                dx_af, dw_af, db_af = affine_backward(dx_bn, caches_aff)
            else:
                caches_aff = self.caches[("aff_"+str(r_ind))]
                dx_af, dw_af, db_af = affine_backward(dx_rl, caches_aff)
            x_aff, w_aff, b_aff = caches_aff
            dw_af += reg * w_aff
            grad_w_name = 'W' + str(r_ind)
            grads[grad_w_name] = dw_af
            grad_b_name = 'b' + str(r_ind)
            grads[grad_b_name] = db_af
            dH = dx_af
            
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
