from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    num_of_example = x.shape[0]
    X = x.reshape(num_of_example,-1) # N * D
    out = X.dot(w) + b
    #print("dim of out = %d" , out.shape)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    dx = dout.dot(w.T)
    dx = dx.reshape(x.shape)
    N = x.shape[0]
    X = x.reshape(N,-1) # N x D
    dw = (X.T).dot(dout)
    one_vec = np.ones((1,N))
    M = dout.shape[1]
    db = (one_vec.dot(dout)).reshape(M)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.maximum(0,x)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    dout[x<0] = 0
    
    dx = dout
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #######################################################################
        sample_mean = x.sum(axis = 0)/N
        sample_var = ((x - sample_mean)**2).sum(axis = 0)/N
        x_hat = (x - sample_mean)/np.sqrt(sample_var + eps)
        y = gamma*x_hat + beta
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        out = y
        cache = (sample_mean,sample_var,x_hat,gamma,beta,x,eps)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        x_hat = (x - running_mean)/np.sqrt(running_var + eps)
        
        y = gamma*x_hat + beta
        

        cache = (x_hat)
        out = y
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    ###########################################################################
    N = dout.shape[0]
    sample_mean, sample_var, x_hat, gamma, beta, x, eps = cache
    
    dx_hat = dout* gamma
    dsample_var = (dx_hat*(x - sample_mean)).sum(axis = 0)*(-1/2)*(sample_var + eps)**(-3/2)
    dsample_mean = (dx_hat*-1/(np.sqrt(sample_var + eps))).sum(axis = 0)
    dx = dx_hat*1/np.sqrt(sample_var + eps) + dsample_var*2*(x - sample_mean)/N + dsample_mean/N
    dgamma = (dout*x_hat).sum(axis = 0)
    dbeta = (dout).sum(axis = 0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta

def rel_error2(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    N = dout.shape[0]
    sample_mean, sample_var, x_hat, gamma, beta, x, eps = cache
   
    vareps = sample_var + eps
    dx_hat = dout* gamma
    common = dx_hat/(np.sqrt(vareps))

    dsample_var = (dx_hat*(x - sample_mean)).sum(axis = 0)*(-1/2)*(sample_var + eps)**(-3/2)
    dsample_mean = (common*-1).sum(axis = 0)
    dx = common + dsample_var*2*(x - sample_mean)/N + dsample_mean/N
    
    
    dgamma = (dout*x_hat).sum(axis = 0)
    dbeta = (dout).sum(axis = 0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        mask = ( np.random.rand(*x.shape) < p ) / p
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        mask = np.ones(x.shape)
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################
    out = mask*x
    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = dout*mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################

    N,C1,H,W = x.shape
    F,C2,HH,WW = w.shape
    #print("HH = %d, WW = %d" %(HH,WW))
    stride = conv_param['stride']
    pad = conv_param['pad']
    H_dot = int(1 + (H + 2 * pad - HH) / stride)
    W_dot = int(1 + (W + 2 * pad - WW) / stride)
    out = np.zeros((N,F,H_dot,W_dot))
   
    pad_x = np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)),mode='constant')   
    #print(pad_x)
    xr = pad_x.reshape(N,-1)
    wr = w.reshape(F,-1)
    #print('xr shape = ', xr.shape)
    x2 = pad_x[:,np.newaxis,:,:,:] #x2[N,1,C,H,W]
    xr2 = xr[:,np.newaxis,:] # xr2[N,1,:]
    w2 = w[np.newaxis,:,:,:,:]#w2[1,F,C,HH,WW]
    wr2 = wr[np.newaxis,:,:]#wr2[1,F,:]
    #print('x shape = ', x.shape)
    #print('pad x shape = ', pad_x.shape)
    #print('x2 shape = ', x2.shape)
    #print('w shape = ', w.shape)
    #print('w2 shape = ', w2.shape)
    #b2= b[np.newaxis,:,np.newaxis,np.newaxis]
    b2 = b.reshape((1,F))
    #b3 = np.vstack((b2,b2)) #(2,3)
    b3 = b[np.newaxis,:]
    #print('b2 shape = ',b2.shape)
    
    for h_ind in range(H_dot):
        h_up_bd = HH + stride*(h_ind)
        h_lw_bd = stride*(h_ind)
        
        for w_ind in range(W_dot):
            w_up_bd = WW + stride*(w_ind)
            w_lw_bd = stride*(w_ind)
            #print("(h_ind,w_ind)(h_u,h_l,w_u,w_l = (%d,%d)(%d,%d,%d,%d)" %(h_ind,w_ind,h_up_bd,h_lw_bd,w_up_bd,w_lw_bd))
 
            xt = x2[:,:,:,h_lw_bd:h_up_bd,w_lw_bd:w_up_bd]
            xtr = xt.reshape(x2.shape[0],x2.shape[1],-1)
            #ot1 = np.sum(xt * w2,axis=2) # 2344
            #print('shape ot1 = ', ot1.shape)
            #ot2 = np.sum(ot1, axis = 2) # 234
            #print('shape ot2 = ', ot2.shape)
            #ot3 = np.sum(ot2, axis = 2)#23
            #print('shape ot3 = ', ot3.shape)
            #out2[:,:,h_ind,w_ind] = ot3 + b3
            
            out[:,:,h_ind,w_ind] = np.sum(xtr*wr2,axis=2) + b3
            #print(out)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    #print('out shape =',out.shape)
    #print(out)
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    """
    
    dx, dw, db = None, None, None
    x,w,b,conv_param = cache
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    print('shape dx = ',dx.shape)
    print('shape dw = ',dw.shape)
    print('shape db = ',db.shape)
    print('shape dout =',dout.shape)
    
    stride = conv_param['stride']
    pad = conv_param['pad']
    N,C1,H,W = x.shape
    F,C2,HH,WW = w.shape
    
    print('H,W,HH,WW=(%d,%d,%d,%d)'%(H,W,HH,WW))
    H_dot = int(1 + (H + 2 * pad - HH) / stride)
    W_dot = int(1 + (W + 2 * pad - WW) / stride)
    print('H_dot,W_dot =(%d,%d)'%(H_dot,W_dot))
    pad_x = np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)),mode='constant')
    pad_dx = np.pad(dx,((0,0),(0,0),(pad,pad),(pad,pad)),mode='constant')  
    
    brod_ones_w = np.ones((N,F,C1,HH,WW))
    brod_ones_x = np.ones((N,F,C2,pad_x.shape[2],pad_x.shape[3]))    
    
    x2 = pad_x[:,np.newaxis,:,:,:]*brod_ones_x #x2[N,1,C,H,W]
    dx2 = pad_dx
    print('x2 shape = ',x2.shape)
    w2 = w[np.newaxis,:,:,:,:]*brod_ones_w#w2[1,F,C,H,W]
    print('w2 shape = ',w2.shape)

    '''
    out[:,:,h_ind,w_ind] = np.sum(xtr*wr2,axis=2) + b3
    - dx: Input data of shape (N, C, H, W)
    - dw: Filter weights of shape (F, C, HH, WW)
    - db: Biases, of shape (F,)
    # (N,F,C,H,W)
    - dout: (N,F,H_dot,W_dot)
    - w2: N,F,C,HH,WW
    - x2: N,F,C,H,W   
    '''
    for h_ind in range(H_dot):
        h_up_bd = HH + stride*(h_ind)
        h_lw_bd = stride*(h_ind)     
        for w_ind in range(W_dot):
            w_up_bd = WW + stride*(w_ind)
            w_lw_bd = stride*(w_ind)
            db += np.sum((dout[:,:,h_ind,w_ind]*1), axis = 0)
            dout2 = dout[:,:,h_ind,w_ind,np.newaxis,np.newaxis] 
            for c_ind in range(C1):             
                dx2[:,c_ind,h_lw_bd:h_up_bd,w_lw_bd:w_up_bd] += (np.sum(dout2 *(w2[:,:,c_ind,:,:]), axis=1 )) 
                dw[:,c_ind,:,:] += np.sum(dout2 * x2[:,:,c_ind,h_lw_bd:h_up_bd,w_lw_bd:w_up_bd], axis=0) 
                
    dx = dx2[:,:,pad:-pad,pad:-pad]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    N,C,H,W = x.shape
    #print('x shape = (%d,%d,%d,%d)'%(N,C,H,W))
    out = None

    pl_h = pool_param['pool_height']
    pl_w = pool_param['pool_width']
    #print('pl_h, pl_w = %d, %d'% (pl_h,pl_w))
    strd = pool_param['stride']
    H2 = int((H-pl_h)/strd +1)
    W2 = int((W- pl_w)/strd +1)
    #print('H2,W2 = (%d,%d)'%(H2,W2))
    out = np.zeros((N,C,H2,W2))
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ########################################################################### 
    for h_ind in range(H2):
        hl_ind = h_ind*strd
        hu_ind = hl_ind + pl_h        
        for w_ind in range(W2):
            wl_ind = w_ind*strd
            wu_ind = wl_ind + pl_w
            
            x2 = x[:,:,hl_ind:hu_ind,wl_ind:wu_ind].reshape(N,C,-1)
            #print('x2 shape = ',x2.shape)
            #print('x2=',x2)
            #print('np.amax(x2,axis = 2) = ',np.amax(x2,axis = 2))
            out[:,:,h_ind,w_ind] = np.amax(x2,axis = 2)
            
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    x, pool_param = cache
    dx = None
    N,C,H2,W2 = dout.shape
    _,_,H,W = x.shape
    #print('dx shape = (%d,%d,%d,%d)'%(N,C,H2,W2))
    #print('x shape = (%d,%d,%d,%d)'%(N,C,H,W))

    pl_h = pool_param['pool_height']
    pl_w = pool_param['pool_width']
    strd = pool_param['stride']
    dx = np.zeros_like(x)
    max_rlv_ind = np.zeros((N,C,1,2))
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    for h_ind in range(H2):
        hl_ind = h_ind*strd
        hu_ind = hl_ind + pl_h        
        for w_ind in range(W2):
            wl_ind = w_ind*strd
            wu_ind = wl_ind + pl_w
            x2 = x[:,:,hl_ind:hu_ind,wl_ind:wu_ind].reshape(N,C,-1)
            max_ind = np.argmax(x2,axis = 2)
           
            rlv_h_ind= (max_ind//pl_h).astype(int)
            rlv_w_ind = (max_ind%pl_h).astype(int)

            hl_ind2 = rlv_h_ind + hl_ind
            hu_ind2 = hl_ind2 + 1
            wl_ind2 = rlv_w_ind + wl_ind
            wu_ind2 = wl_ind2 + 1

            for n_ind in range(N):
                for c_ind in range(C):                
                    dx[n_ind,c_ind,hl_ind2[n_ind,c_ind]:hu_ind2[n_ind,c_ind],wl_ind2[n_ind,c_ind]:wu_ind2[n_ind,c_ind]] = dout[n_ind,c_ind,h_ind,w_ind]
            
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    x2 = x.transpose(1,0,2,3)
    #print('x shape = ', x.shape)
    #print('x2 shape = ', x2.shape)
    x2r = (x2.reshape(x2.shape[0],-1)).T
    #print('x2r shape = ', x2r.shape)
    out2r,cache = batchnorm_forward(x2r, gamma, beta, bn_param)
    out2 = (out2r.T).reshape(x2.shape)
    out = out2.transpose(1,0,2,3)
    #print('out2r shape = ', out2r.shape)
    #print('out2 shape = ', out2.shape)
    #print('out shape = ', out.shape)
    #print('batchnorm forward dout shape = ', out.shape)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    dout2 = dout.transpose(1,0,2,3)
    dout2r = (dout2.reshape(dout2.shape[0],-1)).T
    dx2r, dgamma, dbeta =  batchnorm_backward(dout2r, cache)
    dx2 = (dx2r.T).reshape(dout2.shape)
    dx = dx2.transpose(1,0,2,3)
    #print('dx shape = ', dx.shape)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
