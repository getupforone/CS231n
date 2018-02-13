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
  # compute the loss and the gradient
  num_cls = W.shape[1]
  num_train = X.shape[0]
  train_ind_arr = np.arange(num_train)
  margin_vec = np.zeros(num_cls)
  t_mat = np.zeros((num_train,num_cls))
  t_mat[train_ind_arr,y[train_ind_arr]] = 1 
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  for i in np.arange(num_train):
    x_n = X[i,:] #(1xD)
    scores = x_n.dot(W)#(1xD)*(DxC)=(1xC)

    const_c = np.max(scores)

    c_scores = scores - const_c

    norm = np.sum(np.exp(c_scores))
    
    prb = np.exp(c_scores)/norm
    #print 'prb shape = ', prb.shape
    loss += -1*np.log(prb[y[i]])
    
    #t 1-of-K coding target vector
    tgt = t_mat[i,:]
    for j in np.arange(num_cls):
        dW[:,j]+=(prb[j]-tgt[j])*x_n

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
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
  # compute the loss and the gradient
  num_cls = W.shape[1]
  num_train = X.shape[0]
  num_dim = X.shape[1]
  t_ind = np.arange(num_train)
  c_ind = np.arange(num_cls)
  margin_vec = np.zeros(num_cls)
  t_mat = np.zeros((num_train,num_cls))
  t_mat[t_ind,y[t_ind]] = 1 

  scores = X.dot(W)#(NxD)*(DxC)=(NxC)

  const_c = np.max(scores,axis=1)
  
  const_c = const_c.reshape(const_c.shape[0],1)
  #print 'const_c shape=', const_c.shape
  c_scores = scores - const_c
  
  norm = np.sum(np.exp(c_scores),axis=1)
  
  norm = norm.reshape(norm.shape[0],1)
  
  prb = np.exp(c_scores)/norm

  prb_y = prb[t_ind,y[t_ind]]

  loss += -1*np.sum(np.log(prb_y))
  loss /= num_train
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  diff = (prb - t_mat) #(N,C)

  
  #for t in t_ind:
  #    x_i = X[t,:].reshape(num_dim,1)
  #    d_i = diff[t,:].reshape(num_cls,1)
  #    dW += np.kron(x_i,d_i.T)#kron((D,1) ,(1,C)) = (D,C)
        
  dW = (X.T).dot(diff)#(D,N)*(N,C) = (D,C)
  dW /= num_train

  #dW = np.sum(np.kron(X[t_ind,:].T,diff[t_ind,:]))#kron((D,1) ,(1,C)) = (D,C)
  dW += reg * W
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

