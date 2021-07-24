from builtins import range
from builtins import object
import numpy as np
from layers.layers import *
from layers.layer_utils import *



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

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.00, lr = 0.01):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg
        self.lr = lr;
        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        #self.params['W1'] = np.random.normal(scale=weight_scale, size=(input_dim, hidden_dim))
        #self.params['W2'] = np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes))
        self.params['W1'] = np.random.uniform(-1, 1, (input_dim, hidden_dim))
        self.params['W2'] = np.random.uniform(-1, 1, (hidden_dim, num_classes))
        #self.params['W1'] = np.ones((input_dim, hidden_dim))
        #self.params['W2'] = np.ones((hidden_dim, num_classes))
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['b2'] = np.zeros(num_classes)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################



    def upgrade(self,x, dx):
        """
        Uses the RMSProp update rule, which uses a moving average of squared
        gradient values to set adaptive per-parameter learning rates.

        config format:
        - learning_rate: Scalar learning rate.
        - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
          gradient cache.
        - epsilon: Small scalar used for smoothing to avoid dividing by zero.
        - cache: Moving average of second moments of gradients.
        """
        self.params['cache'] = np.zeros_like(x)
        self.params['learning_rate'] = self.lr

        self.params['decay_rate'] = 0.99
        self.params['epsilon'] = 1e-8
        next_x = None
        ###########################################################################
        # TODO: Implement the RMSprop update formula, storing the next value of x #
        # in the next_x variable. Don't forget to update cache value stored in    #
        # config['cache'].                                                        #
        ###########################################################################
        next_x = x - self.params['learning_rate']*dx
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

        return next_x

    def sgd_momentum(self,w, dw):
        """
        Performs stochastic gradient descent with momentum.

        config format:
        - learning_rate: Scalar learning rate.
        - momentum: Scalar between 0 and 1 giving the momentum value.
          Setting momentum = 0 reduces to sgd.
        - velocity: A numpy array of the same shape as w and dw used to store a
          moving average of the gradients.
        """

        self.params['learning_rate'] = self.lr
        self.params['momentum']= 0.9
        v= np.zeros_like(w)

        next_w = None
        ###########################################################################
        # TODO: Implement the momentum update formula. Store the updated value in #
        # the next_w variable. You should also use and update the velocity v.     #
        ###########################################################################
        vx = self.params['momentum'] * v - self.params['learning_rate'] * dw
        next_w = w
        next_w += vx
        self.params['velocity'] = vx
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

        return next_w



    def rmsprop(self,x, dx):
        """
        Uses the RMSProp update rule, which uses a moving average of squared
        gradient values to set adaptive per-parameter learning rates.

        config format:
        - learning_rate: Scalar learning rate.
        - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
          gradient cache.
        - epsilon: Small scalar used for smoothing to avoid dividing by zero.
        - cache: Moving average of second moments of gradients.
        """
        self.params['cache'] = np.zeros_like(x)
        self.params['learning_rate'] = 0.007
        self.params['decay_rate'] = 0.9
        self.params['epsilon'] = 1e-8
        next_x = None
        ###########################################################################
        # TODO: Implement the RMSprop update formula, storing the next value of x #
        # in the next_x variable. Don't forget to update cache value stored in    #
        # config['cache'].                                                        #
        ###########################################################################
        self.params['cache'] =  self.params['decay_rate'] *  self.params['cache'] + (1 -  self.params['decay_rate']) * dx * dx
        next_x = x - ( self.params['learning_rate'] * dx / (np.sqrt( self.params['cache']) +  self.params['epsilon']))

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

        return next_x

    def loss(self, td, vcount):
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
        loss, grads = 0, {}
        for w_t, w_call in td:

           X = np.array(w_t)
           y = np.array(w_call)
           x2, fc_cache1 = affine_forward(X, self.params['W1'], self.params['b1'])
           #x2, relu_cache = relu_forward(a)
           out, fc_cache2 = affine_forward(x2, self.params['W2'], self.params['b2'])

           out1 = np.exp( out - np.max(out))
           scores = out1/np.sum(out1)

           losstmp, dout = w2vec_loss(scores, y, vcount, out)
           loss = loss + losstmp #+ self.reg * ( np.sum(self.params['W1'] * self.params['W1']) + np.sum(self.params['W2'] * self.params['W2']))


           da, grads['W2'], grads['b2'] = affine_backward(dout, fc_cache2)
           #da = relu_backward(dx2, relu_cache)
           dx, grads['W1'], grads['b1'] = affine_backward(da, fc_cache1)
           grads['W1'] += 2 * self.reg * self.params['W1']
           grads['W2'] += 2 * self.reg * self.params['W2']
           self.params['W1'] = self.upgrade(self.params['W1'],grads['W1'])
           self.params['W2'] = self.upgrade(self.params['W2'], grads['W2'])
           #self.params['W1'] = self.sgd_momentum(self.params['W1'],grads['W1'])
           #self.params['W2'] = self.sgd_momentum(self.params['W2'], grads['W2'])
           #self.params['b1'] = self.rmsprop(self.params['b1'], grads['b1'])
           #self.params['b2'] = self.rmsprop(self.params['b2'], grads['b2'])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss



    def getw1(self):
        return self.params['W1']

    def getw2(self):
        return self.params['W2']


