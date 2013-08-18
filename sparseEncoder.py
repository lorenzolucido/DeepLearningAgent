###################################
### This file has been adapted ####
###   but needs to be recoded  ####
###################################

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
#import utils

# This method is used to enforce the dimensionality of matrices since NumPy is a
# bit aggressive about allowing operators over non-matching dimensions.
def ASSERT_SIZE(matrix, shape):
  if matrix.shape != shape:
    raise AssertionError("Wrong shape: %s expexted: %s" %
                            (matrix.shape, shape))

# This wraps the parameters for the sparse autoencoder.
class SparseEncoderOptions:
  #  These network parameters are specified by by Andrew Ng specifically for
  #  the MNIST data set here:
  #     [[http://ufldl.stanford.edu/wiki/index.php/Exercise:Vectorization]]
  def __init__(self,              
               hidden_size,
               output = 'Sigmoid',
               isAutoEncoder = True,
               useBias = True,
               sparsity = 0.1,
               learning_rate = 3e-3,
               beta = 3,
               output_dir = "output",
               max_iterations = 500):    
    self.hidden_size = hidden_size
    self.output = output
    self.useBias = useBias
    self.isAutoEncoder = isAutoEncoder
    self.sparsity_param = sparsity
    self.learning_rate = learning_rate
    self.beta = beta
    self.output_dir = output_dir
    self.max_iterations = max_iterations
    self.visible_size = None
    self.output_size = None

class SparseEncoderSolution:
  def __init__(self, W1, W2, b1, b2):
    self.W1 = W1
    self.W2 = W2
    self.b1 = b1
    self.b2 = b2

# The SparseAutoEncoder object wraps all the data needed in order to train a
# sparse autoencoder.  Its constructor takes a SparseAutoEncoderOptions and a
# v x m matrix where v is the size of the visible layer of the network.
class SparseEncoder:
  def __init__(self, options, data, labels=None):
    self.options = options
    self.data = data
    self.options.visible_size = data.shape[0]
    if self.options.isAutoEncoder: 
        self.labels = self.data
        self.options.output_size = self.options.visible_size
    else:
        self.labels = labels
        self.options.output_size = self.labels.shape[0]
    self.frame_number = 0

  # Convert the matrices to a flat vector.  This is needed by 'fmin_l_bfgs_b'.
  def flatten(self, W1, W2, b1, b2):
    return np.array(np.hstack([W1.ravel('F'), W2.ravel('F'),
                               b1.ravel('F'), b2.ravel('F')]), order='F')

  # Convert the flat vector back to the W1, W2, b1, and b2 matrices.
  def unflatten(self, theta):
    hidden_size = self.options.hidden_size
    visible_size = self.options.visible_size
    output_size = self.options.output_size
    vh = hidden_size * visible_size
    ho = hidden_size * output_size
    W1 = theta[0:vh].reshape([hidden_size, visible_size], order='F')
    W2 = theta[vh:vh+ho].reshape([output_size, hidden_size], order='F')
    b1 = theta[vh+ho:vh+ho+hidden_size].reshape([hidden_size, 1], order='F')
    b2 = theta[vh+ho+hidden_size:].reshape([output_size, 1], order='F')
    return (W1, W2, b1, b2)

  # Create the random values for the parameters to begin learning.
  def initializeParameters(self):
    hidden_size = self.options.hidden_size
    visible_size = self.options.visible_size
    output_size = self.options.output_size
    r = np.sqrt(6) / np.sqrt(hidden_size + visible_size + 1)
    W1 = np.random.random([hidden_size, visible_size]) * 2 * r - r;
    W2 = np.random.random([output_size, hidden_size]) * 2 * r - r;
    b1 = np.zeros([hidden_size, 1])
    b2 = np.zeros([output_size, 1])

    return self.flatten(W1, W2, b1, b2)

  # <div class='math'>1/(1 + e^{-x})</div>
  def sigmoid(self, x):
    return 1.0 / (1.0 + np.exp(-x))

  # ==Forward pass==
  # Note: even though the dimensionality doesn't match because <p>$$b1$$</p>
  # is a vector, numpy will apply b1 to every column.
  def feedForward(self, x, W1, W2, b1, b2):
    visible_size = self.options.visible_size
    hidden_size = self.options.hidden_size
    output_size = self.options.output_size
    ASSERT_SIZE(W1, (hidden_size, visible_size))

    m = x.shape[1]
    
    z2 = np.dot(W1, x)
    if self.options.useBias: z2 += b1
    a2 = self.sigmoid(z2)
    ASSERT_SIZE(a2, (hidden_size, m))

    z3 = np.dot(W2, a2) 
    if self.options.useBias: z3 += b2
        
    a3 = self.sigmoid(z3) if self.options.output!='Linear' else z3
    ASSERT_SIZE(a3, (output_size, m))
    return a2, a3

  # Compute the cost function J and the gradient for an input.  Note that this
  # takes a flattened W1, W2, b1, b2 because of fmin_l_bfgs_b.
  def sparseEncoderCost(self, theta):
    visible_size = self.options.visible_size
    hidden_size = self.options.hidden_size
    output_size = self.options.output_size
    lamb = self.options.learning_rate
    rho = self.options.sparsity_param
    beta = self.options.beta

    x = self.data
    y = self.labels
    m = x.shape[1]

    W1, W2, b1, b2 = self.unflatten(theta)
    ASSERT_SIZE(W1, (hidden_size, visible_size))
    ASSERT_SIZE(W2, (output_size, hidden_size))
    ASSERT_SIZE(b1, (hidden_size, 1))
    ASSERT_SIZE(b2, (output_size, 1))
    
    self.frame_number += 1

    a2, a3 = self.feedForward(x, W1, W2, b1, b2)

    # Compute average activation for an edge over all data
    rho_hat = np.mean(a2, 1)[:, np.newaxis]
    ASSERT_SIZE(rho_hat, (hidden_size, 1))
    kl = rho*np.log(rho/rho_hat) + (1-rho)*np.log((1-rho)/(1-rho_hat))
    '''
    cost = 0.5/m * np.sum((a3 - y)**2) + \
           (lamb/2.)*(np.sum(W1**2) + np.sum(W2**2)) + \
           beta*np.sum(kl)
    '''
    ###############    
    outputCost = np.sum(-y*np.log(a3) - (1-y)-np.log(1-a3)) if self.options.output == 'binary'\
                    else np.sum((a3 - y)**2)
    ###############
    
    cost = 0.5/m * outputCost + \
           (lamb/2.)*(np.sum(W1**2) + np.sum(W2**2)) + \
           beta*np.sum(kl)
    # We set <span class='math'>y</span> equal to the input since we're learning
    # an identity function
    #y = x
    delta3 = outputCost
    if self.options.output!='linear': delta3 *=  a3*(1-a3)
    ASSERT_SIZE(delta3, (output_size, m))

    sparsity = -rho/rho_hat + (1-rho)/(1-rho_hat)
    ASSERT_SIZE(sparsity, (hidden_size, 1))

    delta2 = (np.dot(W2.T, delta3) + beta * sparsity) * a2 * (1-a2)
    ASSERT_SIZE(delta2, (hidden_size, m))

    W2_grad = 1./m * np.dot(delta3, a2.T) + lamb * W2
    ASSERT_SIZE(W2_grad, (output_size, hidden_size))

    # [:, newaxis] makes this into a matrix
    b2_grad = 1./m * np.sum(delta3, 1)[:, np.newaxis] if self.options.useBias else b2
    ASSERT_SIZE(b2_grad, (output_size, 1))

    # sum the rows of delta3 and then mult by  1/m
    W1_grad = 1./m * np.dot(delta2, x.T) + lamb * W1
    ASSERT_SIZE(W1_grad, (hidden_size, visible_size))

    b1_grad = 1./m * np.sum(delta2, 1)[:, np.newaxis] if self.options.useBias else b1
    ASSERT_SIZE(b1_grad, (hidden_size, 1))

    grad = self.flatten(W1_grad, W2_grad, b1_grad, b2_grad)
    return (cost, grad)

  # Actually run gradient descent.  Call mySparseAutoEncoder.learn() to learn
  # the parameters of W1, W2, b1, and b2 for this network and this data.
  def learn(self):
    def f(theta):
      return self.sparseEncoderCost(theta)
    theta = self.initializeParameters()
    #same_theta = theta.copy()
    x, f, d = scipy.optimize.fmin_l_bfgs_b(f, theta,
                                           maxfun= self.options.max_iterations,
                                           iprint=0, m=20)
    W1, W2, b1, b2 = self.unflatten(x)
    if self.options.output == 'binary':
        _, predictions = self.feedForward(self.data,W1,W2,b1,b2)
        print ' ******** REPORT ******** '
        print ' Error on Training set : ', float(np.sum(self.labels==np.round(predictions)))/float(len(self.labels.flat)), ' %'
        print ' ************************ '
        print self.data, np.round(predictions)
    #utils.save_as_figure(W1.T, "%s/network.png" % self.options.output_dir)

    return SparseEncoderSolution(W1, W2, b1, b2)
