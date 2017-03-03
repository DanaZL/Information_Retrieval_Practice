import numpy as np
import matplotlib.pyplot as plt
import os
import struct
import sklearn.preprocessing
from scipy.misc import logsumexp
import random as r
import scipy.misc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import random

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

# generate the next binary vector
def inc(x):
    for i in xrange(len(x)):
        x[i] += 1
        if x[i] <= 1: return True
        x[i] = 0
    return False


class RBM(object):
    def __init__(self, 
                cnt_visible = 100,
                cnt_hidden = 100,
                units_type = "BB" #first letter - type visible units, second letter - type hidden units  
                ):
        self.cnt_visible = cnt_visible
        self.cnt_hidden = cnt_hidden
        
        self.weights = np.random.normal(0, 0.01, (cnt_visible, cnt_hidden))
        self.vbias = np.zeros(shape = (1, cnt_visible))
        self.hbias = np.zeros(shape = (1, cnt_hidden))
        self.hidden = None
        self.visible= None
        self.input_data = None
        self.momentum = 0
        self.units_type = units_type
        self.sigma_sqr = np.empty(shape = (1, cnt_visible))
        self.sigma_sqr.fill(0.01)
        
    #sigmoid(W * visible_layer + hidden_bias)
    def forward_pass(self):
        self.hidden = sigmoid(np.dot(self.visible, self.weights) + self.hbias)
    
    #sigmoid(W.T * hidden_layer + visible_bias)
    def backward_pass(self):
        if self.units_type == "BB":
            self.visible = sigmoid(np.dot(self.hidden, self.weights.T) + self.vbias)
            
        if self.units_type == "GB":
            raw_visible = np.dot(self.hidden, self.weights.T) + self.vbias
            for unit in xrange(self.visible.shape[1]):
                for sample in xrange(self.visible.shape[0]):
                    self.visible[sample, unit] = np.random.normal(raw_visible[sample, unit],\
                                                                  np.sqrt(self.sigma_sqr[0, unit]) + 0.1e-6, 1)

    def sampling_hid_given_vis(self):
        for row in xrange(self.hidden.shape[0]):
            for col in xrange(self.hidden.shape[1]):
                self.hidden[row, col] = np.random.binomial(1, self.hidden[row, col], 1)

    def sampling_vis_given_hid(self):
        for row in xrange(self.visible.shape[0]):
            for col in xrange(self.visible.shape[1]):
                self.visible[row, col] = np.random.binomial(1, self.visible[row, col], 1)
    
    def sampling_v_h_v(self):
        self.forward_pass()
        self.sampling_hid_given_vis()
        self.backward_pass()
        #self.sampling_vis_given_hid()
    
    def gibbs_sampling(self):
        inp = self.visible 
        #first interation of sampling
        self.sampling_v_h_v()
        if self.units_type == "BB":
            self.pos_phase_weights = inp.T.dot(self.hidden) / float(self.visible.shape[0])  #normalize for batch_size
            self.pos_phase_vbias = inp.mean(axis = 0)
            self.pos_phase_hbias = self.hidden.mean(axis = 0)
        if self.units_type == "GB":
            self.pos_phase_weights = (inp * (1 / (self.sigma_sqr + 0.1e-7))).T.dot(self.hidden) / float(self.visible.shape[0])  #normalize for batch_size
            self.pos_phase_vbias = (inp * (1 / (self.sigma_sqr + 0.1e-7))).mean(axis = 0)
            self.pos_phase_hbias = self.hidden.mean(axis = 0)
            self.pos_phase_log_sqr_sigma = ((1.0 / 2) * (inp - self.vbias)**2 - \
                                            (inp * (self.hidden).dot(self.weights.T).mean(axis = 0)))
        
        for k in range(self.sampling_param - 1):
            self.sampling_v_h_v()
        
        #last iteration of sampling
        self.forward_pass()
        
#         print "last hidden ", self.hidden
        if self.units_type == "BB":
            self.neg_phase_weights = self.visible.T.dot(self.hidden) / float(self.visible.shape[0])  #normalize for batch_size
            self.neg_phase_vbias = self.visible.mean(axis = 0)
            self.neg_phase_hbias = self.hidden.mean(axis = 0)
        if self.units_type == "GB":
            self.neg_phase_weights = (self.visible * (1.0 / (self.sigma_sqr + 0.1e-6))).T.dot(self.hidden)\
                                    / float(self.visible.shape[0])  #normalize for batch_size
            self.neg_phase_vbias = (self.visible * (1.0 / (self.sigma_sqr + 0.01e-6))).mean(axis = 0)
            self.neg_phase_hbias = self.hidden.mean(axis = 0)
            self.neg_phase_log_sqr_sigma = ((1.0 / 2) * (inp - self.vbias)**2 - \
                                            (self.visible * (self.hidden).dot(self.weights.T).mean(axis = 0)))                                

    def update_param(self):
        
        delta_weights = self.momentum * (self.last_delta_weights) + (self.pos_phase_weights - self.neg_phase_weights)
        delta_vbias = self.momentum * (self.last_delta_vbias) + (self.pos_phase_vbias - self.neg_phase_vbias)
        delta_hbias = self.momentum * (self.last_delta_hbias) + (self.pos_phase_hbias - self.neg_phase_hbias)
        
#         print "delta_weights \n", delta_weights
        self.weights += self.learning_ratio * delta_weights
        self.vbias += self.learning_ratio * delta_vbias
        self.hbias += self.learning_ratio * delta_hbias
        if self.units_type == "GB":
            try:
                delta_log_sigma_sqr = np.exp(-1 * np.log(self.sigma_sqr+ 0.0e-6)) * self.momentum * (self.last_delta_log_sqr_sigma)\
                                            + (self.pos_phase_log_sqr_sigma - self.neg_phase_log_sqr_sigma)
            except:
                delta_log_sigma_sqr = 0.1e-6 * self.momentum * (self.last_delta_log_sqr_sigma)\
                                            + (self.pos_phase_log_sqr_sigma - self.neg_phase_log_sqr_sigma)
            
            try:
                self.sigma_sqr = np.exp(np.log(self.sigma_sqr + 0.0e-6) + delta_log_sigma_sqr)
            except:
                self.sigma_sqr = 0.1e-6
            self.last_delta_log_sqr_sigma = delta_log_sigma_sqr
                                            
         
        self.last_delta_weights = delta_weights
        self.last_delta_vbias = delta_vbias
        self.last_delta_hbias = delta_hbias
        
        
    def energy_bin_bin(self, visible, hidden):
        weights_part = np.dot(visible.dot(self.weights), hidden)
#         print self.vbias
#         print visible
        vis_part = self.vbias.dot(visible)
        hid_part = self.hbias.dot(hidden)
        return (-weights_part - vis_part - hid_part)
    
    #calculate normalizing constant
    def calc_log_normalizing_const(self):
        x = np.zeros(shape = self.cnt_visible)
        h = np.zeros(shape = self.cnt_hidden)
        energy_array = []
        while True:
            while True:
                energy_array.append(-1 * self.energy_bin_bin(x,h))
                if not inc(h): 
                    break
            if not inc(x): 
                break  
            
        return logsumexp(np.asarray(energy_array))
    
    def log_likelihood(self, batch):
        LogLikelihood = 0
        log_Z = self.log_normalizing_const
        for X in batch:
            energy_array = []
            h = np.zeros(shape = self.cnt_hidden)
            while True: #sum over all possible values of h
                energy_array.append(self.energy_bin_bin(X,h))
                if not inc(h): break
                    
            LogLikelihood += logsumexp(np.asarray(energy_array)) - log_Z
            
        return LogLikelihood / float(batch.shape[0])
        
    
    def fit(self, X,
            learning_ratio = 0.1, 
            batch_size = 40, 
            training_epochs = 100,
            sampling_param = 4, 
            momentum = 0, 
            verbose = "LogLikelihood"):
        
        self.sampling_param = sampling_param
        self.momentum = momentum
        self.learning_ratio = learning_ratio
    
        self.last_delta_weights = np.zeros(shape = (self.cnt_visible, self.cnt_hidden))
        self.last_delta_vbias = np.zeros(shape = (1, self.cnt_visible))
        self.last_delta_hbias = np.zeros(shape = (1, self.cnt_hidden))
        self.last_delta_log_sqr_sigma = np.zeros(shape = (1, self.cnt_visible))
                                            
        if verbose == "LogLikelihood":
            self.log_normalizing_const = self.calc_log_normalizing_const()
        
        for epoch in range(training_epochs):
            
            batch_samples = r.sample(xrange(X.shape[0]), batch_size)  
            
            self.input_data = X[batch_samples] 
            self.visible = X[batch_samples]
            self.gibbs_sampling()
            self.update_param()
            
            if epoch % 100 == 0: 
                print "Training epoch:" , epoch
                if verbose == 'LogLikelihood':
#                 print "likelihood visible vector", self.visible.mean(axis = 0)
                  print "Log likelihood: ", self.log_likelihood(X[batch_samples])
                if verbose == 'MSE':
#                 print "likelihood visible vector", self.visible.mean(axis = 0)
                  print "MSE distance between input and reconstruction:",\
                        ((X[batch_samples] - self.reconstruct(X[batch_samples])) ** 2).mean(axis=1).mean()
#                 print self.pos_phase_weights
#                 print self.neg_phase_weights
#             print "---------------------------------------------------------"
    def reconstruct(self, X):
        #hid = X.dot(self.weights)
        self.visible = X
        self.sampling_v_h_v()
        self.sampling_vis_given_hid()
        return self.visible
        
        
def soft_softmax(x):
    x = np.array([x[i] - np.max(x[i]) for i in xrange(x.shape[0])])
    x = np.array([np.exp(x[i]) / np.sum(np.exp(x[i])) for i in xrange(x.shape[0])])
    return x

def activate(activate, x):
    if activate == 'sigmoid':
        return (1 / (1 + np.exp(-x)))
    elif activate == 'tanh':
        return np.tanh(x)
    elif activate == 'softmax':
        return soft_softmax(x)
    elif activate == 'ReLU':
        for i in xrange(x.shape[0]):
            for j in xrange(x.shape[1]):
                if (x[i][j] < 0):
                    x[i][j] = 0
        return x 
    
def deriv_activ(activate, x):
    if activate == 'sigmoid':
        return x * (1 - x)
    elif activate == 'tanh':
        return (1.0 - x**2)  
    elif activate == 'softmax':
        return x * (1 - x)
    elif activate == 'ReLU':
        for i in xrange(x.shape[0]):
            for j in xrange(x.shape[1]):
                if (x[i][j] < 0):
                    x[i][j] = 0
                else:
                    x[i][j] = 1
        return x    
        
 
class Layer:
    def __init__(self, num_nodes, prev_num_nodes = -1, bias = True, activate = "sigmoid"):
        self.num_nodes = num_nodes
        self.prev_num_nodes = prev_num_nodes
        self.need_bias = bias;
        # print self.need_bias
        if (self.need_bias):
            self.bias = 2 * np.random.random((1, num_nodes)) - 1
        self.weights_matrix = None
        self.activate = activate

    def forward_pass(self, X):
        if (self.weights_matrix is None) and (self.prev_num_nodes != 1):
            self.weights_matrix = 2*np.random.random((self.prev_num_nodes, self.num_nodes)) - 1

        if (self.need_bias):
            self.Y = activate(self.activate, np.dot(X, self.weights_matrix) + self.bias)
        else:
            self.Y = activate(self.activate, np.dot(X, self.weights_matrix))
        
        return self.Y
    
    def backward_pass(self, WeightedGradientNextLayer, learning_ratio, Y_prev): 
        #calculate gradient and update weights
        Delta = WeightedGradientNextLayer * deriv_activ(self.activate, self.Y)
        Gradient = Y_prev.T.dot(Delta)
        #print "Gradient 0: " + str(Gradient)
        self.weights_matrix -= learning_ratio * Gradient
        
        if (self.need_bias):
            self.bias -= learning_ratio * Delta.mean(0)
        
        return (Delta.dot(self.weights_matrix.T))
    
class OutputLayer(Layer):
    def backward_begin (self, loss_function, OUT, learning_ratio, Y_prev):
        
        if loss_function == "MSE":
            Delta = -1 *(OUT - self.Y) * deriv_activ(self.activate, self.Y)
            Gradient = Y_prev.T.dot(Delta)
            Error = (((OUT - self.Y)**2).mean(axis = 1)).mean()
            #print "Gradient" + str(Gradient)
            try:
                self.weights_matrix -= learning_ratio * Gradient
            except:
                print OUT.shape, self.Y.shape, Delta.shape, Y_prev.shape
            if (self.need_bias):
                #print "bias", self.bias
                #print "delta", Delta.sum()
                self.bias -= learning_ratio * Delta.mean(0)
            
            
        elif loss_function == "NLL":
            
            Error = -1 * (OUT * np.log(self.Y)).sum()
            
            #print self.Y
            Delta = (self.Y - OUT)
            Gradient = Y_prev.T.dot(Delta)
            #print "Gradient: " + str(Gradient[0])
            self.weights_matrix -= learning_ratio * Gradient
            if (self.need_bias):
                #print self.bias
                #print Delta.sum()
                self.bias -= learning_ratio * Delta.mean(0)
        
        return Delta.dot(self.weights_matrix.T), Error
    
    
class InputLayer(Layer):
    def forward_begin (self, X):
        self.Y = X
        
        return self.Y
        
       

class DBN:
    def __init__(self, layers):
        self.layers = layers
        
    def fit(self, X, OUT, batch_size, iteration, learning_ratio, loss_function, RBM_initialize = True):
        
        if (RBM_initialize):
            X_RBM = np.copy(X)
            for layer in xrange(1, len(self.layers)):
                print "____________________________________________________________"
                print "Train weights betweeen layer %i and layer %i" %(layer - 1, layer)
                rbm_model = RBM(cnt_visible = self.layers[layer].prev_num_nodes, cnt_hidden = self.layers[layer].num_nodes)

                rbm_model.fit(X_RBM, learning_ratio = 0.03, batch_size = 30, \
                              training_epochs = 3000, momentum=0.6, sampling_param=3, 
                                 verbose = "MSE")

                self.layers[layer].weights_matrix = rbm_model.weights
                self.layers[layer].bias = rbm_model.hbias
                
                self.visible = X_RBM
                rbm_model.forward_pass()
                rbm_model.sampling_hid_given_vis()
                X_RBM = rbm_model.hidden
#                 #print layer, self.layers[layer].weights_matrix, self.layers[layer].bias
#                 if (self.layers[layer].need_bias):
#                     X_RBM = activate(self.layers[layer].activate, \
#                                      np.dot(X_RBM, self.layers[layer].weights_matrix) + self.layers[layer].bias)
#                 else:
#                     X_RBM = activate(self.layers[layer].activate, np.dot(X_RBM, self.layers[layer].weights_matrix))

                
            
        for iterat in xrange(iteration):
            
            batch_sample = r.sample(xrange(X.shape[0]), batch_size)                
            #batch_sample = xrange(batch_size)
            
            self.layers[0].forward_begin(X[batch_sample])
            
            for i in xrange(1, len(self.layers)):
                self.layers[i].forward_pass(self.layers[i - 1].Y)
    
            WeightedGradientNextLayer_tmp, Error = self.layers[len(self.layers) - 1].backward_begin(loss_function,\
                                                        OUT[batch_sample], learning_ratio, \
                                                        self.layers[len(self.layers) - 2].Y)
        
            for i in xrange(len(self.layers) - 2, 0, -1):
                WeightedGradientNextLayer_tmp = self.layers[i].backward_pass(WeightedGradientNextLayer_tmp, \
                                                                       learning_ratio, self.layers[i - 1].Y)
            
            right_cnt = 0
            if iterat % 100 == 0:
                print "________________________________________________"
                print "Iteration: " + str(iterat)
                print 'Error: '  + str(Error)
                #print "Y_i: " + str(np.argmax(self.layers[len(self.layers) - 1].Y[0]))
                #print "OUT: " + str(np.argmax(OUT[batch_sample[0]]))
                if loss_function == 'NLL':
                    for i in xrange(batch_size):
                        if np.argmax(self.layers[len(self.layers) - 1].Y[i]) == np.argmax(OUT[batch_sample[i]]):
                            right_cnt += 1

                    print "True identified: " + str(right_cnt) + "|" + str(batch_size)
            
                
    
    def predict(self, X):
        self.layers[0].forward_begin(X)
        for i in xrange(1, len(self.layers)):
            self.layers[i].forward_pass(self.layers[i - 1].Y)
        
        if self.layers[len(self.layers) - 1].activate == 'softmax':
            self.layers[len(self.layers) - 1].Y = soft_softmax(self.layers[len(self.layers) - 1].Y)
        return self.layers[len(self.layers) - 1].Y
                
                