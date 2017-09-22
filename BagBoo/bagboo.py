import numpy as np
import sklearn.datasets as ds
import pylab as pl
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
import random as r
import sys

def MSE(x, y):
    return ((x - y)**2).sum() / float(len(x))


def reduce_var(X):
    return X.sum() ** 2 / len(X)
#calculate variance of set
def variance_calculate(X):
#     n_points = X.shape[0]
#     var = 0
    
#     for i1 in xrange(n_points):
#         for i2 in xrange(i1 + 1, n_points):
#             var += (X[i1] - X[i2]) ** 2;
#             print i1, i2, var
    
#     print "calculate varianse of ", X, (1.0 / (2 * n_points ** 2)) * var
    
#     return (1.0 / n_points ** 2) * var
#     print X
    return np.var(X)
    
# search best split in X of split_feature
def search_best_split_of_feat(X, Y, split_feature, orig_var, common_term, min_samples_leaf):
    best_gain = 0
    checked_split_value = []
    best_split_value = None
    res_X_l = None
    res_X_r = None
    res_Y_l = None
    res_Y_r = None
    
    for split_value in X[:, split_feature]:
        if split_value in checked_split_value:
            continue
        checked_split_value.append(split_value)
#         print np.where(X[:, split_feature] < split_value)
        X_l = X[np.where(X[:, split_feature] < split_value)]
        X_r = X[np.where(X[:, split_feature] >= split_value)]
        Y_l = Y[np.where(X[:, split_feature] < split_value)]
        Y_r = Y[np.where(X[:, split_feature] >= split_value)]
#         print "split_value: ", split_value
#         print X_l, X_r, Y_l, Y_r
#         print X_l.shape[0]
#         print X_r.shape[0]
        if X_l.shape[0] >= min_samples_leaf and X_r.shape[0] > min_samples_leaf:
            # var_l = variance_calculate(Y_l)
            # var_r = variance_calculate(Y_r)
            var_l = reduce_var(Y_l)
            var_r = reduce_var(Y_r)
            
#             print orig_var, var_l, var_r,\
#                 (orig_var - (len(Y_l) / float(len(Y))) * var_l - (len(Y_r) / float(len(Y))) * var_r), best_gain
#             if (orig_var - var_l - var_r) > best_gain:
            if (orig_var - common_term +  var_l + var_r) > best_gain:
                best_gain = orig_var - (len(Y_l) / float(len(Y))) * var_l - (len(Y_r) / float(len(Y))) * var_r
                best_split_value = split_value
                res_X_l = X_l
                res_X_r = X_r
                res_Y_l = Y_l
                res_Y_r = Y_r
    
    return best_gain, best_split_value, res_X_l, res_X_r, res_Y_l, res_Y_r

class Tree:
    def __init__(self, split_feature = None, split_val = None,\
                 left = None, right = None, n_points = None, depth = 10, min_samples_leaf = 1):
        self.split_feature = split_feature
        self.split_val = split_val
        self.left = left
        self.right = right
        self.data = None
        self.n_points = n_points
        self.depth = depth
        self.min_samples_leaf = min_samples_leaf
    
    #return MSE
    def score(self, X, Y):
        predicted = self.predict(X)
        MSE = ((predicted - Y)**2).sum() / float(len(X))
        print "MSE: ", MSE

    def predict_one(self, sample):
        if self.data is not None:
            return self.data

        if sample[self.split_feature] >= self.split_value:
            return self.right.predict_one(sample)
        else:
            return self.left.predict_one(sample)

    def predict(self, X):
        predicted = []
        for sample in X:
#             print sample
            predicted.append(self.predict_one(sample))
        return np.asarray(predicted)
            
    #building tree
    def fit (self, X, Y):
#         print "start_fitting:", X, Y
        self.n_points = X.shape[0]
        if self.n_points <= self.min_samples_leaf or self.depth == 0:
#             print "N_POINTS: ", self.n_points
            self.data = np.mean(Y)
            return self
        
        best_gain = 0
        res_split_value = 0
        res_split_feature = None
        res_X_r = None
        res_X_l = None
        res_Y_l = None
        res_Y_r = None
        orig_var = variance_calculate(Y)
        common_term = Y**2.sum() / len(Y)
        
        n_feat = X.shape[1]

        for feat in xrange(n_feat):
#             print 
#             print "Analusys of feature ", feat
            gain, split_value, X_l , X_r, Y_l, Y_r = search_best_split_of_feat(X, Y, feat, orig_var, common_term, self.min_samples_leaf)
            if gain > best_gain:
                best_gain = gain
                res_split_value = split_value
                res_split_feature = feat
                res_X_r = X_r
                res_X_l = X_l
                res_Y_l = Y_l
                res_Y_r = Y_r
        
#         print "BEST SPLIT: "
#         print res_split_feature, res_split_value
#         print res_X_l, res_X_r, res_Y_l, res_Y_r
#         print 
            
        if best_gain > 0:
            self.split_feature = res_split_feature
            self.split_value = res_split_value
            if self.depth is None:
                self.left = Tree(min_points_leaf = self.min_points_leaf).fit(res_X_l, res_Y_l)
                self.right = Tree(min_points_leaf = self.min_points_leaf).fit(res_X_r, res_Y_r)
            else:
                self.left = Tree(depth = self.depth - 1, min_samples_leaf = self.min_samples_leaf).fit(res_X_l, res_Y_l)
                self.right = Tree(depth = self.depth - 1, min_samples_leaf = self.min_samples_leaf).fit(res_X_r, res_Y_r)
                
        else:
            self.n_points = len(Y)
#             print "N_POINTS: ", self.n_points
#             print X
#             print Y
            self.data = np.mean(Y)
        return self
            
    def print_tree(self):
        if self.data != None:
            print str(self.data)
        else:
            print str(self.split_feature) + ": " + str(self.split_value) + "?"
            print "\t T->"
            if (self.right):
                self.right.print_tree()
            print "\t F->"
            if self.left:
                self.left.print_tree()
        

class Gradient_Boosting:
    def __init__(self, n_estimators=10, shrinkage=0.05, max_depth=10, min_samples_leaf=1):
        self.estimators_list = []
        self.n_estimators = n_estimators
        self.shrinkage = shrinkage
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        
    def fit(self, X, Y):
        self.estimators_list = []
        first_estimator = Tree(depth= self.max_depth, min_samples_leaf= self.min_samples_leaf).fit(X, Y)
        self.estimators_list.append(first_estimator)
        
        current_predict = first_estimator.predict(X)
#         sys.stderr.write('\rLearning estimator number: 0'+ "/" + str(self.n_estimators))
#         print  "\tLearning estimator number: 0 ; MSE error on train dataset: ", MSE(current_predict, Y)
        
        for i in xrange(1, self.n_estimators):
            
            antigrad = Y - current_predict
            
            new_estimator = Tree(depth=self.max_depth, min_samples_leaf=self.min_samples_leaf)
            new_estimator = new_estimator.fit(X, antigrad)
#             new_estimator.print_tree()
            
#             print set(antigrad)
#             print new_estimator.predict(X)[:10]
            current_predict += self.shrinkage * new_estimator.predict(X)
            
#             if i % 10 == 0:
#                 print "\tLearning estimator number: ", i,\
#                         "; MSE error on train dataset: ", MSE(current_predict, Y)
            
            sys.stderr.write('\rLearning estimator number: '+ str(i)+"/" + str(self.n_estimators) \
                             + "; MSE error on train dataset: " + str(MSE(current_predict, Y)))
            self.estimators_list.append(new_estimator)
    
    def predict(self, X):
        y =  self.estimators_list[0].predict(X)
#         print len(y)
        for estimator in self.estimators_list[1:]:
            y += estimator.predict(X) * self.shrinkage
#             print MSE(y, Y)
        return y

#n_bag -count of bagging iteration
#n_boo - count of tree in Gradien Boosting
#max_depth - max depth of trees TODO:dinamic depth of trees
#min_samples_leaf
#bagging_ratio -cnt of samples (in percent), which using for bagging interation
#RSM and Bagging without replacement
class BagBoo:
    def __init__ (self, n_boo = 10, n_bag = 10, bagging_ratio = 0.1, rsm_ratio = 1, max_depth = 10,\
                  min_samples_leaf = 1, shrinkage = 0.1):
        self.n_boo = n_boo
        self.n_bag = n_bag
        self.bagging_ratio = bagging_ratio
        self.rsm_ratio = rsm_ratio
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.boosting_list = []
        self.shrinkage = shrinkage
        
    def fit(self, X, Y, verbose = 0, X_test = None, Y_test = None):
        cur_sum_predict = 0 
        cur_sum_train = 0
        error_statistic = []
        rsm_cnt = int(self.rsm_ratio * X.shape[1])
        bagging_cnt = int(self.bagging_ratio * X.shape[0])
        print "features in RMS: ", rsm_cnt
        print "samples in bagging: ", bagging_cnt
        for bag_iter in xrange(self.n_bag):
            sys.stderr.write('\rIteration of bagging:'+str(bag_iter) + "/" + str(self.n_bag))
#            print "Iteration of bagging: "+ str(bag_iter) + "/" + str(self.n_bag)
            shuffle_idx_bagging = range(X.shape[0])
            shuffle_idx_rsm = range(X.shape[1])
            
            r.shuffle(shuffle_idx_bagging)
            r.shuffle(shuffle_idx_rsm)
            shuffle_idx_bagging = shuffle_idx_bagging[:bagging_cnt]
            shuffle_idx_rsm = shuffle_idx_rsm[:rsm_cnt]

#             print shuffle_idx_bagging, shuffle_idx_rsm
            X_bag = X[shuffle_idx_bagging][:, shuffle_idx_rsm]
            Y_bag = Y[shuffle_idx_bagging]
#             print "sgs", X_bag, Y_bag
            
            new_boosting = Gradient_Boosting(n_estimators= self.n_boo, max_depth=self.max_depth,\
                                             min_samples_leaf= self.min_samples_leaf, shrinkage = self.shrinkage)
            new_boosting.fit(X_bag, Y_bag)
            
            self.boosting_list.append(new_boosting)
            
            if verbose:
                cur_sum_predict += new_boosting.predict(X_test)
                cur_sum_train += new_boosting.predict(X)
                error_test =  MSE(cur_sum_predict / float(len(self.boosting_list)), Y_test)
                error_train =  MSE(cur_sum_train / float(len(self.boosting_list)), Y)
                print "MSE on test Dataset:", error_test, "Iteration of Bagging:", bag_iter, "/", self.n_bag
                print "MSE on train Dataset:", error_train, "Iteration of Bagging:", bag_iter, "/", self.n_bag
                error_statistic.append(error_test)
               
        return error_statistic    

    def predict(self, X):
        y = np.array([0.0] * X.shape[0])
        for boosting in self.boosting_list:
            y += boosting.predict(X)
        return y / float(self.n_bag)
