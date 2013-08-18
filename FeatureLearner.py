# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 11:27:04 2013

@author: LoL0
"""

from game import Agent

import sys
import numpy as np
import layout
from game import Directions, Actions
import random, util
import time
import cPickle as pickle
import matplotlib.pyplot as plt

from sklearn import preprocessing  # Used in order to encode categorical features
from scikits.statsmodels.tools import categorical  # Used in order to encode categorical features
from sklearn import preprocessing # Used in order to encode categorical features
from scipy import linalg
from sklearn.cluster import KMeans

###### PYBRAIN IMPORTS for usual Neural Networks ######
from pybrain.datasets               import SupervisedDataSet
from pybrain.supervised.trainers    import BackpropTrainer
from pybrain.tools.shortcuts        import _buildNetwork
from pybrain.structure              import SigmoidLayer
from pybrain.structure              import LinearLayer
from pybrain.utilities              import percentError
#######################################################

###### Sparse Encoder and AutoEncoder Imports ######
from sparseEncoder                  import SparseEncoderOptions
from sparseEncoder                  import SparseEncoder
from sparseEncoder                  import SparseEncoderSolution
####################################################

############# Sparse Encoder Options
hidden_size = 500
target_activation = 0.5
lamb = 0.001
beta = 0.1

############# KMeans Options
KMeansMaxIter = 300
numCentroids = 30

####################
## FEATURE LEARNER CLASS
####################
## 
class FeatureLearner(object):
    '''
    The FeatureLearner uses a 'sparse coder' in order to learn
    a mapping from the (gameStates,Action) tuple to the Q-Value.
    '''
    def __init__(self):
        self.learnerType = 'KMeans' #### Choose either 'KMeans', 'Encoder', or 'ClassicNNet'       
        self.trained = False
        self.labelEncoder = preprocessing.LabelEncoder()
        self.encoderOptions = SparseEncoderOptions(        hidden_size,
                                                           output = 'binary',
                                                           isAutoEncoder = True,
                                                           useBias = False,
                                                           learning_rate = lamb,
                                                           beta = beta,
                                                           sparsity = target_activation)
                                                           
        self.dataPreprocessor = DataPreprocessor()                                                           
        self.numCentroids = numCentroids
        self.kMeansMaxIter = KMeansMaxIter
        self.KMeans = KMeans(n_clusters = self.numCentroids, init='k-means++',\
                        n_init=5, max_iter = self.kMeansMaxIter, \
                        precompute_distances=True, verbose=True)
        self.centroids = None
        
    ##### LOAD / SAVE METHODS #### ** Not implemented yet **
    def loadData(self,filename):
        loadedData = pickle.load( open(filename,'r') )
    
    def saveData(self,filename):
        pickle.dump(self, open(filename,'w'))
    #################################
    
    def getFeatures(self, state, action):#, QTable):
        feats = util.Counter()
        if self.trained:
            feats = self.getLearnedFeatures(state,action)#,QTable)
        else:
            feats[self.dataPreprocessor.flattenStateActionTuple(state,action)] = 1.0
        return feats
    
    ### FUNCTION updateFeatures
    #
    # The heart of the Feature Learner :
    # Once called, the function takes the Q Table as a parameter
    # and automatically parameterize an encoder for the (gameState,
    # action) tuples
    #
    def updateFeatures(self,QTable): 
        #print self.maxStateValues
        #self.catEncoder = preprocessing.OneHotEncoder(n_values=self.maxStateValues)
        features_X, labels_Y = self.preprocessData(QTable,train=True)
        
        
        'Starting with full (non-sparse) neural networks'
        #print features_X, labels_Y, labels_Y.shape
        if self.learnerType == 'ClassicNNet': self.classicNeuralNetwork(features_X,labels_Y,autoencoder=False)
        elif self.learnerType == 'Encoder':
            self.Encoder = SparseEncoder(self.encoderOptions, data= features_X.T, labels= labels_Y.T)
            self.trainedParameters = self.Encoder.learn()
            #print self.trainedParameters.W1
        elif self.learnerType == 'KMeans':
            #self.centroids = self.KMeans.fit(features_X).cluster_centers_
            self.runKmeans(features_X)
        
        self.trained = True
        
         
 
    ### FUNCTION preprocessData
    #
    # Prepares the Data for neural network fitting
    # The categorical variables are encoded using
    # one-of-K.
    #
    def preprocessData(self,QTable,train=True):
        if train:
            labels_Y = np.array(QTable.values(),dtype=np.float)
            labels_Y = labels_Y.reshape(-1,labels_Y.shape[1])              
            features_X = self.dataPreprocessor.binarizeData(QTable.keys(),train)
            #if self.learnerType == 'KMeans': features_X = labels_Y
            #features_X = self.dataPreprocessor.normalizeData(features_X,train)
            #features_X = self.dataPreprocessor.orthogonalizeData(features_X,train)
            return features_X, labels_Y
        else:
            features_X = self.dataPreprocessor.binarizeData(QTable,train)
            #features_X = self.dataPreprocessor.normalizeData(features_X,train)
            #features_X = self.dataPreprocessor.orthogonalizeData(features_X,train)
            return features_X
        

       
    ### FUNCTION classicNeuralNetwork 
    #
    # Trains a simple neural network with single hidden layer   
    #
    def classicNeuralNetwork(self,features,labels,autoencoder=False):
        dataSet = SupervisedDataSet(features.shape[1], 1)
        dataSet.setField('input', features)
        if autoencoder: labels = features      
        dataSet.setField('target', labels)
        tstdata, trndata = dataSet.splitWithProportion( 0.25 )
        print features.shape
        simpleNeuralNetwork = _buildNetwork(\
                                    (LinearLayer(features.shape[1],'in'),),\
                                    (SigmoidLayer(20,'hidden0'),),\
                                    (LinearLayer(labels.shape[1],'out'),),\
                                    bias=True)
        trainer = BackpropTrainer(simpleNeuralNetwork, dataset=trndata, verbose=True)#, momentum=0.1)
        trainer.trainUntilConvergence(maxEpochs=15)
        
        trnresult = percentError( trainer.testOnData( dataset=trndata ), trndata['target'] )
        tstresult = percentError( trainer.testOnData( dataset=tstdata ), tstdata['target'] )

        print "epoch: %4d" % trainer.totalepochs, \
          "  train error: %5.2f%%" % trnresult, \
          "  test error: %5.2f%%" % tstresult

        self.neuralNetwork = simpleNeuralNetwork
        
                
    #### FUNCTION extractFeatures
    #
    # Extract features in the same way than the previously 
    # trained data set.
    #
    def extractFeatures(self,state,action):
        assert self.trained   
        return self.preprocessData(self.dataPreprocessor.flattenStateActionTuple(state,action),train=False)
    
    #### FUNCTION getLearnedFeatures
    #
    # If trained, returns the activation values
    # of the hidden layer.
    #
    def getLearnedFeatures(self,state,action):#,QTable):
        assert self.trained   

        if self.learnerType == 'ClassicNNet':
            self.neuralNetwork.activate(self.extractFeatures(state,action).reshape(-1,))        
            feats = np.round(self.neuralNetwork['in'].outputbuffer[self.neuralNetwork['in'].offset])   
        elif self.learnerType == 'Encoder':
            #print self.trainedParameters.W1.shape, self.trainedParameters.W2.shape
            feats = np.round(self.Encoder.sigmoid(np.dot(self.trainedParameters.W1, self.extractFeatures(state,action).reshape(-1,1)) + self.trainedParameters.b1),decimals=5) 
        elif self.learnerType == 'KMeans':
            assert self.centroids != None
            #gameState = self.extractFeatures(state,action)
            #key = self.dataPreprocessor.flattenStateActionTuple(state,action)
            #print QTable[key]
            #gameState = self.dataPreprocessor.normalizeData(np.array([QTable[key][0]]),train=False) if QTable[key] !=0 else np.zeros((1,1))
            #print QTable
            #stateMinMax = self.dataPreprocessor.normalizeData(np.array(QTable),train=False)
            #print stateMinMax, self.centroids
            gameState = self.extractFeatures(state,action)
            centDist = [self.gameStateDistance(gameState,self.centroids[i,:]).sum() for i in range(self.numCentroids)]  
            
            #### Soft Coding ###
            #mu = sum(centDist)/len(centDist)
            #feats = np.array([max(0,mu-dist) for dist in centDist])
            
            ##### Hard Coding ###
            feats = np.array((np.array(centDist) == np.argmin(np.array(centDist))),dtype=int)
            #feats = np.array(centDist)
        
        feats = dict(np.vstack((np.array(range(feats.shape[0])),feats.T)).T)
        #if self.learnerType == 'Kmeans':
        for key in feats.keys():
            feats[key] /= 10.
        #print feats
        return feats

   # Function: gameStateDistance
    # -------------
    # Compute the number of different boolean variables in a game state
    def gameStateDistance(self, gameStateA, gameStateB):
        return np.abs(gameStateA - gameStateB)

    def gameStateDistanceVect(self, gameStateA, gameStateB):
        func = np.vectorize(self.gameStateDistance)
        return func(gameStateA,gameStateB)


    # Function: Run K Means
    # -------------
    # Given a set of game state, and a number of features
    # self.numCentroids. This 
    # function will be called only once. It does not return
    # a value. Instead this function fills in self.centroids.
    def runKmeans(self, trainGameStates, pooling=False):
        assert not self.trained
        
        self.centroids = np.random.choice(2,(self.numCentroids, trainGameStates.shape[1]))
        for centroid in range(self.numCentroids):
            #print np.array(random.choice(zip(trainGameStates))), type(np.array(random.choice(zip(trainGameStates))))
            #print centroid
            self.centroids[centroid,:] = np.array(random.choice(zip(trainGameStates)))

        numSamples = trainGameStates.shape[0]
        tolerance = 5
        error =1e10
        hasConverged = False
        i=1
        
        self.gameStateAssignments = np.zeros(numSamples,dtype=int)
        
        while i <= self.kMeansMaxIter and hasConverged==False:    
            start_time = time.time()
            
            ###### Assignment Step ######
            #print self.gameStateDistance(trainGameStates,self.centroids[0,:]), self.gameStateAssignments.shape, self.numCentroids
            stateDistances = np.array([np.sum(self.gameStateDistance(trainGameStates,centroid),axis=1) for centroid in zip(*self.centroids.T)])
            selectedCentroids = np.argmin(stateDistances,axis=0)            
            #print stateDistances.shape, selectedCentroids
            
            ###### Update Step ######
            for centroid in range(self.numCentroids):
                if trainGameStates[selectedCentroids==centroid,:].size == 0:
                    #self.centroids[centroid,:] = np.random.choice(2,trainGameStates.shape[1])
                    self.centroids[centroid,:] = np.array(random.choice(zip(trainGameStates)))
                else:
                    self.centroids[centroid,:] = np.round(np.mean(trainGameStates[selectedCentroids==centroid,:],axis=0))
                    #print self.centroids[centroid,:]
                      
            ###### Error printing ######
            newError = sum([self.gameStateDistance(trainGameStates,self.centroids[centroid,:]).sum() for centroid in range(self.numCentroids)])
            
            if abs(newError - error) < tolerance: hasConverged = True
            error = newError
            print 'Iteration : ', i, ' | Error : ', error, ' | Computation time : ', round(time.time() - start_time), " seconds"
            i += 1
        
        'Finally, we remove the empty centroids (pooling)'
        if pooling:        
            centroidsToKeep = []
            for centroid in range(self.numCentroids):
                if trainGameStates[selectedCentroids==centroid,:].size != 0:
                    centroidsToKeep.append(centroid)
            self.centroids = self.centroids[centroidsToKeep,:]
            oldK = int(self.numCentroids)
            self.numCentroids = len(centroidsToKeep)            
            print 'Number of centroids removed : ', oldK - self.numCentroids
        



####################
# DataPreprocessor Class     
# Used in order to adequately
# transform the game states
# in order to learn features
class DataPreprocessor(object):
    def __init__(self, regularization=10e-5, copy=False):
        self.regularization = regularization
        self.maxStateValues = None
        self.catEncoder = preprocessing.OneHotEncoder(n_values=self.maxStateValues)
        self.labelEncoder = preprocessing.LabelEncoder()
        self.copy = copy
        self.readyToBinarize = False
        self.readyToNormalize = False
        self.readyToOrthogonalize = False
    
    
    ### FUNCTION flattenGameState 
    #
    # Translates a (gameState,action) into a useful tuple, later used for 
    # feature learning. This should be less manual !
    # Question : Is there an automated way to flatten the gameState instance ?
    def flattenStateActionTuple(self,state,action):
        Output = tuple()
        
        Layout = state.data.layout
        maxWidth = Layout.width
        #maxHeight = Layout.height
        allFood = Layout.food.asList()
        allCapsules = Layout.capsules
        actualCapsules = state.getCapsules()
        numGhosts = Layout.getNumGhosts()
        #print maxWidth, len(allFood), allCapsules, numGhosts, tuple(state.hasFood(x,y) for (x,y) in allFood)
        #print tuple(ghostState.scaredTimer>1 for ghostState in state.getGhostStates())
        
        # Food
        Output += tuple(int(state.hasFood(x,y)) for (x,y) in allFood)
        
        # Capsules 
        Output += tuple(int((x,y) in actualCapsules) for (x,y) in allCapsules)
        
        # VulnerableGhosts
        Output += tuple(int(ghostState.scaredTimer>1) for ghostState in state.getGhostStates())
        
        # Positions
        Output += tuple(sum([s.getPosition() for s in state.data.agentStates], ()))
        #print tuple(sum([s.getPosition() for s in state.data.agentStates], ()))
        # Action
        Output += (action,)
        
        # We finally modify the maxVariables for the oneHotEncoder
        self.maxStateValues = np.ones((len(Output),))*2
        self.maxStateValues[-1-2*(numGhosts+1):-1] = maxWidth
        self.maxStateValues[-1] = 5
        
        return Output        
    
    
    def binarizeData(self,gameStatesFlattened,train=True):
        if train:
            'Step 1 : Encode the data with One Hot Encoder'
            # -- Note that we treat the float variables and the actions (Strings) separately
            self.catEncoder = preprocessing.OneHotEncoder(n_values=self.maxStateValues)
            data = np.array(gameStatesFlattened)            
            data = np.hstack((np.array(data[:,:-1],dtype=np.float),\
                                self.labelEncoder.fit_transform(data[:,-1]).reshape(-1,1))).astype(int)
            #print np.max(data,axis=0)
            data = self.catEncoder.fit_transform(data).toarray()
            
            'Step 2 : We remove the duplicate columns '
            self.duplicates_ = [np.all(column==column[0]) for column in data.T]
            data = data[:,np.logical_not(self.duplicates_)]
            
            'Step 3 : We remove adjacent opposite columns'
            self.opposites_ = [np.all(column1==np.logical_not(column2)) for (column1,column2) in zip(data[:,:-1].T,data[:,1:].T)]
            data = data[:,:-1][:,np.logical_not(self.opposites_)]
            
            self.readyToBinarize = True
            return data
        else:
            assert self.readyToBinarize
            data = np.array(gameStatesFlattened)
            #print data
            data = np.hstack((np.array(data[:-1],dtype=np.float),\
                                self.labelEncoder.transform(np.array(data[-1]))))
            #print features_X.shape
            data = self.catEncoder.transform(data.reshape((1,-1))).toarray()
            data = data[:,np.logical_not(self.duplicates_)]
            data = data[:,:-1][:,np.logical_not(self.opposites_)]
            data = data.reshape(-1,)
            return data
    

    ## Function normalizeData
    #  
    def normalizeData(self,binarizedData,train=True):
        if train==False: assert self.readyToNormalize
        data = binarizedData.astype(float)
        #print data.shape
                
        'Step 1 : Normalize the data'
        if train:
            self.mean_ = np.mean(data,axis=0).reshape(1,-1)
            self.variance_ = np.var(data,axis=0).reshape(1,-1)
        data = (data- self.mean_) / ( np.sqrt(self.variance_) )
        #print data.shape
        '''
        if train:
            clr1 = '#2026B2'
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            print data#[:,0:2]
            ax1.plot(data[:,0], data[:,1], '.', mfc=clr1, mec=clr1)
            plt.show()
        '''
        if train: self.readyToNormalize=True
        return data
    
   
    ## Function orthogonalizeData
    # Applies ZCA whitening in order to 
    # learn features.    
    def orthogonalizeData(self,normalizedData,train=True):
        if train==False: assert self.readyToOrthogonalize
     
        'Step 2 : Whiten the data'
        if train:
            self.U_, self.s_,_ = linalg.svd(np.cov(normalizedData.T))
        'Updating our array of gameStates'
        diagMat = np.diag((self.s_ + self.regularization)**(-0.5))
        data = np.dot( normalizedData, np.dot( np.dot( self.U_, diagMat ), self.U_.T))
        '''
        if train:
            clr1 = '#2026B2'
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            print data[:,0:2]
            ax1.plot(data[:,0], data[:,1], '.', mfc=clr1, mec=clr1)
            plt.show()
        '''
        if train: self.readyToOrthogonalize=True
        return data
        
