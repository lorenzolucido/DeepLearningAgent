# -*- coding: utf-8 -*-
"""
Created on Fri Aug 09 11:07:08 2013

@author: LoL0
"""

from game import Agent

import sys, os
import numpy as np

from game import Directions, Actions
import random, util
import time
from math import *

from qlearningAgents import PacmanQAgent
from featureExtractors import *
from featureLearner import FeatureLearner

processingFrequency = 250# Used by both DeepLearningAgent and KMeansAgent


class DeepLearningAgent(PacmanQAgent):
  """
   The DeepLearningAgent is meant to learn in a similar way than
   humans learn. The agent uses a practice step where it tests 
   its policy and update its feature weights, followed by an assimilation
   step, where it analyzes the past games and try to discover new features.
   
   (this class is inspired by the ApproximateQAgent from the Pacman projects)
  """
  
  def __init__(self, extractor='FeatureLearner', processFreq=processingFrequency, **args):
    self.featExtractor = util.lookup(extractor, globals())()
    
    PacmanQAgent.__init__(self, **args)

    # Weight Initialization
    self.featuresWeights = util.Counter()
    self.rewardCounter = util.Counter()
    self.rewardDict = util.Counter()

    # We use the processingFrequency only if we are in 
    # the featExtractor is a FeatureLearner (= we have a deep learning agent)    
    self.processingFrequency = processingFrequency #if self.featExtractor==FeatureLearner else 10e9

  def getQValue(self, state, action):
    """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
    #miniMax = self.MiniMaxEval_withAlphaBetaPruning(state.generateSuccessor(0,action), 1, 1, 999999, -99999)
    features = self.featExtractor.getFeatures(state, action)#, miniMax)#self.rewardDict)
    #if self.featExtractor.trained:
        #flattenedState = self.featExtractor.dataPreprocessor.flattenStateActionTuple(state,action)
        #key = self.featExtractor.dataPreprocessor.binarizeData(self.featExtractor.dataPreprocessor.flattenStateActionTuple(state,action),train=False)
        #print 'FlattenedState', type(flattenedState) 
        #print 'miniMaxVal', self.rewardDict[flattenedState]
        #print 'features', features
        #print 'feature weights', self.featuresWeights
    #if self.featExtractor.trained: print sum(self.featuresCounter[feature] * features[feature] for feature in features), type(sum(self.featuresCounter[feature] * features[feature] for feature in features))
    return self.featuresWeights * features
    #return sum(self.featuresWeights[feature] * features[feature] for feature in features)
    

  def update(self, state, action, nextState, reward):
    """
       During practice step, we adjust the feature weights accordingly
       to the agent experience.
    """
    miniMax = self.MiniMaxEval_withAlphaBetaPruning(nextState, 1, 1, 999999, -99999)
    features = self.featExtractor.getFeatures(state, action)#, miniMax)
    flattenedState = self.featExtractor.dataPreprocessor.flattenStateActionTuple(state,action)
    
    #print features
    #print 'UPDATE !'
    #print 'MinimaxVal' , miniMax
    #print reward, self.getValue(nextState), self.getQValue(state, action)
    correction = (reward + self.discount * self.getValue(nextState)) - self.getQValue(state, action)
    #print 'STATE',state, 'NEXT STATE', nextState, 'ACTION', action, 'REWARD', reward, 'VALUE',self.getValue(nextState), 'QVALUE', self.getQValue(state, action)
    #print self.featuresCounter
    for feature in features.keys():
        self.featuresWeights[feature] = self.featuresWeights[feature] + self.alpha * correction * features[feature]
        #print 'FlattenedState', flattenedState 
        self.rewardDict[flattenedState] = [miniMax]#, nextState.getScore()]#, self.getValue(nextState)]
        self.rewardCounter[feature] += 1
        

  def MiniMaxEval_withAlphaBetaPruning(self,gameState, agentNumber, depth, alpha, beta):
    if (gameState.isWin() or gameState.isLose() or depth==0): return gameState.getScore()  
  
    if (agentNumber==gameState.getNumAgents()):
        agentNumber = 0
        depth -= 1
    
    legalMoves = gameState.getLegalActions(agentNumber)
    searchResult = -999999 if agentNumber==0 else 999999
    #print agentNumber, depth, gameState.getNumAgents()
    for move in legalMoves:
       if agentNumber==0: 
          searchResult = max(searchResult, self.MiniMaxEval_withAlphaBetaPruning(gameState.generateSuccessor(agentNumber, move), agentNumber+1, depth, alpha, beta))
          if searchResult >= beta: return searchResult
          alpha = max(alpha, searchResult)
       else:
          searchResult = min(searchResult, self.MiniMaxEval_withAlphaBetaPruning(gameState.generateSuccessor(agentNumber, move), agentNumber+1, depth, alpha, beta))
          if searchResult <= alpha: return searchResult
          beta = min(beta, searchResult)
    #print alpha
    return searchResult

  def final(self, state):
    "Called at the end of each game."
    # call the super-class final method
    PacmanQAgent.final(self, state)
    #print len(self.featuresCounter.keys())
    
    # Is it time to update the agent features ?
    # If so, we call the featureLearner and 
    # re-initialize the feature table
    #if self.episodesSoFar%self.processingFrequency == 0: 
    if self.episodesSoFar==self.processingFrequency:
        'Assimilation step'
        #print self.featuresWeights
        self.featExtractor.updateFeatures(self.rewardDict)
        self.featuresWeights = util.Counter()
        #self.rewardDict = util.Counter()
        self.rewardCounter= util.Counter()
    
    # did we finish training?
    if self.episodesSoFar == self.numTraining:
      # you might want to print your weights here for debugging
      "*** YOUR CODE HERE ***"
      #print len(self.featuresCounter.keys())           
      print self.featuresWeights



### FUNCTION flattenGameState 
#
# Translates a gameState into a Numpy array, later used for 
# feature learning. This should be less manual !
# Question : Is there an automated way to flatten the gameState instance ?
def flattenGameState(gameState):
    #print vars(gameState.data)
    #print gameState.data.agentStates[0].getPosition(), type(gameState), type(gameState.data)
    
    # Score
    #gameStateArray = np.array([gameState.getScore()])
    
    # Capsules
    #gameStateArray = np.hstack((gameStateArray,list(sum([s for s in gameState.getCapsules()], ()))))
    
    # Food
    #gameStateArray = np.hstack((gameStateArray,gameState.getNumFood()))
    gameStateArray = np.array([gameState.getNumFood()])
    
    # Positions
    gameStateArray = np.hstack((gameStateArray,list(sum([s.getPosition() for s in gameState.data.agentStates], ())))) 
    
    
    # isWin
    #gameStateArray = np.hstack((gameStateArray,int(gameState.isWin())))
    
    # isLose
    #gameStateArray = np.hstack((gameStateArray,int(gameState.isLose())))
    #print gameStateArray
    return tuple(gameStateArray.astype(np.float))
    



#############################
#### NOT A SMART AGENT ######
#############################
from learner import Learner

learn = True
load = True
saveFile = True
learnerFile = 'deepAgentdata.dat'

class KMeansAgent(Agent):
  """
   K-Means agent is built on the same idea that we can extract features 
   from K-means in the image recognition.
   Unfortunately, this agent cannot get better as the euclidean distance
   between states is meaningless !
  """
  
  def __init__(self):
    self.learn = learn
    self.load = load    
    self.learner = Learner()
    if self.load and os.path.isfile(learnerFile): self.learner.loadData(learnerFile)
    self.learner.newGame()

  def getAction(self, gameState):
    """
    The selected action is simply the best action computed by the 
    (not that) smart evaluationFunction.
    """
    # Before trying to find an action, we save the gameState
    #print flattenGameState(gameState)
    self.learner.addState(flattenGameState(gameState))
    
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()
    legalMoves.remove('Stop')
    # Choose one of the best actions
    scores = [self.notThatSmartEvaluationFunction(gameState, action) for action in legalMoves]
    #print scores    
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    return legalMoves[chosenIndex]

  def final(self,finalState):
      self.learner.labelizeData(finalState)
                 
      if self.learner.gamesInMemory % processingFrequency == 0:
          self.learner.processLearning()
          if saveFile: 
              print 'Saving learner file...'
              self.learner.saveData(learnerFile)
      
      #self.learner.saveData(learnerFile)
      self.learner.newGame()
      #flattenGameState(finalState)
      
  def notThatSmartEvaluationFunction(self, currGameState, pacManAction):
    nextGameState = currGameState.generatePacmanSuccessor(pacManAction)  
    if self.learner.readyToPredict == False:
        return nextGameState.getScore()
    else:
        nextGameState = currGameState.generatePacmanSuccessor(pacManAction)  
        stateFeatures = self.learner.extractFeatures(flattenGameState(nextGameState))
        predictedScore = self.learner.predictScore(stateFeatures)  
        #print predictedScore        
        return float(predictedScore)
#############################
#### NOT A SMART AGENT ######
#############################

