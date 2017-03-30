#!/usr/bin/python
import os
import sys
import math
import operator
from sklearn.neural_network import MLPClassifier
import numpy as np

"""
Author: Parag Pravin Dakle.
Course: Advanced Machine Learning.
"""

numberOfArguments = 9
eta = 0.05
iterations = 100
hamDirName = 'ham'
spamDirName = 'spam'
biasWeigthKeyword = 'b_IA_s'
classValues = {hamDirName: 1.0, spamDirName: 0.0}
docCount = {hamDirName: 0.0, spamDirName: 0.0}

"""
Helper Functions
These functions carry out basic file reading, vocabulary creation, feature generation tasks.
"""
def loadData(dataDir):
    data = {hamDirName: [], spamDirName: []}
    fileDirPath = os.getcwd() + '/' + dataDir + '/' + hamDirName
    for fileName in os.listdir(fileDirPath):
        words = getWords(fileDirPath + '/' + fileName)
        if len(words) > 0:
            data[hamDirName].append(words)
        docCount[hamDirName] += 1.0
    fileDirPath = os.getcwd() + '/' + dataDir + '/' + spamDirName
    for fileName in os.listdir(fileDirPath):
        words = getWords(fileDirPath + '/' + fileName)
        if len(words) > 0:
            data[spamDirName].append(words)
        docCount[spamDirName] += 1.0
    return data

def buildVocabulary(data, skipWordsList):
    vocabulary = []
    for classType in data.iterkeys():
        for item in data[classType]:
            for word in item:
                if word not in vocabulary and word.lower() not in skipWordsList:
                    vocabulary.append(word)
    return vocabulary

def getStopWords(fileName):
    fileDirPath = os.getcwd() + '/' + fileName
    words = getWords(fileDirPath)
    return words

def getWords(filePath):
    words = []
    try:
       with open(filePath) as f:
           words = [word for line in f for word in line.split()]
    except OSError as ex:
        print ex.message
    finally:
        return words

def getClassText(docs):
    classText = []
    for docWords in docs:
        for word in docWords:
            classText.append(word)
    return classText

def initializeWeights(vocabulary):
    weights = {biasWeigthKeyword: 0.0}
    for word in vocabulary:
        weights[word] = 0.0
    return weights

def getNNFeatures(doc, vocabulary):
    features = {biasWeigthKeyword: 1.0}
    for word in vocabulary:
        features[word] = 0.0
    for word in doc:
        features[word] = doc.count(word)
    return features.values()

def getFeatures(doc):
    features = {biasWeigthKeyword: 1.0}
    for word in doc:
        features[word] = doc.count(word)
    return features

def getNNFileFeatures(fileDir, filename, vocabulary):
    features = {biasWeigthKeyword: 1.0}
    for word in vocabulary:
        features[word] = 0.0
    words = getWords(fileDir + '/' + filename)
    for word in words:
        if word in features:
            features[word] = words.count(word)
    return features.values()

def getFileFeatures(fileDir, filename):
    features = {biasWeigthKeyword: 1.0}
    words = getWords(fileDir + '/' + filename)
    for word in words:
        features[word] = words.count(word)
    return features

def getClassWeightedSum(features, weights):
    weightedSum = 0.0
    for feature, value in features.items():
        if weights.has_key(feature):
            weightedSum += value * weights[feature]
    return weightedSum

def getCalculatedClass(features, weights):
    weightedSum = getClassWeightedSum(features, weights)
    if(weightedSum > 0):
        return classValues[hamDirName]
    return classValues[spamDirName]

"""
Perceptron training and testing functions.
"""
def trainPerceptron(data, vocabulary, I, E):
    weights = initializeWeights(vocabulary)
    for i in range(0, I):
        for classType in data:
            for item in data[classType]:
                features = getFeatures(item)
                classError = classValues[classType] - getCalculatedClass(features, weights)
                if classError != 0:
                    for feature, value in features.items():
                        if(weights.has_key(feature)):
                            weights[feature] += (E * (classError) * value)
    return weights

def testPerceptron(testingDataDir, weights):
    accuracy = {1: 0.0, 0: 0.0}
    hamFilePathPrefix = os.getcwd() + '/' + testingDataDir + '/' + hamDirName
    for filename in os.listdir(hamFilePathPrefix):
        features = getFileFeatures(hamFilePathPrefix, filename)
        classWeightedSum = getClassWeightedSum(features, weights)
        if(classWeightedSum > 0):
            accuracy[1] += 1.0
        else:
            accuracy[0] += 1.0
    hamCorrectCount = accuracy[1]
    hamFileCount = sum(accuracy.values())
    spamFilePathPrefix = os.getcwd() + '/' + testingDataDir + '/' + spamDirName
    for filename in os.listdir(spamFilePathPrefix):
        features = getFileFeatures(spamFilePathPrefix, filename)
        classWeightedSum = getClassWeightedSum(features, weights)
        if(classWeightedSum <= 0):
            accuracy[1] += 1.0
        else:
            accuracy[0] += 1.0
    return (accuracy[1] * 100) / sum(accuracy.values())

"""
Neural Network training and testing functions
"""

"""
Method generates the input in the format needed by scikit-learn.
"""
def generateNNInput(data, vocabulary):
    classVector = []
    featureInputVectorList = []
    for classType in data:
        for item in data[classType]:
            featureInputVectorList.append(getNNFeatures(item, vocabulary))
            classVector.append(classValues[classType])
    return featureInputVectorList, classVector

def trainNN(data, vocabulary, eta, iterations, hidden_units, momentum):
    featureInputVectorList, featureClassVector = generateNNInput(data, vocabulary)
    neuralNetwork = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(hidden_units, ), random_state=1, learning_rate_init=eta, momentum=momentum, max_iter=iterations)
    neuralNetwork.fit(featureInputVectorList, featureClassVector)
    return neuralNetwork

def testNN(testDataDir, neuralNetwork, vocabulary):
    accuracy = {1: 0.0, 0: 0.0}
    hamFilePathPrefix = os.getcwd() + '/' + testDataDir + '/' + hamDirName
    for filename in os.listdir(hamFilePathPrefix):
        features = getNNFileFeatures(hamFilePathPrefix, filename, vocabulary)
        features = np.array(features).reshape(1, -1)
        predictedClass = neuralNetwork.predict(features)
        if predictedClass == classValues[hamDirName]:
            accuracy[1] += 1.0
        else:
            accuracy[0] += 1.0
    spamFilePathPrefix = os.getcwd() + '/' + testDataDir + '/' + spamDirName
    for filename in os.listdir(spamFilePathPrefix):
        features = getNNFileFeatures(spamFilePathPrefix, filename, vocabulary)
        features = np.array(features).reshape(1, -1)
        predictedClass = neuralNetwork.predict(features)
        if predictedClass == classValues[spamDirName]:
            accuracy[1] += 1.0
        else:
            accuracy[0] += 1.0
    return (accuracy[1] * 100) / sum(accuracy.values())

def main(args):
    eta = float(args[3])
    iterations = int(args[4])
    data = loadData(args[1])
    vocabulary = buildVocabulary(data, [])
    stopWords = getStopWords('stop_words.txt')
    restrictedVocabulary = buildVocabulary(data, stopWords)
    weights = trainPerceptron(data, vocabulary, iterations, eta)
    print "Perceptron Accuracy : ", testPerceptron(args[2], weights)
    weights = trainPerceptron(data, restrictedVocabulary, iterations, eta)
    print "Perceptron Accuracy without Stop Words : ", testPerceptron(args[2], weights)
    eta = float(args[5])
    iterations = int(args[6])
    hidden_units = int(args[7])
    momentum = float(args[8])
    neuralNetwork = trainNN(data, vocabulary, eta, iterations, hidden_units, momentum)
    print "Neural Network Accuracy : ", testNN(args[2], neuralNetwork, vocabulary)

if len(sys.argv) == numberOfArguments:
	main(sys.argv)
else:
	print "Invalid number of arguments found!"
	print "Expected:"
	print "python ann.py <training-set-dir> <test-set-dir> <p_e> <p_i> <nn_e> <nn_i> <nn_hu> <nn_m>"
	print "training-set-dir: The directory path containing folders titled 'ham' and 'spam' which contain the training data."
	print "test-set-dir: The directory path containing folders titled 'ham' and 'spam' which contain the testing data."
	print "p_e: eta value or the learning rate for Perceptron."
	print "p_i: Number of iterations for Perceptron."
	print "nn_e: eta value or the learning rate for Neural Network."
	print "nn_i: Number of iterations for Neural Network."
	print "nn_hu: Number of hidden units for Neural Network."
	print "nn_m: Momentum for Neural Network."
