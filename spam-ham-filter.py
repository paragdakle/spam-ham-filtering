import os
import sys
import math
import operator

"""
Homework 2 Submission.
Author: Parag Pravin Dakle.
NetID: pxd160530.
Course: Advanced Machine Learning.
"""

featureSelectionCount = 1500
numberOfArguments = 7
eta = 0.05
lamda = 1.3
iterations = 100
hamDirName = 'ham'
spamDirName = 'spam'
biasWeigthKeyword = 'b_IA_s'
classValues = {hamDirName: 1.0, spamDirName: 0.0}
docCount = {hamDirName: 0.0, spamDirName: 0.0}

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

def getStopWords(fileName):
    fileDirPath = os.getcwd() + '/' + fileName
    words = getWords(fileDirPath)
    return words

def buildVocabulary(data, skipWordsList):
    vocabulary = []
    for classType in data.iterkeys():
        for item in data[classType]:
            for word in item:
                if word not in vocabulary and word.lower() not in skipWordsList:
                    vocabulary.append(word)
    return vocabulary

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

def trainBN(data, vocabulary):
    prior = {}
    conditionalProbabilities = {}
    for classType in docCount.keys():
        conditionalProbabilities[classType] = {}
        prior[classType] = docCount[classType] / sum(docCount.values())
        classText = getClassText(data[classType])
        wordCount = {}
        for word in vocabulary:
            wordCount[word] = classText.count(word) * 1.0
        for word in vocabulary:
            conditionalProbabilities[classType][word] = (wordCount[word] + 1.0) / (sum(wordCount.values()) + len(wordCount.values()))
    return prior, conditionalProbabilities

def getFileClass(filePath, vocabulary, prior, conditionalProbabilities):
    words = getWords(filePath)
    classScore = {hamDirName: 0.0, spamDirName: 0.0}
    for classType in classScore.keys():
        classScore[classType] = math.log(prior[classType])
        for word in words:
            if word in conditionalProbabilities[classType]:
                classScore[classType] += math.log(conditionalProbabilities[classType][word])
    return classScore.keys()[classScore.values().index(max(classScore.values()))]

def testBN(testDataDir, vocabulary, prior, conditionalProbabilities):
    accuracy = {hamDirName: 0.0, spamDirName: 0.0}
    totalSize = 0.0
    for classType in accuracy.keys():
        fileDirPath = os.getcwd() + '/' + testDataDir + '/' + classType
        for fileName in os.listdir(fileDirPath):
            fileClass = getFileClass(fileDirPath + '/' + fileName, vocabulary, prior, conditionalProbabilities)
            if fileClass == classType:
                accuracy[classType] += 1.0
            totalSize += 1.0
    return (sum(accuracy.values()) / totalSize) * 100

def initializeWeights(vocabulary):
    weights = {biasWeigthKeyword: 0.0}
    for word in vocabulary:
        weights[word] = 0.0
    return weights

def getFeatures(doc):
    features = {biasWeigthKeyword: 1.0}
    for word in doc:
        features[word] = doc.count(word)
    return features

def getFileFeatures(fileDir, filename):
    features = {biasWeigthKeyword: 1.0}
    words = getWords(fileDir + '/' + filename)
    for word in words:
        features[word] = words.count(word)
    return features

def getClassWeightedSum(features, weights):
    weightedSum = 0.0
    for feature, value in features.items():
        if feature in weights:
            weightedSum += value * weights[feature]
    return weightedSum

def getClassProbability(features, weights):
    weightedSum = getClassWeightedSum(features, weights)
    try:
        exponentValue = math.exp(weightedSum) * 1.0
    except OverflowError as exp:
        return 1
    return round((exponentValue) / (1.0 + exponentValue), 5)

def trainMCAPLogisticRegression(data, vocabulary, I, E, L):
    weights = initializeWeights(vocabulary)
    for i in range(0, I):
        errorSummation = {}
        for classType in data:
            for item in data[classType]:
                features = getFeatures(item)
                classError = classValues[classType] - getClassProbability(features, weights)
                if classError != 0:
                    for feature in features.iterkeys():
                        if(errorSummation.has_key(feature)):
                            errorSummation[feature] += (features[feature] * classError)
                        else:
                            errorSummation[feature] = (features[feature] * classError)
        for weight in weights.iterkeys():
            if weight in errorSummation:
                weights[weight] = weights[weight] + (E * errorSummation[weight]) - (E * L * weights[weight])
    return weights

def testMCAPLogisticRegression(testingDataDir, weights):
    accuracy = {1: 0.0, 0: 0.0}
    hamFilePathPrefix = os.getcwd() + '/' + testingDataDir + '/' + hamDirName
    for filename in os.listdir(hamFilePathPrefix):
        features = getFileFeatures(hamFilePathPrefix, filename)
        classWeightedSum = getClassWeightedSum(features, weights)
        if(classWeightedSum >= 0):
            accuracy[1] += 1.0
        else:
            accuracy[0] += 1.0
    spamFilePathPrefix = os.getcwd() + '/' + testingDataDir + '/' + spamDirName
    for filename in os.listdir(spamFilePathPrefix):
        features = getFileFeatures(spamFilePathPrefix, filename)
        classWeightedSum = getClassWeightedSum(features, weights)
        if(classWeightedSum < 0):
            accuracy[1] += 1.0
        else:
            accuracy[0] += 1.0
    return (accuracy[1] * 100) / sum(accuracy.values())

def featureSelection(data, vocabulary):
    totalDocCount = sum(docCount.values())
    newVocabulary = {}
    for word in vocabulary:
        wordDocCount = {hamDirName: 0.0, spamDirName: 0.0}
        for classType in data:
            for doc in data[classType]:
                if word in doc:
                    wordDocCount[classType] += 1.0
        wordPresentCount = sum(wordDocCount.values())
        wordAbsentCount = totalDocCount - wordPresentCount + 1
        term1 = wordDocCount[hamDirName] / totalDocCount
        if wordPresentCount == 0:
            term2 = 1
        else:
            term2 = ((totalDocCount * wordDocCount[hamDirName]) / (wordPresentCount * docCount[hamDirName]))
            if term2 == 0:
                term2 = 1
        term3 = (docCount[hamDirName] - wordDocCount[hamDirName]) / totalDocCount
        if wordAbsentCount == 0:
            term4 = 1
        else:
            term4 = (totalDocCount * (docCount[hamDirName] - wordDocCount[hamDirName])) / (wordAbsentCount * docCount[hamDirName])
            if term4 == 0:
                term4 = 1
        term5 = wordDocCount[spamDirName] / totalDocCount
        if wordPresentCount == 0:
            term6 = 1
        else:
            term6 = (totalDocCount * wordDocCount[spamDirName]) / (wordPresentCount * docCount[spamDirName])
            if term6 == 0:
                term6 = 1
        term7 = (docCount[spamDirName] - wordDocCount[spamDirName]) / totalDocCount
        if wordAbsentCount == 0:
            term8 = 1
        else:
            term8 = (totalDocCount * (docCount[spamDirName] - wordDocCount[spamDirName])) / (wordAbsentCount * docCount[spamDirName])
            if term8 == 0:
                term8 = 1
        newVocabulary[word] = term1 * math.log(term2, 2)
        newVocabulary[word] += term3 * math.log(term4, 2)
        newVocabulary[word] += term5 * math.log(term6, 2)
        newVocabulary[word] += term7 * math.log(term8, 2)
    sortedList = sorted(newVocabulary.items(), key=operator.itemgetter(1), reverse=True)
    newVocabulary = []
    for i in range(0, featureSelectionCount):
        newVocabulary.append(sortedList[i][0])
    return newVocabulary

def main(args):
    eta = float(args[3])
    lamda = float(args[4])
    iterations = int(args[5])
    featureSelectionCount = int(args[6])
    data = loadData(args[1])
    vocabulary = buildVocabulary(data, [])
    prior, conditionalProbabilities = trainBN(data, vocabulary)
    print "Naive Bayes Accuracy with Stop Words : ", testBN(args[2], vocabulary, prior, conditionalProbabilities)
    weights = trainMCAPLogisticRegression(data, vocabulary, iterations, eta, lamda)
    print "Logistic Regression Accuracy with Stop Words : ", testMCAPLogisticRegression(args[2], weights)
    stopWords = getStopWords('stop_words.txt')
    restrictedVocabulary = buildVocabulary(data, stopWords)
    prior, conditionalProbabilities = trainBN(data, restrictedVocabulary)
    print "Naive Bayes Accuracy without Stop Words : ", testBN(args[2], restrictedVocabulary, prior, conditionalProbabilities)
    weights = trainMCAPLogisticRegression(data, restrictedVocabulary, iterations, eta, lamda)
    print "Logistic Regression Accuracy with Stop Words : ", testMCAPLogisticRegression(args[2], weights)
    reducedVocabulary = featureSelection(data, vocabulary)
    prior, conditionalProbabilities = trainBN(data, reducedVocabulary)
    print "Naive Bayes Accuracy with Feature Selection : ", testBN(args[2], reducedVocabulary, prior, conditionalProbabilities)
    weights = trainMCAPLogisticRegression(data, reducedVocabulary, iterations, eta, lamda)
    print "Logistic Regression Accuracy with Feature Selection : ", testMCAPLogisticRegression(args[2], weights)

if len(sys.argv) == numberOfArguments:
	main(sys.argv)
else:
	print "Invalid number of arguments found!"
	print "Expected:"
	print "python homework2.py <training-set-dir> <test-set-dir> <e> <l> <i> <k>"
	print "training-set-dir: The directory path containing folders titled 'ham' and 'spam' which contain the training data."
	print "test-set-dir: The directory path containing folders titled 'ham' and 'spam' which contain the testing data."
	print "e: eta value or the learning rate for Logistic Regression."
	print "l: lambda value for Logistic Regression."
	print "i: Number of iterations for Logistic Regression."
	print "k: Feature Selection desired size."
