import numpy as np
import pandas as pd
import math
import random as rd
#encode class to give each  class label a number

def encode_class(data):
    classes = []
    for i in range(len(data)):
        if data[i][-1] not in classes:
            classes.append(data[i][-1])
    for i in range(len(classes)):
        for j in range(len(data)):
            if data[j][-1] == classes[i]:
                data[j][-1] = i
    return data

#splitting to split the dataset to training and testing set

def splitting(data,ratio):
    
    train_num = int(len(data)*ratio)
    train=[]
    test=list(data)

    while(len(train)<train_num):
        index=rd.randrange(len(test))
        train.append(test.pop(index))

        return train, test
    

#grouping to get a dictionary with class label as key and list of data of that class as value

def groupclass(data):
    data_dict = {}
    for i in range(len(data)):
        if data[i][-1] not in data_dict:
            data_dict[data[i][-1]] = []
        data_dict[data[i][-1]].append(data[i])
    return data_dict

#to get the mean and standard deviation of every class 
#making a mean&std function for numbers and use that in class for convinience 

def MeanAndStdDev(numbers):
    avg = np.mean(numbers)
    stddev = np.std(numbers)
    return avg, stddev

def MeanAndStdDevForClass(data):
    info = {}
    data_dict = groupclass(data)
    for classValue, instances in data_dict.items():
        info[classValue] = [MeanAndStdDev(attribute) for attribute in zip(*instances)]
    return info

#to calculate gaussian probability
# same as before, for numbers and use it for class

def calculateGaussianProbability(x, mean, stdev):
    epsilon = 1e-10
    expo = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev + epsilon, 2))))
    return (1 / (math.sqrt(2 * math.pi) * (stdev + epsilon))) * expo

def calculateClassProbabilities(info, test):
    probabilities = {}
    for classValue, classSummaries in info.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, std_dev = classSummaries[i]
            x = test[i]
            probabilities[classValue] *= calculateGaussianProbability(x, mean, std_dev)
    return probabilities

# for prediction function, making a function to get prediction for a single data point
# and make a new function to interate for the whole list

def predict(info, test):
    probabilities = calculateClassProbabilities(info, test)
    bestLabel = max(probabilities, key=probabilities.get)
    return bestLabel

def getPredictions(info, test):
    predictions = [predict(info, instance) for instance in test]
    return predictions

#accuracy 

def accuracy_rate(test, predictions):
    correct = sum(1 for i in range(len(test)) if test[i][-1] == predictions[i])
    return (correct / float(len(test))) * 100.0


