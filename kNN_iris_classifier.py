'''
The script returns the ten most similar data points in the famous Iris data based on user input.

@author: Jason Zhang

'''
import pandas as pd
import numpy as np
import operator

def classifier(instance, dataSet, labels, k):
    '''
    Input:      
                instance: vector to compare to existing dataset (1xN). In Iris case, N = 4
                dataSet: size m data set of known vectors (NxM). In Iris case, N = 4, M = 150
                labels: data set labels (1xM vector)
                k: number of neighbors to use for comparison. In our case, k = 10
                
    Output:     
                the predicted flower type based on most voted labels
                nearest K neighbors' indices

    '''
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(instance, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()     
    classCount={}
    nearestNeighborIndice = []          
    for i in range(k):
        votedlabel = labels[sortedDistIndicies[i]]
        nearestNeighborIndice.append(sortedDistIndicies[i])
        classCount[votedlabel] = classCount.get(votedlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0],nearestNeighborIndice

def file2matrix(filename):
    '''
    Input:      
                filename has the data and target/labels
                
    Output:     
                dataSet: size m data set of known vectors (NxM). In Iris case, N = 4, M = 150
                labels: data set labels (1xM vector)

    '''
    fr = open(filename)
    numberOfLines = len(fr.readlines())             #get the number of lines in the file
    returnMat = np.zeros((numberOfLines,4))         #prepare matrix to return
    classLabelVector = []                           #prepare labels return   
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split(',')
        returnMat[index,:] = listFromLine[0:4]
        classLabelVector.append(listFromLine[-1])
        index += 1
    return returnMat,classLabelVector

def file2df(filename, colNames):
    '''
    Input:      
                filename has the data and target/labels
                
    Output:     
                dataframe
    '''
    df = pd.read_csv(filename, sep=',', 
                     header = None, 
                     names=colNames)
    return df

        
def normalizer(dataSet):
    '''
    Input:      
                dataSet: size m data set of known vectors (NxM). In Iris case, N = 4, M = 150
                
    Output:     
                normalize dataSet to the scale from 0 to 1. normalizedValue = (RawValue - minVal)/(maxVal - minVal)
                ranges: (1xN) vector of feature range, (maxVal - minVal)
                minVals: (1xN) vector of feature minimum values

    '''
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m,1))
    normDataSet = normDataSet/np.tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals

def irisKNNClassifier_accuracy(k,testSize=0.3):
    irisDataMat,irisLabels = file2matrix('iris.data')       #load data setfrom file
    normMat, ranges, minVals = normalizer(irisDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*testSize)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classifier(normMat[i,:],normMat[numTestVecs:m,:],irisLabels[numTestVecs:m],k)[0]
        if (classifierResult != irisLabels[i]):
            errorCount += 1.0
            
    return 1 - errorCount/float(numTestVecs)
    

def main(k = 10):
    
    sepal_length= float(input("Sepal Length: "))
    sepal_width= float(input("Sepal Width: "))
    petal_length= float(input("Petal Length: "))
    petal_width= float(input("Petal Width: "))
    
    colNames = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Class']
    irisDataMat, irisLabels = file2matrix("iris.data")
    irisData = file2df("iris.data",colNames)
    
    normMat, ranges, minVals = normalizer(irisDataMat)
    inArr = np.array([sepal_length,sepal_width,petal_length,petal_width])
    
    classifierResult = classifier((inArr - minVals)/ranges, normMat, irisLabels, k)[0]
    nearestNeighborIndice = classifier((inArr - minVals)/ranges, normMat, irisLabels, k)[1]
    
    print("The flower is likely to be: {}".format(classifierResult))
    print("The top {} record indices are: \n{} ".format(k, irisData.iloc[nearestNeighborIndice]))
    print("The accuracy of the classifier: {}".format(irisKNNClassifier_accuracy(k,testSize=0.3)))

if __name__ == '__main__':
    main()
    
