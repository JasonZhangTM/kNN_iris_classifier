import numpy as np
import pandas as pd
import streamlit as st
from kNN_iris_classifier import classifier,file2matrix,file2df,normalizer,irisKNNClassifier_accuracy

FILENAME = 'iris.data'

# App title and description
def main():

    # Define components for the sidebar
    st.sidebar.header('Input Iris Measurements')
    sepal_length = float(st.sidebar.slider(
        label='Sepal Length',
        min_value= 0.0 ,
        max_value= 8.0,
        value= 4.0,
        step= 0.1))
    sepal_width = float(st.sidebar.slider(
        label='Sepal Width',
        min_value= 0.0 ,
        max_value= 8.0,
        value=4.0,
        step=0.1))
    petal_length = float(st.sidebar.slider(
        label='Petal Length',
        min_value= 0.0 ,
        max_value= 8.0,
        value=4.0,
        step=0.1))
    petal_width = float(st.sidebar.slider(
        label='Petal Width',
        min_value= 0.0 ,
        max_value= 8.0,
        value=2.0,
        step=0.1))

    colNames = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    irisDataMat, irisLabels = file2matrix("iris.data")
    irisData = file2df("iris.data",colNames)
    
    normMat, ranges, minVals = normalizer(irisDataMat)
    inArr = np.array([sepal_length,sepal_width,petal_length,petal_width])
    
    st.sidebar.header('Top N most similar')
    k = int(st.sidebar.slider(
        label='Top',
        min_value= 5 ,
        max_value= 15,
        value= 10,
        step= 1))    
    
    classifierResult = classifier((inArr - minVals)/ranges, normMat, irisLabels, k)[0]
    nearestNeighborIndice = classifier((inArr - minVals)/ranges, normMat, irisLabels, k)[1]
    
    st.title('Iris Flower Classifier')
    st.markdown("""
    Predict the species of an Iris flower using sepal and petal measurements.  
    Return __`Top {}`__ most similar records in iris Dataset
    """.format(k))
    
    st.markdown("The flower is most likely to be: __{}__\n".format(classifierResult))
    if classifierResult == 'Iris-setosa':
        st.image('docs/setosa.jpg', width=None)
    if classifierResult == 'Iris-versicolor':
        st.image('docs/versicolor.jpg', width=None)
    if classifierResult == 'Iris-virginica':
        st.image('docs/virginica.jpg', width=None)
    
    st.markdown("The top {} similar records: ".format(k))
    st.dataframe(irisData.iloc[nearestNeighborIndice])

    
if __name__ == '__main__': 
    try: 
        main()
    except: 
        st.error('Oops! Something went wrong...')
        raise