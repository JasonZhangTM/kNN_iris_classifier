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

    colNames = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Class']
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
    The tool is designed for garden owner who'd like to examine the Iris flower classes.\n
    The flower class: `Iris Setosa`, `Iris Versicolour`,`Iris Virginica`
    """)
    st.image('docs/iris_classes.png', use_column_width=True)
    st.markdown("""
    The flower measurements: 
    - `Sepal Length`, `Sepal Width`
    - `Petal length`, `Petal width`
    \n
    Based on inputted measurements, it returns __`Top {}`__ similar data points for reference.\n
    
    """.format(k))

    st.markdown("__Top {} similar records: __".format(k))
    st.dataframe(irisData.iloc[nearestNeighborIndice])
    
    st.markdown("The flower is most likely to be: __{}__\n".format(classifierResult))
    if classifierResult == 'Iris-setosa':
        st.image('docs/setosa.jpg', use_column_width=True, caption='Setosa')
    if classifierResult == 'Iris-versicolor':
        st.image('docs/versicolor.jpg',use_column_width=True,caption='Versicolor')
    if classifierResult == 'Iris-virginica':
        st.image('docs/virginica.jpg', use_column_width=True,caption='Virginica')
    


    
if __name__ == '__main__': 
    try: 
        main()
    except: 
        st.error('Oops! Something went wrong...')
        raise