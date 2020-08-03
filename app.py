import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

iris = datasets.load_iris()
breast = datasets.load_breast_cancer()
st.write("""
         # IMPLEMENTACIÓN DE ALGORITMOS
""")

dataset = st.sidebar.selectbox(" Selecciona el Dataset", ("Iris","Breast Cancer"))
st.write("## Ha seleccionado el siguiente Dataset: ",dataset)
test_valor = st.sidebar.slider(" Tamaño del Valor de Test", 0.1, 0.99,0.2)

clasificador = st.sidebar.selectbox("Seleccione el algoritmo de clasificacion", ("KNN", "RANDOM FOREST", "SVM", "NAIVE BAYES","REDES NEURONALES","REGRESION LOGISTICA"))
st.write("con el siguiente clasificador: ",clasificador)

def dataset_nombre(dataset):
    if dataset == "Iris":
        data = datasets.load_iris()
        st.write(iris.target_names)
    elif dataset == "Breast Cancer":
        data = datasets.load_breast_cancer()
        st.write(breast.target_names)
    X = data.data    
    y = data.target
    return X,y
if dataset == "Iris":
    st.write("Descripción del Dataset: El conjunto de datos contiene 3 clases de 50 instancias cada una, donde cada clase se refiere a un tipo de planta de iris.")
else:
    st.write("""Descripcion del Dataset: Las características se calculan a partir de una imagen digitalizada de un aspirado con aguja fina (FNA) de una masa mamaria. Describen las características de los núcleos celulares presentes y las cuales pueden servir para detectar si un tumor es maligno o benigno.""")
st.write("Descripción del Dataset:")

if dataset =='Iris':
    st.write("""
                   "-"      | valor
             ------------ | -------------
             Instancias | *150* (Filas) x *4* (columnas) 
             clases | 3
             """)          
    st.write("""
         ## Clases: Setosa, Versicolor, Virginica
             """)         
    st.write("""
             ***
             """)  
else:             
    st.write("""
                   "-"      | valor
             ------------ | -------------
             Instancias | *569* (filas) x *30* (columnas)
             clases | 2
             """)             
    st.write("""
         ## Clases: Maligno, Benigno
             """)
    st.write("""
             ***
             """)     
             
X, y =dataset_nombre(dataset)        

st.write("""
         ***
         """)     
    
def parametros(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("Número de vecinos", 1, 15)
        params["K"] = K 
    elif clf_name == "SVM":
        C = st.sidebar.slider("Parámetro de regularización C", 0.01, 10.0)
        params["C"] = C
    elif clf_name == "NAIVE BAYES":
        G = ("G")
    elif clf_name =="REDES NEURONALES":
        R = ("R")
    elif clf_name =="REGRESION LOGISTICA":
        L = ("R")
    else:
        max_depth = st.sidebar.slider("profundidad máxima del árbol", 2, 15)
        n_estimators = st.sidebar.slider("El número de árboles", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators 
    return params
   
params = parametros(clasificador)    

def get_clasificadores(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "SVM":
        clf = SVC(C=params["C"])
    elif clf_name == "NAIVE BAYES":
        clf = GaussianNB(priors=None, var_smoothing=1e-9)
    elif clf_name =="REDES NEURONALES":
        clf = MLPClassifier(activation='identity',alpha=1e-5,hidden_layer_sizes=(5, 2), verbose=False, random_state=1)
    elif clf_name == "REGRESION LOGISTICA":
        clf = LogisticRegression()
    else:
        clf = RandomForestClassifier(n_estimators=params["n_estimators"],
                                     max_depth=params["max_depth"], random_state=1234)
    return clf


if dataset != "Iris":
    clf = get_clasificadores(clasificador, params)
    X_train, X_test,  y_train, y_test = train_test_split(X, y, test_size=test_valor, random_state=1234)
    clf.fit(X_train,y_train)
    y_predict = clf.predict(X_test)
    f1 = f1_score(y_test, y_predict,average='micro')
    acc = accuracy_score(y_test, y_predict)
    recall = recall_score(y_test, y_predict)
    confusion = confusion_matrix(y_test, y_predict)
    
    st.write(f" <h3 style='text-align: center; color: blue;'> Accuracy: {acc} </h3>",unsafe_allow_html=True)
    if st.checkbox('Mostrar Definicion de Accuracy'):
        st.write(""" <h3 style='text-align: center;'> Accuracy es la medida de rendimiento más intuitiva y es simplemente una relación de observación correctamente pronosticada con respecto a las observaciones totales. Uno puede pensar que, si tenemos un valor alto de Accuracy entonces nuestro modelo es mejor. Sí solo cuando se tiene un conjuntos de datos simétricos donde los valores de falsos positivos y falsos negativos son casi los mismos </h3> """,unsafe_allow_html=True)
    
    st.write(f" <h3 style='text-align: center; color: blue;'> Recall Scord : {recall} </h3>",unsafe_allow_html=True)
    if st.checkbox('Mostrar Definicion de Recall'):
        st.write(""" <h3 style='text-align: center;'> Recall es la relación de observaciones positivas correctamente predichas a todas las observaciones en la clase real </h3> """,unsafe_allow_html=True)
    
    st.markdown(f" <h3 style='text-align: center; color: blue;'> f1: {f1}  </h3> ",unsafe_allow_html=True)
    if st.checkbox('Mostrar Definicion de f1 Measure'):
        st.write(""" <h3 style='text-align: center;'> Puntuación F1 es el promedio ponderado de Precisión y Recuperación. Por lo tanto, esta puntuación tiene en cuenta tanto falsos positivos como falsos negativos. </h3> """,unsafe_allow_html=True)
        
        
        
    st.markdown("<h3 style='text-align: center; color: red'>Matriz de Confusion</h3>", unsafe_allow_html=True)
    st.write(confusion )
    
    
        
    if st.checkbox('Desea Visualizar el Valor de las Predicciones'):
        st.markdown(f"<h3 style='text-align: center; color: black;'> Y test: {y_test}  </h3>",unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center; color: red;'> Prediccion: {y_predict}</h3>",unsafe_allow_html=True)
        st.markdown(f" <h3 style='text-align: center; color: blue;'> Aciertos del Modelo : {y_test==y_predict} </h3>", unsafe_allow_html=True)
    
    
    
    
    
    st.write("""
             ***
             """) 
             
    pca = PCA(2)
    X_projected = pca.fit_transform(X)
    
    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]
    
    fig = plt.figure
    plt.scatter(x1,x2, c=y, alpha=0.8, cmap="viridis")
    plt.xlabel("Componente principal 1")
    plt.ylabel("Componente principal 2")
    plt.colorbar()
    
    st.pyplot()

else:
    clf = get_clasificadores(clasificador, params)
    X_train, X_test,  y_train, y_test = train_test_split(X, y, test_size=test_valor,random_state=1234)
    clf.fit(X_train,y_train)
    y_predict = clf.predict(X_test)
    f1 = f1_score(y_test, y_predict,average='micro')
    acc = accuracy_score(y_test, y_predict)
    confusion = confusion_matrix(y_test, y_predict)
    
    st.write(f" <h3 style='text-align: center; color: blue;'> Accuracy: {acc} </h3>",unsafe_allow_html=True)
    if st.checkbox('Mostrar Definicion de Accuracy'):
        st.write(""" <h3 style='text-align: center;'> Accuracy es la medida de rendimiento más intuitiva y es simplemente una relación de observación correctamente pronosticada con respecto a las observaciones totales. Uno puede pensar que, si tenemos un valor alto de Accuracy entonces nuestro modelo es mejor. Sí solo cuando se tiene un conjuntos de datos simétricos donde los valores de falsos positivos y falsos negativos son casi los mismos </h3> """,unsafe_allow_html=True)
    
    
    st.markdown(f" <h3 style='text-align: center; color: blue;'> f1: {f1}  </h3> ",unsafe_allow_html=True)
    if st.checkbox('Mostrar Definicion de f1 Measure'):
        st.write(""" <h3 style='text-align: center;'> Puntuación F1 es el promedio ponderado de Precisión y Recuperación. Por lo tanto, esta puntuación tiene en cuenta tanto falsos positivos como falsos negativos. </h3> """,unsafe_allow_html=True)
        
        
        
    st.markdown("<h3 style='text-align: center; color: red'>Matriz de Confusion</h3>", unsafe_allow_html=True)
    st.write(confusion )
    
    
        
    if st.checkbox('Desea Visualizar el Valor de las Predicciones'):
        st.markdown(f"<h3 style='text-align: center; color: black;'> Y test: {y_test}  </h3>",unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center; color: red;'> Prediccion: {y_predict}</h3>",unsafe_allow_html=True)
        st.markdown(f" <h3 style='text-align: center; color: blue;'> Aciertos del Modelo : {y_test==y_predict} </h3>", unsafe_allow_html=True)
    
    
    
    
    
    st.write("""
             ***
             """) 
             
    pca = PCA(2)
    X_projected = pca.fit_transform(X)
    
    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]
    
    fig = plt.figure
    plt.scatter(x1,x2, c=y, alpha=0.8, cmap="viridis")
    plt.xlabel("Componente principal 1")
    plt.ylabel("Componente principal 2")
    plt.colorbar()
    
    st.pyplot()




