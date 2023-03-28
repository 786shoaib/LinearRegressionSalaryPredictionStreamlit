from email import header
from operator import index
from statistics import mode
from textwrap import indent
from turtle import width
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
st.set_option('deprecation.showPyplotGlobalUse', False)
from sklearn.linear_model import LinearRegression


st.title("Salary Predictor")

data = pd.read_csv("Salary_Data.csv")

# Training the model
x = np.array(data['YearsExperience']).reshape(-1,1)
lr = LinearRegression()
lr.fit(x,np.array(data['Salary']))


nav = st.sidebar.radio("Navigation",["Home","Prediction","Contribute"])
st.image("image//salary.jpg",width=800)
if nav == "Home":
    if st.checkbox("Show Data"):
        st.table(data)
    
    graph = st.selectbox("What kind of Graph ?",["Interactive","Non-Interactive"])

    val = st.slider("Filter data using year",0,20)
    data = data.loc[data['YearsExperience']>=val]
    if graph == "Non-Interactive":
        plt.figure(figsize=(10,5))
        plt.scatter(data['YearsExperience'],data['Salary'])
        plt.ylim(0)
        plt.xlabel("Years Of Experience")
        plt.ylabel("Salary")
        plt.tight_layout()
        st.pyplot()
    if graph == "Interactive":
        layout = go.Layout(
            xaxis = dict(range=[0,16]),
            yaxis = dict(range=[0,150000])
        )
        fig = go.Figure(data=go.Scatter(x=data['YearsExperience'],y=data['Salary'],mode='markers'),layout = layout)
        st.plotly_chart(fig)

if nav == "Prediction":
    st.write("Predict your Salary!!")
    val = st.number_input("Enter your year of Experience!",0.00,20.00,step= 0.25)
    val = np.array(val).reshape(1,-1)
    pred = lr.predict(val)[0]

    if st.button("Predict"):
        st.success(f"Your predicted salary is {round(pred)}")



if nav == "Contribute":
    st.header("Add extra data to the original data")
    ex = st.number_input("Enter your Experience",0.0,20.0)
    sal = st.number_input("Enter your Salary",0.00,1000000.00,step = 1000.0)

    if st.button("submit"):
        to_add =  {"YearsExperience":[ex],"Salary":[sal]}
        to_add = pd.DataFrame(to_add)
        to_add.to_csv("Salary_Data.csv",mode='a' ,header=False,index=False) 
        st.success("Submitted")
