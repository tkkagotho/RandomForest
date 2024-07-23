#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd  


# In[2]:


import numpy as np


# In[3]:


import pickle


# In[4]:


import streamlit as st


# In[5]:


from PIL import Image


# In[6]:


pickle_in1 = open('classifier1.pkl', 'rb') 


# In[7]:


classifier1 = pickle.load(pickle_in1)


# In[8]:


def prediction1(sepal_length1, sepal_width1, petal_length1, petal_width1):  
    prediction = classifier1.predict([[sepal_length1, sepal_width1, petal_length1, petal_width1]])  
    return prediction  
def main():  
    st.title("Iris Flower Prediction")  
  
    html_temp = """  
    <div style="background-color: #FFFF00; padding: 16px">  
    <h1 style="color: #000000; text-align: center;">Streamlit Iris Flower Classifier ML App</h1>  
    </div>  
    """  
  
    st.markdown(html_temp, unsafe_allow_html=True)  
  
    sepal_length1 = st.text_input("Sepal Length", "Type Here")  
    sepal_width1 = st.text_input("Sepal Width", "Type Here")  
    petal_length1 = st.text_input("Petal Length", "Type Here")  
    petal_width1 = st.text_input("Petal Width", "Type Here")  
    result = ""  
  
    if st.button("Predict"):  
        result = prediction1(sepal_length1, sepal_width1, petal_length1, petal_width1)  
    st.write('The output of the above is', result)  
  
if __name__ == '__main__':  
    main()  


# In[ ]:




