import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.preprocessing import LabelEncoder
import torch

# title of streamlit app
st.title('Amazon Fine Food Recommendations')

# load data
df = pd.read_csv('data/processed/test.csv')

# load all unique values of products and ids
productIds = set(df.ProductId)
userIDs = set(df.UserId)

# load device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ask user to input fields needed for predictions
user_id = st.selectbox('Select User Profile:', userIDs)
product_id = st.selectbox('Select Product ID:', productIds)
score = st.select_slider('Enter Review Score:',
     options=[1,2,3,4,5])
review = st.text_input('Enter Review:')

#encode product and user labels
encoder = LabelEncoder()
encoder.fit(df['ProductId'])
df['ProductId'] = encoder.transform(df['ProductId'])

encoder = LabelEncoder()
encoder.fit(df['UserId'])
df['UserId'] = encoder.transform(df['UserId'])

def predict_rating(model, userId, productId, encoder, device):
    # Encode genre
    userId = encoder.transform(np.array(userId).reshape(-1))
    productId = encoder.transform(np.array(userId).reshape(-1))

    # Get predicted rating
    model = model.to(device)
    with torch.no_grad():
        model.eval()
        X = torch.Tensor([userId,productId]).long().view(1,-1)
        X = X.to(device)
        pred = model.forward(X)
        return pred

# if submitted, make predictions
if st.button('Submit'):
    
    # load model
    model = torch.load('models/model.pt')

    pred = predict_rating(model, user_id, product_id, encoder, device)
    
    # # Store inputs into dataframe
    # X = pd.DataFrame([[height, weight, eyes]], 
    #                  columns = ["Height", "Weight", "Eyes"])
    # X = X.replace(["Brown", "Blue"], [1, 0])
    
    # # Get prediction
    # prediction = clf.predict(X)[0]
    
    # # Output prediction
    # st.text(f"This instance is a {prediction}")
    st.text('nice', pred)
