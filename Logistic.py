import streamlit as st
import pandas as pd
import sklearn as skl

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.title("Loan Approval Prediction")

df = pd.read_csv("Loan_Dataset.csv")

st.subheader("Dataset")
st.dataframe(df)

X =df[["Income","CreditScore","LoanAmount"]]
Y =df["Approved"]

X_train, X_test, y_train, y_test= train_test_split(X, Y, test_size=0.2,random_state=42)

#-------------------------------------
#Train model
#-------------------------------------

model=LogisticRegression()
model.fit(X_train,y_train)

#-------------------------------------
#User input
#-------------------------------------

accuracy= accuracy_score(y_test, prediction)
st.sucess(f"Accuracy:{accuracy*100:.2f}%")

#-------------------------------------
#User input
#-------------------------------------

st.header("Check loan approval")

income=st.number_input(
    "Annual Income",
    min_value=20000,
    max_value=100000,
    value=50000,
    step=1000
)

credit = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=650
)

loan = st.number_input(
    "Loan Amount",
    min_value=10000,
    max_value=300000,
    value=100000,
    step=5000
)


#-------------------------------------
#Prediction
#-------------------------------------

if st.button("Predict Loan Approval"):

    result=model.predict([[income,credit,loan]])
    print(result)
    if result[0] == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Rejected")
