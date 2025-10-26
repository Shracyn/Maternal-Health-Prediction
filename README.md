Maternal Health Risk Prediction App

This project is a Streamlit web application that predicts the risk level of maternal health complications and it classifies patients into three classes: high risk, low risk and medium risk based on inputted parameters.

The model used is XGBOOST trained on maternal health dataset from Bangledesh

- Predicts maternal risk level (High,Mid or  Low)
- Interactive Streamlit web interface
- Real-time probability confidence display
- Built with a trained XGBoost model and StandardScaler
- Ready for deployment on Streamlit Cloud
   Input Features

The model uses seven key health indicators:

 Features and  Description 
BS- Blood Sugar Level in (mmol/L)
SystolicBP - Systolic Blood Pressure (mm Hg)
DiastolicBP - Diastolic Blood Pressure (mm Hg)
Age - Age (years)
MAP - Mean Arterial Pressure (mm Hg) |
HeartRate - Heart Rate (bpm)
BodyTemp - Body Temperature (°F)


maternal health app

streamlit.app.py       Streamlit application file
xgb_model.pkl          Trained XGBoost model
scaler.pkl             StandardScaler for input normalization
requirements.txt        Dependencies
README.md              Project documentation

Model Information

Algorithm - XGBoost Classifier

Scaler - StandardScaler (for feature scaling)

Number of features - 7

Output Classes:

0 → High Risk

1 → Low Risk

2 → Mid Risk

The model was trained and evaluated using accuracy, confusion matrix, and ROC-AUC score to ensure robustnes
