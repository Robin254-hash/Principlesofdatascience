#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import seaborn as sns


# In[2]:


data = pd.read_csv("Principlesofdatascience.csv.csv")


# In[3]:


data.sample(5)


# In[4]:


data.describe()


# In[5]:


data.info()


# # Checking Data Types

# In[6]:


data.dtypes


# # Check for missing values

# In[7]:


data.isnull().any()


# In[8]:


(data.columns.tolist())


# # Distrubution of data

# In[9]:


sns.histplot(data['Age'], bins =35)
plt.title('Age Distribution')
plt.show()

sns.countplot(x ='No-show', data=data)
plt.title('No-show Distribution')
plt.show()


# In[10]:


sns.countplot(x ='SMS_received', data=data)
plt.title('SMS-sent Distribution')
plt.show()


# In[11]:


sns.countplot(x ='Hipertension', data=data)
plt.title('Health condition')
plt.show()


# # OUTLIERS

# In[12]:


sns.boxplot(x=data['Age'])
plt.title('Plot of age')
plt.show()


# In[13]:


#sns.countplot(x='No-show', data=data)
#plt.title('Target Variable Distribution')

#data.hist(figsize=(12, 8), bins=30)
#plt.tight_layout()


# # OUTLIER DETECTION

# In[14]:


from scipy.stats import zscore

# Z-score method
z_scores = np.abs(zscore(data.select_dtypes(include=np.number)))
outlier_rows_z = (z_scores > 3).any(axis=1)
print("Z-score outliers:", outlier_rows_z.sum())

# IQR method
Q1 = data.select_dtypes(include=np.number).quantile(0.25)
Q3 = data.select_dtypes(include=np.number).quantile(0.75)
IQR = Q3 - Q1
outlier_rows_iqr = ((data.select_dtypes(include=np.number) < (Q1 - 1.5 * IQR)) | 
                    (data.select_dtypes(include=np.number) > (Q3 + 1.5 * IQR))).any(axis=1)
print("IQR outliers:", outlier_rows_iqr.sum())


# In[15]:


data.rename(columns=lambda x: x.strip().replace("-", "_").replace(" ", "_"), inplace=True)
data['ScheduledDay'] = pd.to_datetime(data['ScheduledDay'])
data['AppointmentDay'] = pd.to_datetime(data['AppointmentDay'])


# # Encoding Categorical values (Converting discrete data into numericals) Dropping unnecessary columns and Feature Engineering

# In[16]:


data['No_show'] = data['No_show'].map({'No': 0, 'Yes': 1})


# In[17]:


data.drop(['PatientId', 'AppointmentID'], axis=1, inplace=True)


# In[18]:


data['WaitingDays'] = (data['AppointmentDay'] - data['ScheduledDay']).dt.days
data['AppointmentDay'] = data['AppointmentDay'].dt.day_name()


# # Python Pandas and scikit-learn pipeline

# In[19]:


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

# Drop invalid rows if any
data = data[data['Age'] >= 0]

# Define numeric and categorical columns
numeric_features = ['Age', 'WaitingDays']
categorical_features = ['Gender', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received', 'Neighbourhood'] #'No-show', 'ScheduledDay', 'AppointmentDay'

# Define transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformations
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])


# In[20]:


print(data.columns.tolist())


# # MODEL SELECTION AND TRAINING [RANDOM FOREST]

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve

# Train/test split
X = data.drop(['No_show', 'ScheduledDay','AppointmentDay'], axis=1)
y = data['No_show']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Final model pipeline
from sklearn.pipeline import make_pipeline
model_pipeline = make_pipeline(preprocessor, RandomForestClassifier(n_estimators=100, random_state=42))
model_pipeline.fit(X_train, y_train)

# Predictions
y_pred = model_pipeline.predict(X_test)
y_proba = model_pipeline.predict_proba(X_test)[:, 1]


# # Threshold-based metrics (Accuracy, Precision, Recall, F1-Score)
# # Probability Metrics (ROC-AUC)

# In[ ]:


print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_proba))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.show()


# In[ ]:


import joblib
joblib.dump(model_pipeline, "model_pipeline.joblib")


# # Model interpretability

# In[ ]:


model = model_pipeline.named_steps['randomforestclassifier']
feature_names = model_pipeline.named_steps['columntransformer'].get_feature_names_out()
importances = model.feature_importances_
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(10)
feat_imp.plot(kind='bar')
plt.title("Top 10 Feature Importances")
plt.show()


# # Bias and Error analysis

# In[ ]:


# Check if Gender impacts performance


# In[ ]:


import streamlit as st
import pandas as pd
import joblib

# Load the trained model pipeline
model = joblib.load("model_pipeline.joblib")

st.title("🩺 No-Show Appointment Predictor")

st.markdown("Fill in the patient's details to predict whether they will miss their appointment.")

with st.form("input_form"):
    Gender = st.selectbox("Gender", ["F", "M"])
    Age = st.slider("Age", 0, 115, 30)
    Scholarship = st.radio("Scholarship", [0, 1], format_func=lambda x: "Yes" if x else "No")
    Hypertension = st.radio("Hypertension", [0, 1], format_func=lambda x: "Yes" if x else "No")
    Diabetes = st.radio("Diabetes", [0, 1], format_func=lambda x: "Yes" if x else "No")
    Alcoholism = st.radio("Alcoholism", [0, 1], format_func=lambda x: "Yes" if x else "No")
    Handcap = st.selectbox("Handcap Level", [0, 1, 2, 3, 4])
    SMS_received = st.radio("SMS Received", [0, 1], format_func=lambda x: "Yes" if x else "No")
    Neighbourhood = st.text_input("Neighbourhood")
    WaitingDays = st.slider("Waiting Days", 0, 150, 10)
    AppointmentWeekday = st.selectbox("Appointment Weekday", 
                                      ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])

    submit = st.form_submit_button("Predict")

if submit:
    input_data = pd.DataFrame([{
        'Gender': Gender,
        'Age': Age,
        'Scholarship': Scholarship,
        'Hypertension': Hypertension,
        'Diabetes': Diabetes,
        'Alcoholism': Alcoholism,
        'Handcap': Handcap,
        'SMS_received': SMS_received,
        'Neighbourhood': Neighbourhood,
        'WaitingDays': WaitingDays,
        'AppointmentWeekday': AppointmentWeekday
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"❌ The patient is likely to NO-SHOW. Confidence: {probability:.2%}")
    else:
        st.success(f"✅ The patient is likely to SHOW UP. Confidence: {1 - probability:.2%}")


# In[ ]:


# get_ipython().system('streamlit --version')


# In[ ]:


# get_ipython().system('jupyter nbconvert --to script Principlesofdatascience.ipynb')


# In[ ]:


# get_ipython().system('streamlit run Principlesofdatascience.py')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




