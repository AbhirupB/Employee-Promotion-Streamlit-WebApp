# Importing required libraries
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import shap

# Creating a title for our project
st.title("Employee Promotion Prediction")
st.write('''
### Fill in the fields to make the prediction!
''')

st.sidebar.header('Employee Data')

# Mapping for categorical variables
department_mapping = {
    'Analytics': 0,
    'Finance': 1,
    'HR': 2,
    'Legal': 3,
    'Operations': 4,
    'Procurement': 5,
    'R&D': 6,
    'Sales & Marketing': 7,
    'Technology': 8
}

education_mapping = {
    'Bachelor/UG': 0,
    'Below Secondary': 1,
    'Master/PhD/PostDoc': 2
}

award_mapping = {
    'Yes': 1,
    'No': 0
}

gender_mapping = {
    "Female": 0,
    "Male": 1
}

referred_mapping = {
    "No": 0,
    "Yes": 1
}
# Function to get user input features from the sidebar
def user_input_features():
    department_feature = st.sidebar.selectbox("Select Department", ("Sales & Marketing", "Operations",
                                                                   "Technology", "Analytics", "R&D",
                                                                   "Procurement", "Finance", "HR",
                                                                   "Legal"))
    department = department_mapping[department_feature]
    
    education_feature = st.sidebar.selectbox("Select Education", ("Master/PhD/PostDoc",
                                                                  "Bachelor/UG",
                                                                  "Below Secondary"))
    education = education_mapping[education_feature]
   
    award_won = st.sidebar.selectbox("Award Won this year", ("Yes", "No"))
    award = award_mapping[award_won]
    
    user_gender = st.sidebar.selectbox("Select Gender", ("Male", "Female"))
    user_gender_encoded = gender_mapping[user_gender]
    
    age = st.sidebar.slider("Select your Age?", 15, 70, 30)
    
    no_of_training = st.sidebar.slider("Number of trainings attended?", 1, 10, 2)
    
    avg_training_score = st.sidebar.slider("Average training score", 40, 99, 70)
    
    previous_year_rating = st.sidebar.slider("Previous Year Performance Rating", 1, 5, 4)
    
    length_of_service = st.sidebar.slider("Length of Service (in years)", 1, 13, 5)
    
    user_referred = st.sidebar.radio("Referred", ("No", "Yes"))
    user_referred_encoded = referred_mapping[user_referred]

    user_sourcing = st.sidebar.checkbox("Sourcing")
    user_sourcing_encoded = int(user_sourcing)
    

    data = {
        'department': department,
        'education': education,
        'no_of_trainings': no_of_training,
        'age': age,
        'previous_year_rating': previous_year_rating,
        'length_of_service': length_of_service,
        'awards_won?': award,
        'avg_training_score': avg_training_score,
        'gender_m' : user_gender_encoded,
        'recruitment_channel_referred' : user_referred_encoded,
        'recruitment_channel_sourcing' : user_sourcing_encoded
        
    }

    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Reading the test dataset
promotion_test = pd.read_csv('./data/test_data_final.csv')

# Concatenating user data with the test dataset
df = pd.concat([input_df, promotion_test], axis=0)

# Selecting the first row (the user's input)
df = df[:1]

# Loading the saved model
load_xgboost = pickle.load(open('empPromoModel1.pkl', 'rb'))

# Applying the model for prediction
prediction = load_xgboost.predict(input_df)
prediction_probability = load_xgboost.predict_proba(input_df)

# Displaying the prediction result
st.subheader('Prediction')
result = np.array(['You are probably not promoted!', 'You are promoted!'])
st.write(result[prediction][0])

# Displaying the prediction probability
st.subheader('Prediction Probability')
st.write('Based on the data,\nyou have {0:.2f}% chance of being promoted.'.format(prediction_probability[0][1] * 100))


explainer = shap.TreeExplainer(load_xgboost)
shap_values = explainer.shap_values(input_df)


plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, input_df, plot_type="bar")
# Get the current Matplotlib figure object
fig = plt.gcf()
# Display the Matplotlib figure using Streamlit
st.pyplot(fig, bbox_inches='tight')



# FOR deployment:
# we need the requirements.txt:
#     get it using pip install pipreqs
#     then pipreqs --encoding=utf8
# everything should be run inside the folder where the app is present.