import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

st.write("""
# Sales Forecasting app

This app Predicts **Sales** for the future data provided to it based on the previous data provided to it from which this Random Forest Regressor model is trained
                  
""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://mail.google.com/mail/u/0?ui=2&ik=ec65b0975a&attid=0.1&permmsgid=msg-a:r14194070767432967&th=18bedd89da8a4d3d&view=att&disp=safe&realattid=f_lp779kbt0)
""")
uploaded_file = st.sidebar.file_uploader("Upload your input CSV", type=["csv"])

def show_image_and_text():
    image_path = "https://mail.google.com/mail/u/0?ui=2&ik=ec65b0975a&attid=0.18&permmsgid=msg-f:1783083945579169957&th=18bec966bcf2c8a5&view=att&disp=safe&realattid=f_lp6uo9hl17"
    st.text('This is the result of our EDA')
    col1, col2 = st.columns(2)
    col1.image('https://mail.google.com/mail/u/0?ui=2&ik=ec65b0975a&attid=0.16&permmsgid=msg-f:1783083945579169957&th=18bec966bcf2c8a5&view=att&disp=safe&realattid=f_lp6uo9hj15', use_column_width=True)
    col2.image("https://mail.google.com/mail/u/0?ui=2&ik=ec65b0975a&attid=0.15&permmsgid=msg-f:1783083945579169957&th=18bec966bcf2c8a5&view=att&disp=safe&realattid=f_lp6uo9hi14", use_column_width=True)
    col1.image("https://mail.google.com/mail/u/0?ui=2&ik=ec65b0975a&attid=0.18&permmsgid=msg-f:1783083945579169957&th=18bec966bcf2c8a5&view=att&disp=safe&realattid=f_lp6uo9hl17", use_column_width=True)
    col2.image("https://mail.google.com/mail/u/0?ui=2&ik=ec65b0975a&attid=0.30&permmsgid=msg-f:1783083945579169957&th=18bec966bcf2c8a5&view=att&disp=safe&realattid=f_lp6uo9i029", use_column_width=True)
    col1.image("https://mail.google.com/mail/u/0?ui=2&ik=ec65b0975a&attid=0.28&permmsgid=msg-f:1783083945579169957&th=18bec966bcf2c8a5&view=att&disp=safe&realattid=f_lp6uo9hy27", use_column_width=True)
    col2.image("https://mail.google.com/mail/u/0?ui=2&ik=ec65b0975a&attid=0.32&permmsgid=msg-f:1783083945579169957&th=18bec966bcf2c8a5&view=att&disp=safe&realattid=f_lp6uo9i331", use_column_width=True)
    col1.image("https://mail.google.com/mail/u/0?ui=2&ik=ec65b0975a&attid=0.1&permmsgid=msg-f:1783199870581618141&th=18bf32d5a18a35dd&view=att&disp=safe&realattid=f_lp8oh9wo2", use_column_width=True)
    col2.image("https://mail.google.com/mail/u/0?ui=2&ik=ec65b0975a&attid=0.2&permmsgid=msg-f:1783199870581618141&th=18bf32d5a18a35dd&view=att&disp=safe&realattid=f_lp8oh9wn1", use_column_width=True)
    col1.image("https://mail.google.com/mail/u/0?ui=2&ik=ec65b0975a&attid=0.3&permmsgid=msg-f:1783199870581618141&th=18bf32d5a18a35dd&view=att&disp=safe&realattid=f_lp8oh9w40", use_column_width=True)
    col2.image("https://mail.google.com/mail/u/0?ui=2&ik=ec65b0975a&attid=0.1&permmsgid=msg-f:1783200157214424559&th=18bf33185e2fbdef&view=att&disp=safe&realattid=f_lp8oo8gk0", use_column_width=True)    


if st.button("Analysis of Data"):
    show_image_and_text()
    if st.button("Close"):
        st.text("Content closed.")

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    input_df = input_df.iloc[:, 1:]
    st.write(input_df)

    st.subheader('Prediction by CNN model')
    prediction_1 = pd.read_csv('CNN_output.csv')
    st.write(prediction_1)
    image_1="https://mail.google.com/mail/u/0?ui=2&ik=ec65b0975a&attid=0.1&permmsgid=msg-a:r-3373415729738618814&th=18bf305a4c25d7be&view=att&disp=safe&realattid=f_lp8myuku8"
    image_2="https://mail.google.com/mail/u/0?ui=2&ik=ec65b0975a&attid=0.3&permmsgid=msg-a:r-3373415729738618814&th=18bf305a4c25d7be&view=att&disp=safe&realattid=f_lp8myul39"
    col1, col2 = st.columns(2)
    col1.image(image_1, use_column_width=True)
    col2.image(image_2, use_column_width=True)

    st.subheader('Prediction by MLP model')
    prediction_3 = pd.read_csv('MLP_output.csv')
    st.write(prediction_3)
    col1, col2 = st.columns(2)
    image_1="https://mail.google.com/mail/u/0?ui=2&ik=ec65b0975a&attid=0.8&permmsgid=msg-a:r-3373415729738618814&th=18bf305a4c25d7be&view=att&disp=safe&realattid=f_lp8myul914"
    image_2="https://mail.google.com/mail/u/0?ui=2&ik=ec65b0975a&attid=0.7&permmsgid=msg-a:r-3373415729738618814&th=18bf305a4c25d7be&view=att&disp=safe&realattid=f_lp8myulb15"
    col1.image(image_1, use_column_width=True)
    col2.image(image_2, use_column_width=True)
   
    st.subheader('Prediction by RNN model')
    prediction_4 = pd.read_csv('RNN_output.csv')
    st.write(prediction_4)
    col1, col2 = st.columns(2)
    image_1="https://mail.google.com/mail/u/0?ui=2&ik=ec65b0975a&attid=0.6&permmsgid=msg-a:r-3373415729738618814&th=18bf305a4c25d7be&view=att&disp=safe&realattid=f_lp8myul813"
    image_2="https://mail.google.com/mail/u/0?ui=2&ik=ec65b0975a&attid=0.2&permmsgid=msg-a:r-3373415729738618814&th=18bf305a4c25d7be&view=att&disp=safe&realattid=f_lp8myul712"
    col1.image(image_1, use_column_width=True)
    col2.image(image_2, use_column_width=True)

    st.subheader('Prediction by LSTM model')
    prediction_5 = pd.read_csv('LSTM_output.csv')
    st.write(prediction_5)
    col1, col2 = st.columns(2)
    image_1="https://mail.google.com/mail/u/0?ui=2&ik=ec65b0975a&attid=0.5&permmsgid=msg-a:r-3373415729738618814&th=18bf305a4c25d7be&view=att&disp=safe&realattid=f_lp8myul410"
    image_2="https://mail.google.com/mail/u/0?ui=2&ik=ec65b0975a&attid=0.4&permmsgid=msg-a:r-3373415729738618814&th=18bf305a4c25d7be&view=att&disp=safe&realattid=f_lp8myul511"
    col1.image(image_1, use_column_width=True)
    col2.image(image_2, use_column_width=True)
else:
    pass

    
    
st.write("""
         .

########################################################################################

        
.""")

# st.sidebar.header('User Input Features for FNN')

# st.sidebar.markdown("""
# [Example CSV input file](https://testfile)
# """)

# uploaded_file_2 = st.sidebar.file_uploader("Upload your input CSV for family by MLP", type=["csv"])
# if uploaded_file_2 is not None:
#     input_df_2 = pd.read_csv(uploaded_file_2)
#     input_df_2 = input_df_2.iloc[:, 1:]
# else:
#     def user_input_features_2():
#         Date= st.sidebar.slider('Date.',1,31,15)
#         Month = st.sidebar.selectbox('Month.',('01','02','03','04','05','06','07','08','09','10','11','12'))
#         Year = st.sidebar.slider('Year.',2018,2023,2018)
#         On_Promotion= st.sidebar.selectbox('On_Promotion.',('0','211'))
#         Oil= st.sidebar.slider('Oil.',0.00,170.00,93.33)
#         Family=st.sidebar.slider('Family',1,32,15)
#         user_input_df_2 = pd.DataFrame({
#             'date': [f'{Year}-{Month.zfill(2)}-{Date:02}'],
#             'On_Promotion': [On_Promotion],
#             'Oil': [Oil],
#             'Trans': [Transactions]
#             })
#         # Convert the 'date' column to datetime
#         user_input_df['date'] = pd.to_datetime(user_input_df_2['date'])

#         # Append the user input DataFrame to the original DataFrame
#         df = df.append(user_input_df_2, ignore_index=True)

#         # Sort the DataFrame based on the 'date' column
#         df.sort_values(by='date', inplace=True)

#         return df
#     input_df_2 = user_input_features_2()

# df_fam_mlp=input_df_2


# # change the SalesForecasting-pkl module here according to other dataset of family
# load_model_2 = pickle.load(open('MLP-model-pkl','rb'))

# prediction_2 = load_model_2.predict(df_fam_mlp)


# st.subheader('df_family Dataset')
# st.write(df_fam_mlp)

# st.subheader('Prediction by MLP with family dataset')
# st.write(prediction_2)

# st.write("""
#          .

# ########################################################################################

        
# .""")

# st.sidebar.header('User Input Features for MLP')

# st.sidebar.markdown("""
# [Example CSV input file](https://testfile)
# """)

# uploaded_file_3 = st.sidebar.file_uploader("Upload your input CSV for trans with CNN", type=["csv"])
# if uploaded_file_3 is not None:
#     input_df_3 = pd.read_csv(uploaded_file_3)
#     input_df_3 = input_df_3.iloc[:, 1:]
# else:
#     def user_input_features_3():
#         Date= st.sidebar.slider('Date..',1,31,15)
#         Month = st.sidebar.selectbox('Month..',('01','02','03','04','05','06','07','08','09','10','11','12'))
#         Year = st.sidebar.slider('Year..',2018,2023,2018)
#         On_Promotion= st.sidebar.selectbox('On_Promotion..',('0','211'))
#         Oil= st.sidebar.slider('Oil..',40.00,120.00,93.33)
#         Transactions=st.sidebar.slider("Transactions..",200,8256,700)
#         user_input_df_3 = pd.DataFrame({
#             'date': [f'{Year}-{Month.zfill(2)}-{Date:02}'],
#             'On_Promotion': [On_Promotion],
#             'Oil': [Oil],
#             'Trans': [Transactions]
#             })
#         # Convert the 'date' column to datetime
#         user_input_df['date'] = pd.to_datetime(user_input_df_3['date'])

#         # Append the user input DataFrame to the original DataFrame
#         df = df.append(user_input_df_3, ignore_index=True)

#         # Sort the DataFrame based on the 'date' column
#         df.sort_values(by='date', inplace=True)

#         return df
#     input_df_3 = user_input_features_3()

# df_trans_cnn=input_df_3

# load_model_3 = pickle.load(open('CNN-model-pkl','rb'))

# prediction_3 = load_model_3.predict(df_trans_cnn)

# st.subheader('df_trans Dataset')
# st.write(df_trans_cnn)

# st.subheader('Prediction by CNN with trans dataset')
# st.write(prediction_3)
 

