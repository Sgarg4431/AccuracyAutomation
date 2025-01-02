import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP

st.title("Accuracy Working")

uploaded_file1 = st.file_uploader("Upload your Requirement file", type=["csv","xlsx"])
uploaded_file2 = st.file_uploader("Upload your Sales file here", type=["csv","xlsx"])
uploaded_file3 = st.file_uploader("Upload your Live days file here", type=["csv","xlsx"])


def calculate_value(row):
    M3 = row['Actual Sales']  # Actual Sales
    N3 = row['ABS']           # ABS
    L3 = row['Target Sales']  # Target Sales

    # Apply conditional logic
    if M3 == 0 and N3 == 0:
        return 0
    elif L3 == 0 and M3 > 0:
        return 1
    elif L3 == 0:
        return 0
    elif M3 == 0:
        return 1
    else:
        return N3 / M3


def calculate_value2(row):
    M3 = row['Normalised Sales']  # Actual Sales
    N3 = row['ABS2']           # ABS
    L3 = row['Target Sales']  # Target Sales

    # Apply conditional logic
    if M3 == 0 and N3 == 0:
        return 0
    elif L3 == 0 and M3 > 0:
        return 1
    elif L3 == 0:
        return 0
    elif M3 == 0:
        return 1
    else:
        return N3 / M3



def processedFile(uploaded_file1,uploaded_file2,uploaded_file3):
    if uploaded_file1 and uploaded_file2 and uploaded_file3: 
        df=pd.read_csv(uploaded_file1)
        df['Target Sales'] = np.where(
        df['Seasonality factor applicable (y/n)'] == 'n',
        df['Raw projected sales'],
        df['Projected sales with seasonality- 35 days']
        )
        df_sales=pd.read_excel(uploaded_file2)
        pivot_sales=pd.pivot_table(df_sales, index =['ean'],values=['quantity'],aggfunc='sum').reset_index()
        df_final = pd.merge(df, pivot_sales, left_on='EAN', right_on='ean', how='left',indicator=True)
        df_final.drop(columns=['ean','_merge'],inplace=True)
        df_final.rename(columns={'quantity':'Actual Sales'},inplace=True)
        df_final['Actual Sales'] = df_final['Actual Sales'].fillna(0)
        df_final['ABS']=(df_final['Actual Sales']-df_final['Target Sales']).abs()
        
       

# Apply the function row-wise
        df_final['MAPE'] = df_final.apply(calculate_value, axis=1)
        df_final['MAPE'] = df_final['MAPE'].apply(lambda x: float(Decimal(x).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)))
        df_final['Accuracy'] = df_final['MAPE'].apply(lambda x: max(1 - x, 0))
        df_live_days=pd.read_excel(uploaded_file3)
        pivot_live_days=pd.pivot_table(df_live_days, index =['style_code'],values=['Distinct Days'],aggfunc='sum').reset_index()
        df_final2 = pd.merge(df_final, pivot_live_days, left_on='EAN', right_on='style_code', how='left',indicator=True)
        df_final2.drop(columns=['style_code','_merge'],inplace=True)
        df_final2.rename(columns={'Distinct Days':'Live Days'},inplace=True)
        df_final2['Live Days'] = df_final2['Live Days'].fillna(0)
        df_final2['Normalised Sales'] = np.where((df_final2['Actual Sales'] == 0) | (df_final2['Live Days'] == 0), 0, (df_final2['Actual Sales'] / 35) * (df_final2['Live Days']))
        df_final2['Normalised Sales'] = df_final2['Normalised Sales'].fillna(0)
        df_final2['ABS2']=(df_final2['Normalised Sales']-df_final2['Target Sales']).abs()

# Apply the function row-wise
        df_final2['MAPE2'] = df_final2.apply(calculate_value2, axis=1)
        df_final2['MAPE2'] = df_final2['MAPE2'].apply(lambda x: float(Decimal(x).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)))
        df_final2['Accuracy2'] = df_final2['MAPE2'].apply(lambda x: max(1 - x, 0))




        file_path = "Accuracy.csv"
        df_final2.to_csv(file_path, index=False)
        st.write(f"All pivot tables saved in {file_path}")

# Provide a download button in Streamlit
        with open(file_path, "r") as file:
            st.download_button(
            label="Download Accuracy File",
            data=file,
            file_name="Accuracy.csv",
            mime="csv",
            )
    else:
        st.info("Please upload the valid CSV and XLSX file")



processedFile(uploaded_file1,uploaded_file2,uploaded_file3)
