import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

def clean_data(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return
    
    df = pd.read_csv(input_path)
    
    # 1. DROP ID AND USELESS COLUMNS - This is the fix for your ID problem
    useless_cols = ['id', 'ID', 'age_desc', 'used_app_before', 'result']
    df.drop(columns=[col for col in useless_cols if col in df.columns], inplace=True)

    # 2. Handle missing values and Age
    df.replace('?', np.nan, inplace=True)
    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        df['age'] = df['age'].fillna(df['age'].median())

    # 3. Encode text to numbers
    le = LabelEncoder()
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = le.fit_transform(df[col].astype(str))

    # 4. Final conversion and save
    df = df.astype(float)
    df.to_csv(output_path, index=False)
    print(f"SUCCESS: Cleaned data saved to {output_path} (ID column removed)")

if __name__ == "__main__":
    if not os.path.exists('data'): os.makedirs('data')
    clean_data('data/autism_data.csv', 'data/cleaned_autism_data.csv')