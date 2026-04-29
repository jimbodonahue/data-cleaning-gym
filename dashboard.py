import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import io

import matplotlib.pyplot as plt

st.set_page_config(page_title="Missing Value Imputation Practice", layout="wide")

st.title("Missing Value Imputation Practice Dashboard")

# Sidebar for file upload and configuration
with st.sidebar:
    st.header("Setup")
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    
    if uploaded_file:
        df_original = pd.read_csv(uploaded_file)
        st.write(f"Dataset shape: {df_original.shape}")
        
        target_col = st.selectbox("Select target column", df_original.columns)
        y_col = st.selectbox("Select Y-axis for regression", df_original.columns)
        difficulty = st.radio("Select difficulty", ["Easy", "Medium", "Hard"])
        
        # Medium & Hard: conditional deletion setup
        if difficulty in ["Medium", "Hard"]:
            condition_col = st.selectbox("Column for condition", 
                                        [c for c in df_original.columns if c != target_col])
            condition_value = st.number_input("Condition value (delete if greater than)", 
                                             value=float(df_original[condition_col].mean()))
        
        create_practice = st.button("Create Practice Dataset")
        
        if create_practice:
            st.session_state.df_original = df_original
            st.session_state.df_working = df_original.copy()
            st.session_state.target_col = target_col
            st.session_state.y_col = y_col
            st.session_state.difficulty = difficulty
            
            # Create missing values
            df_work = st.session_state.df_working.copy()
            mask = df_work[target_col].notna()
            indices = np.where(mask)[0]
            
            if difficulty == "Easy":
                pct_missing = np.random.uniform(0.1, 0.3)
                n_delete = int(len(indices) * pct_missing)
                delete_idx = np.random.choice(indices, n_delete, replace=False)
                
            elif difficulty == "Medium":
                condition_mask = df_work[condition_col] > condition_value
                eligible = np.where(mask & condition_mask)[0]
                pct_missing = np.random.uniform(0.1, 0.3)
                n_delete = int(len(eligible) * pct_missing)
                delete_idx = np.random.choice(eligible, min(n_delete, len(eligible)), replace=False)
                
            elif difficulty == "Hard":
                condition_mask = df_work[condition_col] > condition_value
                col_mean = df_work[target_col].mean()
                subset_mask = df_work[target_col] > col_mean
                eligible = np.where(mask & condition_mask & subset_mask)[0]
                pct_missing = np.random.uniform(0.1, 0.3)
                n_delete = int(len(eligible) * pct_missing)
                delete_idx = np.random.choice(eligible, min(n_delete, len(eligible)), replace=False)
            
            df_work.loc[delete_idx, target_col] = np.nan
            st.session_state.df_working = df_work
            st.success("Practice dataset created!")

# Main content area
if "df_working" in st.session_state:
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Data Manipulation")
        st.write(f"Missing values in {st.session_state.target_col}: {st.session_state.df_working[st.session_state.target_col].isna().sum()}")
        
        pandas_code = st.text_area("Enter Pandas code to impute missing values:", height=150)
        
        if st.button("Execute Code"):
            try:
                df = st.session_state.df_working
                exec_globals = {"df": df, "pd": pd, "np": np}
                exec(pandas_code, exec_globals)
                st.session_state.df_working = exec_globals.get("df", df)
                st.success("Code executed successfully!")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        st.dataframe(st.session_state.df_working.head(10))
    
    with col2:
        st.header("Regression Comparison")
        
        # Prepare data
        x_col = st.session_state.target_col
        y_col = st.session_state.y_col
        
        df_orig_clean = st.session_state.df_original[[x_col, y_col]].dropna()
        df_user_clean = st.session_state.df_working[[x_col, y_col]].dropna()
        
        # Fit regressions
        X_orig = df_orig_clean[x_col].values.reshape(-1, 1)
        y_orig = df_orig_clean[y_col].values
        model_orig = LinearRegression().fit(X_orig, y_orig)
        
        X_user = df_user_clean[x_col].values.reshape(-1, 1)
        y_user = df_user_clean[y_col].values
        model_user = LinearRegression().fit(X_user, y_user)
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        x_range = np.linspace(df_orig_clean[x_col].min(), df_orig_clean[x_col].max(), 100).reshape(-1, 1)
        
        ax.plot(x_range, model_orig.predict(x_range), 'b-', label="Original Data", linewidth=2)
        ax.plot(x_range, model_user.predict(x_range), 'r--', label="Your Data", linewidth=2)
        
        # Scatter plot of random user data subset
        sample_size = min(100, len(df_user_clean))
        sample = df_user_clean.sample(n=sample_size)
        ax.scatter(sample[x_col], sample[y_col], alpha=0.5, s=30, color="gray")
        
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
else:
    st.info("👈 Upload a CSV file and configure settings to start practicing!")
