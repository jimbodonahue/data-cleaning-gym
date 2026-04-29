import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import io
import contextlib
import seaborn as sns

import matplotlib.pyplot as plt

st.set_page_config(page_title="Missing Value Imputation Practice", layout="wide")

st.title("Missing Value Imputation Practice Dashboard")

# Sidebar for file upload and configuration
with st.sidebar:
    st.header("Setup")
    data_source = st.radio("Data source", ["Upload CSV", "Seaborn sample"])

    df_original = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV file", type="csv")
        if uploaded_file:
            df_original = pd.read_csv(uploaded_file)
    else:
        sample_name = st.selectbox("Select seaborn dataset", [
            "exercise",
            "car_crashes",
            "tips",
            "planets",
            "healthexp",
            "mpg",
            "penguins",
        ])
        if sample_name:
            try:
                df_original = sns.load_dataset(sample_name)
            except Exception as e:
                st.error(f"Could not load dataset {sample_name}: {e}")

    if df_original is not None:
        st.write(f"Dataset shape: {df_original.shape}")

        # AUTOMATIC variable selection
        # Ensure target_col is a float column; coerce if necessary
        float_cols = [
            c for c in df_original.select_dtypes(include=[np.number]).columns
            if np.issubdtype(df_original[c].dtype, np.floating)
        ]
        if float_cols:
            target_col = float_cols[0]
        else:
            numeric_cols = df_original.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                # coerce first numeric column to float
                target_col = numeric_cols[0]
                df_original[target_col] = df_original[target_col].astype(float)
            else:
                # try coercing first column
                target_col = df_original.columns[0]
                df_original[target_col] = pd.to_numeric(df_original[target_col], errors="coerce")
                df_original[target_col] = df_original[target_col].astype(float)

        # pick y_col automatically as another numeric column (preferably not the target)
        numeric_cols_all = df_original.select_dtypes(include=[np.number]).columns.tolist()
        y_candidates = [c for c in numeric_cols_all if c != target_col]
        y_col = y_candidates[0] if y_candidates else target_col

        st.write(f"Target column selected: {target_col} (float)")
        st.write(f"Y column selected: {y_col}")

        difficulty = st.radio("Select difficulty", ["Easy", "Medium", "Hard"])

        # Create practice dataset button
        create_practice = st.button("Create Practice Dataset")

        if create_practice:
            st.session_state.df_original = df_original
            st.session_state.df_working = df_original.copy()
            st.session_state.target_col = target_col
            st.session_state.y_col = y_col
            st.session_state.difficulty = difficulty

            # Hidden condition: pick a condition column automatically (another numeric column if possible)
            cond_candidates = [c for c in numeric_cols_all if c != target_col]
            condition_col = cond_candidates[0] if cond_candidates else target_col
            condition_value = float(df_original[condition_col].mean()) if condition_col in df_original.columns else 0.0
            st.session_state.condition_col = condition_col
            st.session_state.condition_value = condition_value

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
            # keep a copy of the initial missing-value dataset so user can revert
            st.session_state.df_initial_missing = df_work.copy()
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
                code = pandas_code or ""
                result = None

                # Capture stdout/stderr (e.g., print(), df.info()) and attempt
                # to evaluate the last expression so expressions like `df.head()` show output.
                f_out = io.StringIO()
                f_err = io.StringIO()
                captured = ""
                try:
                    with contextlib.redirect_stdout(f_out), contextlib.redirect_stderr(f_err):
                        lines = code.rstrip().splitlines()
                        if len(lines) == 0:
                            pass
                        elif len(lines) == 1:
                            try:
                                result = eval(code, exec_globals)
                            except Exception:
                                exec(code, exec_globals)
                        else:
                            body = "\n".join(lines[:-1])
                            last = lines[-1]
                            try:
                                if body.strip():
                                    exec(body, exec_globals)
                                result = eval(last, exec_globals)
                            except Exception:
                                # fallback: run full code block
                                exec(code, exec_globals)
                finally:
                    out_text = f_out.getvalue()
                    err_text = f_err.getvalue()
                    # Combine stderr after stdout so errors appear after prints
                    combined = "".join([t for t in (out_text, err_text) if t])
                    captured = combined

                # Save any changes to `df` back to session state
                st.session_state.df_working = exec_globals.get("df", df)

                # Display result if there was one
                if result is not None:
                    if isinstance(result, (pd.DataFrame, pd.Series)):
                        st.dataframe(result)
                    else:
                        try:
                            st.write(result)
                        except Exception:
                            st.text(str(result))

                # If code printed to stdout/stderr, show that output in a
                # terminal-style monospace block so `df.info()` looks like a console.
                if captured:
                    st.code(captured, language="text")

                st.success("Code executed successfully!")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        # Revert button: restore the dataset to the initial missing-values state
        if "df_initial_missing" in st.session_state:
            if st.button("Revert Missing Values"):
                st.session_state.df_working = st.session_state.df_initial_missing.copy()
                st.success("Reverted to original missing-values dataset.")

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
        # Compute slopes and show difference
        slope_orig = float(model_orig.coef_[0]) if hasattr(model_orig.coef_, '__len__') else float(model_orig.coef_)
        slope_user = float(model_user.coef_[0]) if hasattr(model_user.coef_, '__len__') else float(model_user.coef_)
        slope_delta = slope_user - slope_orig
        pct_change = (slope_delta / slope_orig * 100) if slope_orig != 0 else float('nan')

        m1, m2, m3 = st.columns(3)
        m1.metric("Expected slope", f"{slope_orig:.4f}")
        m2.metric("Your slope", f"{slope_user:.4f}")
        # show numeric difference and percent change in the third metric
        pct_display = f"{pct_change:+.1f}%" if not np.isnan(pct_change) else "n/a"
        m3.metric("Slope Δ", f"{slope_delta:+.4f}", delta=pct_display)

        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        x_range = np.linspace(df_orig_clean[x_col].min(), df_orig_clean[x_col].max(), 100).reshape(-1, 1)
        
        ax.plot(x_range, model_orig.predict(x_range), 'b-', label="Original Data", linewidth=2)
        ax.plot(x_range, model_user.predict(x_range), 'r--', label="Your Data", linewidth=2)
        
        # Scatter plot of a consistent user data subset across iterations
        sample_size = min(100, len(df_user_clean))
        # Use saved sample indices if available and still valid
        saved_idx = st.session_state.get("scatter_idx", None)
        if saved_idx is not None:
            # keep only indices that still exist in the current cleaned df
            valid_idx = [i for i in saved_idx if i in df_user_clean.index]
            if len(valid_idx) >= sample_size:
                sample = df_user_clean.loc[valid_idx].head(sample_size)
            else:
                sample = df_user_clean.sample(n=sample_size, random_state=42)
                st.session_state.scatter_idx = sample.index.tolist()
        else:
            sample = df_user_clean.sample(n=sample_size, random_state=42)
            st.session_state.scatter_idx = sample.index.tolist()

        ax.scatter(sample[x_col], sample[y_col], alpha=0.5, s=30, color="gray")

        # Annotate plot with slope difference
        ann_text = f"Δ slope = {slope_delta:+.4f}\n% change = {pct_change:.1f}%"
        ax.text(0.02, 0.98, ann_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
        
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
else:
    st.info("👈 Upload a CSV file and configure settings to start practicing!")