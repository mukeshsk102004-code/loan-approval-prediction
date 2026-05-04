import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import streamlit as st

def load_data(path):
    return pd.read_csv(path)

@st.cache_data
def preprocess_data(df, target_col='loan_approval', sensitive_col=None):
    df = df.copy()
    df = df.dropna(subset=[target_col])
    y_raw = df[target_col]
    if isinstance(y_raw, pd.DataFrame):
        y_raw = y_raw.iloc[:, 0]
    target_was_continuous = False
    if pd.api.types.is_numeric_dtype(y_raw):
        if y_raw.nunique() > 2:
            median_val = y_raw.median()
            y = (y_raw > median_val).astype(int)
            target_was_continuous = True
        else:
            # Numeric binary, ensure it is strictly 0 and 1
            unique_vals = y_raw.dropna().unique()
            if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                y = y_raw.to_numpy()
            else:
                le_target = LabelEncoder()
                y = le_target.fit_transform(y_raw.astype(str).to_numpy())
    else:
        if y_raw.nunique() > 2:
            # Multiclass categorical: Binarize against the most frequent class
            top_class = y_raw.mode()[0]
            y = (y_raw == top_class).astype(int)
        else:
            le_target = LabelEncoder()
            y = le_target.fit_transform(y_raw.astype(str).to_numpy())
    y = np.ravel(y)
    X = df.drop(columns=[target_col]).copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    sensitive_features_raw = None
    if sensitive_col and sensitive_col in df.columns:
        sf_raw = df[sensitive_col].copy()
        if pd.api.types.is_numeric_dtype(sf_raw):
            imputer_sf = SimpleImputer(strategy='median')
        else:
            imputer_sf = SimpleImputer(strategy='most_frequent')
        sf_reshaped = sf_raw.to_numpy().reshape(-1, 1)
        sf_imputed = imputer_sf.fit_transform(sf_reshaped)
        sensitive_features_raw = pd.Series(sf_imputed.ravel(), index=sf_raw.index, name=sensitive_col)
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            if X[col].nunique() > 100:
                X = X.drop(columns=[col])
    X = X.dropna(axis=1, how='all')
    num_cols = X.select_dtypes(include=['number']).columns.tolist()
    cat_cols = X.select_dtypes(exclude=['number']).columns.tolist()
    if len(num_cols) > 0:
        num_imputer = SimpleImputer(strategy='median')
        imputed_data = num_imputer.fit_transform(X[num_cols])
        X[num_cols] = pd.DataFrame(imputed_data, columns=num_cols, index=X.index)
    if len(cat_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        imputed_cat = cat_imputer.fit_transform(X[cat_cols])
        X[cat_cols] = pd.DataFrame(imputed_cat, columns=cat_cols, index=X.index)
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str).to_numpy())
        encoders[col] = le
    sensitive_features = None
    if sensitive_col and sensitive_col in X.columns:
        sensitive_features = X[sensitive_col].copy()
    if len(num_cols) > 0:
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])
    return (X, y, sensitive_features, encoders, sensitive_features_raw)

@st.cache_data
def get_data_profile(df):
    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    numeric_df = df.select_dtypes(include=['number'])
    corr_matrix = numeric_df.corr().round(2).to_dict() if not numeric_df.empty else {}
    
    profile = {
        'row_count': len(df),
        'col_count': len(df.columns),
        'missing_counts': missing.to_dict(),
        'missing_pct': missing_pct.to_dict(),
        'total_missing': int(missing.sum()),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'numeric_cols': df.select_dtypes(include=['number']).columns.tolist(),
        'categorical_cols': df.select_dtypes(exclude=['number']).columns.tolist(),
        'correlations': corr_matrix
    }
    return profile
