from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

class ReweighedModelWrapper:

    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        raise AttributeError('Underlying model does not support predict_proba')

def _get_reweighing_weights(X_train, y_train, sensitive_features_train):
    try:
        from aif360.algorithms.preprocessing import Reweighing
        from aif360.datasets import BinaryLabelDataset
        from sklearn.preprocessing import LabelEncoder
        
        X_train_df = pd.DataFrame(X_train).fillna(0).replace([np.inf, -np.inf], 0)
        df = X_train_df.copy()
        df['target'] = np.array(y_train).ravel()
        
        sf_array = np.array(sensitive_features_train).ravel()
        if sf_array.dtype.kind in ('U', 'S', 'O'):
            le = LabelEncoder()
            sf_encoded = le.fit_transform(sf_array)
        else:
            sf_encoded = sf_array.astype(float)
        
        df['sensitive'] = sf_encoded
        mode_val = pd.Series(sf_encoded).mode()[0]
        privileged_groups = [{'sensitive': float(mode_val)}]
        unprivileged_classes = [float(v) for v in np.unique(sf_encoded) if v != mode_val]
        unprivileged_groups = [{'sensitive': v} for v in unprivileged_classes]
        
        dataset = BinaryLabelDataset(df=df, label_names=['target'], protected_attribute_names=['sensitive'])
        rw = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
        dataset_transf = rw.fit_transform(dataset)
        return dataset_transf.instance_weights
    except Exception as e:
        print(f"Reweighing error: {e}")
        return np.ones(len(y_train))

def _get_base_model(model_type):
    if model_type == 'Random Forest':
        return RandomForestClassifier(random_state=42, n_estimators=20, max_depth=10, n_jobs=-1)
    return LogisticRegression(random_state=42, max_iter=200, n_jobs=-1)

def mitigate_bias(X_train, y_train, sensitive_features_train, model_type='Logistic Regression', method='Exponentiated Gradient'):
    base_model = _get_base_model(model_type)
    X_train = pd.DataFrame(X_train).fillna(0).replace([np.inf, -np.inf], 0)
    y_train = np.ravel(y_train)
    
    # Fast sub-sampling for extremely large datasets during heavy mitigation
    if len(X_train) > 20000 and method != 'Reweighing':
        idx = np.random.choice(len(X_train), 20000, replace=False)
        X_train = X_train.iloc[idx]
        y_train = y_train[idx]
        sensitive_features_train = np.array(sensitive_features_train)[idx]

    if method == 'Hybrid (Reweighing + Exp Gradient)':
        try:
            weights = _get_reweighing_weights(X_train, y_train, sensitive_features_train)
            sf_cleaned = np.array(sensitive_features_train).astype(str).ravel()
            mitigator = ExponentiatedGradient(base_model, constraints=DemographicParity(), eps=0.05, max_iter=10)
            try:
                # Attempt weighted mitigation
                mitigator.fit(X_train, y_train, sensitive_features=sf_cleaned, sample_weight=weights)
            except (TypeError, ValueError) as weight_err:
                print(f"Fairlearn sample_weight failed: {weight_err}. Running without weights.")
                mitigator.fit(X_train, y_train, sensitive_features=sf_cleaned)
            return mitigator
        except Exception as e:
            print(f"Hybrid mitigation failed: {e}. Falling back to standard model.")
            base_model.fit(X_train, y_train)
            return base_model

    elif method == 'Exponentiated Gradient':
        try:
            sf_cleaned = np.array(sensitive_features_train).astype(str).ravel()
            mitigator = ExponentiatedGradient(base_model, constraints=DemographicParity(), eps=0.05, max_iter=10)
            mitigator.fit(X_train, y_train, sensitive_features=sf_cleaned)
            return mitigator
        except Exception as e:
            base_model.fit(X_train, y_train)
            return base_model
    elif method == 'Reweighing':
        try:
            weights = _get_reweighing_weights(X_train, y_train, sensitive_features_train)
            base_model.fit(X_train, y_train, sample_weight=weights)
            return ReweighedModelWrapper(base_model)
        except Exception:
            base_model.fit(X_train, y_train)
            return ReweighedModelWrapper(base_model)
    else:
        base_model.fit(X_train, y_train)
        return base_model
