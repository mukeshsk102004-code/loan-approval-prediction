import numpy as np
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

def detect_bias(y_test, y_pred, sensitive_features):
    if sensitive_features is None:
        return (None, None)
    
    # Ensure all inputs are numpy arrays for consistent indexing
    import pandas as pd
    y_t = np.array(y_test)
    y_p = np.array(y_pred)
    sf = np.array(sensitive_features)
    
    unique_vals = np.unique(np.concatenate((y_t, y_p)))
    
    if not set(unique_vals).issubset({0, 1}):
        top_class = pd.Series(y_t).mode()[0]
        y_t = (y_t == top_class).astype(int)
        y_p = (y_p == top_class).astype(int)
    else:
        y_t = y_t.astype(int)
        y_p = y_p.astype(int)
        
    # Standard Fairlearn metrics
    dpd = demographic_parity_difference(y_t, y_p, sensitive_features=sf)
    eod = equalized_odds_difference(y_t, y_p, sensitive_features=sf)
    
    unique_groups = np.unique(sf)
    approval_rates = {}
    for group in unique_groups:
        mask = (sf == group)
        if np.sum(mask) > 0:
            approval_rates[str(group)] = float(np.mean(y_p[mask]))
        else:
            approval_rates[str(group)] = 0.0
            
    if len(approval_rates) > 1:
        rates = list(approval_rates.values())
        min_rate = min(rates)
        max_rate = max(rates)
        disparate_impact = min_rate / max_rate if max_rate > 0 else 0.0
    else:
        disparate_impact = 1.0
        
    metrics = {
        'Demographic Parity Difference': float(dpd),
        'Equal Opportunity Difference': float(eod),
        'Disparate Impact': float(disparate_impact),
        'Statistical Parity Ratio': float(disparate_impact) # Often used interchangeably
    }
    return (metrics, approval_rates)

def detect_intersectional_bias(y_test, y_pred, df_sensitive):
    """
    Analyzes bias across combinations of sensitive attributes.
    df_sensitive: pd.DataFrame with multiple sensitive columns
    """
    if df_sensitive is None or df_sensitive.empty:
        return (None, None)
    
    # Robustly create intersectional groups using vectorized string operations
    # This avoids 'apply' and 'join' which can hit TypeError in specific pandas/python versions
    try:
        # Convert all selected columns to string and handle NaNs
        df_str = df_sensitive.astype(str).replace(['nan', 'None', 'NaN'], 'Unknown')
        
        # Start with the first column
        intersectional_groups = df_str.iloc[:, 0]
        
        # Vectorized concatenation for all subsequent columns
        for i in range(1, df_str.shape[1]):
            intersectional_groups = intersectional_groups + " | " + df_str.iloc[:, i]
            
        return detect_bias(y_test, y_pred, intersectional_groups.values)
    except Exception as e:
        # Fallback for unexpected structural issues
        import pandas as pd
        fallback = df_sensitive.apply(lambda row: " | ".join([str(val) for val in row]), axis=1)
        return detect_bias(y_test, y_pred, fallback.values)

def classify_risk(disparate_impact):
    if disparate_impact < 0.8:
        return ('High Risk', '#DC2626')
    elif disparate_impact < 0.9:
        return ('Moderate Risk', '#F59E0B')
    else:
        return ('Fair', '#16A34A')
