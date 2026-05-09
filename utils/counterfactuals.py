import dice_ml
import pandas as pd
import numpy as np

def generate_counterfactuals(model, train_df, target_col, query_instance, total_cfs=3):
    """
    Generates counterfactual explanations for a given applicant.
    
    model: The trained ML model (RandomForest or LogisticRegression)
    train_df: The training dataframe (including target)
    target_col: Name of the target column
    query_instance: pd.DataFrame with 1 row (the applicant to explain)
    """
    try:
        # 1. Define Data object
        # DiCE needs to know which features are continuous
        continuous_features = train_df.drop(columns=[target_col]).select_dtypes(include=['number']).columns.tolist()
        
        d = dice_ml.Data(dataframe=train_df, 
                         continuous_features=continuous_features, 
                         outcome_name=target_col)
        
        # 2. Define Model object
        # DiCE supports sklearn models directly
        m = dice_ml.Model(model=model, backend="sklearn")
        
        # 3. Initialize DiCE
        exp = dice_ml.Dice(d, m, method="random") # Random is faster and more robust for generic datasets
        
        # 4. Generate counterfactuals
        # We want to change the outcome from 0 (Denied) to 1 (Approved)
        dice_exp = exp.generate_counterfactuals(query_instance, 
                                                total_CFs=total_cfs, 
                                                desired_class="opposite")
        
        # Convert to dataframe for easy display
        cf_df = dice_exp.cf_examples_list[0].final_cfs_df
        return cf_df
        
    except Exception as e:
        print(f"Counterfactual generation failed: {e}")
        return None

def get_actionable_diff(query_instance, cf_df):
    """
    Compares the original instance with counterfactuals to find what changed.
    """
    if cf_df is None or len(cf_df) == 0:
        return []
    
    diffs = []
    for i in range(len(cf_df)):
        changes = []
        for col in query_instance.columns:
            orig_val = query_instance[col].iloc[0]
            cf_val = cf_df[col].iloc[i]
            
            if str(orig_val) != str(cf_val):
                changes.append({
                    'feature': col,
                    'from': orig_val,
                    'to': cf_val
                })
        diffs.append(changes)
    return diffs
