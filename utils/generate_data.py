import pandas as pd
import numpy as np

def generate_synthetic_loan_data(n_samples=5000, output_path='data/loan_data.csv'):
    np.random.seed(42)
    age = np.random.choice(['18-25', '26-35', '36-45', '46-60', '60+'], n_samples, p=[0.1, 0.4, 0.3, 0.15, 0.05])
    gender = np.random.choice(['Male', 'Female'], n_samples, p=[0.5, 0.5])
    income = np.random.lognormal(mean=11.0, sigma=0.5, size=n_samples)
    credit_score = np.random.normal(loc=650, scale=80, size=n_samples)
    credit_score = np.clip(credit_score, 300, 850)
    loan_amount = np.random.lognormal(mean=9.5, sigma=0.8, size=n_samples)
    prob_approval = -5.0 + income / 100000 * 1.5 + credit_score / 100 * 0.8 - loan_amount / 50000 * 1.2
    prob_approval = np.where(age == '18-25', prob_approval - 1.5, prob_approval)
    prob_approval = np.where(age == '60+', prob_approval + 0.5, prob_approval)
    prob_approval = np.where(gender == 'Female', prob_approval - 1.0, prob_approval)
    prob_approval = 1 / (1 + np.exp(-prob_approval))
    loan_approval = np.random.binomial(1, prob_approval)
    income[np.random.choice(n_samples, int(n_samples * 0.05), replace=False)] = np.nan
    credit_score[np.random.choice(n_samples, int(n_samples * 0.02), replace=False)] = np.nan
    df = pd.DataFrame({'age': age, 'gender': gender, 'income': income, 'credit_score': credit_score, 'loan_amount': loan_amount, 'loan_approval': loan_approval})
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f'Dataset generated at {output_path}')
if __name__ == '__main__':
    generate_synthetic_loan_data()
