import pandas as pd
import numpy as np
import os

os.makedirs('data', exist_ok=True)

# 1. German Credit Mock
np.random.seed(42)
n = 5000
age = np.random.randint(18, 75, n)
gender = np.random.choice(['Male', 'Female'], n, p=[0.6, 0.4])
credit_amount = np.random.randint(500, 15000, n)
duration = np.random.randint(6, 72, n)
housing = np.random.choice(['Own', 'Rent', 'Free'], n, p=[0.7, 0.2, 0.1])
purpose = np.random.choice(['Car', 'Furniture', 'Education', 'Business'], n)

# Introduce some bias against Female and Older individuals for testing purposes
score = (age < 30) * 1.5 + (gender == 'Male') * 2.0 - (duration / 12) * 0.5 + (credit_amount < 5000) * 1.0 + np.random.normal(0, 1, n)
prob = 1 / (1 + np.exp(-score))
loan_approval = np.where(prob > 0.5, 1, 0)

df_german = pd.DataFrame({
    'age': age,
    'gender': gender,
    'credit_amount': credit_amount,
    'duration_months': duration,
    'housing': housing,
    'purpose': purpose,
    'loan_approval': loan_approval
})
df_german.to_csv('data/german_credit_mock.csv', index=False)

# 2. Home Credit Mock
n2 = 8000
age2 = np.random.randint(21, 65, n2)
gender2 = np.random.choice(['M', 'F'], n2, p=[0.5, 0.5])
income_total = np.random.randint(20000, 150000, n2)
credit = np.random.randint(10000, 200000, n2)
education = np.random.choice(['Secondary', 'Higher education', 'Incomplete higher'], n2, p=[0.6, 0.3, 0.1])

score2 = (income_total > 50000) * 2.0 - (credit / income_total) * 1.5 + (gender2 == 'M') * 0.8 + np.random.normal(0, 1, n2)
prob2 = 1 / (1 + np.exp(-score2))
target = np.where(prob2 > 0.5, 1, 0)

df_home = pd.DataFrame({
    'age': age2,
    'gender': gender2,
    'income_total': income_total,
    'credit_amount': credit,
    'education_type': education,
    'loan_status': target
})
df_home.to_csv('data/home_credit_mock.csv', index=False)
print("Mock datasets generated successfully.")
