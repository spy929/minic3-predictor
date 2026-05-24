"""
Simulated data generator for MINIC3 dual-task prediction study.

This script generates the simulated patient data used in:
"A Python-Based Interactive Web Tool for Dual-Task Prediction of Treatment Response
and Adverse Events in MINIC3 Immunotherapy: A Proof-of-Concept Study"
"""

import numpy as np
import pandas as pd

def generate_clinical_data(n=2000, seed=42):
    """
    Generate synthetic clinical data for MINIC3 immunotherapy.

    Parameters:
    -----------
    n : int
        Number of simulated patients (default: 2000)
    seed : int
        Random seed for reproducibility (default: 42)

    Returns:
    --------
    df : pandas.DataFrame
        Simulated dataset with clinical features and outcome labels.
    """
    np.random.seed(seed)

    data = {
        'Patient_ID': [f'MC3-{str(i).zfill(5)}' for i in range(1, n+1)],
        'Age': np.random.normal(62, 12, n).astype(int).clip(25, 90),
        'Gender': np.random.choice(['Male', 'Female'], n, p=[0.55, 0.45]),
        'ECOG': np.random.choice([0, 1, 2, 3], n, p=[0.2, 0.4, 0.3, 0.1]),
        'Dose': np.random.choice([0.3, 1.0, 3.0, 10.0], n, p=[0.1, 0.2, 0.4, 0.3]),
        'Prior_Therapies': np.random.choice([0, 1, 2, 3], n, p=[0.2, 0.4, 0.3, 0.1]),
        'Metastasis_Sites': np.random.poisson(2, n).clip(0, 5),
        'Liver_Mets': np.random.choice([0, 1], n, p=[0.7, 0.3]),
        'Brain_Mets': np.random.choice([0, 1], n, p=[0.85, 0.15]),
        'PDL1': np.random.choice(['Negative', 'Low', 'High'], n, p=[0.3, 0.4, 0.3]),
        'TMB': np.random.exponential(8, n).round(1).clip(0, 50),
        'NLR': np.random.normal(3, 1.5, n).round(2).clip(0.5, 15),
        'LDH': np.random.normal(200, 80, n).round(0).clip(100, 600),
        'CRP': np.random.exponential(15, n).round(1).clip(1, 150),
        'Albumin': np.random.normal(38, 5, n).round(1).clip(25, 50),
        'Tumor_Type': np.random.choice(['NSCLC', 'Melanoma', 'RCC', 'Urothelial', 'HNSCC'], n),
    }

    df = pd.DataFrame(data)

    # Response probability (dose + PDL1 + ECOG)
    response_prob = 0.2 + 0.05*(df['Dose'] > 1) + 0.1*(df['PDL1'] == 'High') - 0.05*df['ECOG']
    response_prob = response_prob.clip(0.1, 0.85)
    df['Response'] = np.random.binomial(1, response_prob)

    # Adverse event probability (dose + age + CRP)
    ae_prob = 0.3 + 0.05*(df['Dose'] > 3) + 0.008*(df['Age'] - 60).clip(0, 30) + 0.002*df['CRP']
    ae_prob = ae_prob.clip(0.15, 0.9)
    df['AE'] = np.random.binomial(1, ae_prob)

    # Progression‑free survival (simplified)
    df['PFS'] = np.where(df['Response'] == 1,
                         np.random.normal(18, 6, n),
                         np.random.normal(5, 2, n)).clip(1, 48).round(1)

    # Simplified risk group for illustration
    risk_score = df['ECOG']*2 + (df['LDH'] > 250).astype(int)*3 + (df['NLR'] > 5).astype(int)*2
    df['Risk_Group'] = pd.cut(risk_score, bins=[0, 3, 6, 10], labels=['Low', 'Medium', 'High'])

    return df

if __name__ == "__main__":
    # Generate dataset and save to CSV
    df = generate_clinical_data()
    df.to_csv('simulated_data.csv', index=False)
    print("Dataset saved as 'simulated_data.csv'")
    print(f"Total patients: {len(df)}")
    print(f"Response rate: {df['Response'].mean():.2%}")
    print(f"AE rate: {df['AE'].mean():.2%}")