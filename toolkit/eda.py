import pandas as pd
import numpy as np

def summarize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Computes high-level feature metrics: data types, missing counts, null ratios, and cardinality."""
    summary = pd.DataFrame({
        'dtype': df.dtypes,
        'null_count': df.isnull().sum(),
        'null_pct': (df.isnull().sum() / len(df) * 100).round(2),
        'unique_values': df.nunique()
    })
    return summary

def detect_outliers_iqr(df: pd.DataFrame, column: str, multiplier: float = 1.5) -> pd.DataFrame:
    """Identifies outlier rows using the Interquartile Range (IQR) technique."""
    q25 = df[column].quantile(0.25)
    q75 = df[column].quantile(0.75)
    iqr = q75 - q25
    lower_bound = q25 - (multiplier * iqr)
    upper_bound = q75 + (multiplier * iqr)
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

def compute_correlation_matrix(df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """Extracts pairs of highly correlated continuous features to mitigate multicollinearity."""
    corr = df.select_dtypes(include=[np.number]).corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high_corr = upper.stack().reset_index()
    high_corr.columns = ['Feature_1', 'Feature_2', 'Correlation']
    return high_corr[high_corr['Correlation'] >= threshold].sort_values(by='Correlation', ascending=False)
