import pandas as pd
from scipy import stats

def run_two_sample_ttest(group_a: pd.Series, group_b: pd.Series, alpha: float = 0.05) -> dict:
    """Executes a two-sample independent t-test with automatic variance checks."""
    levene_stat, levene_p = stats.levene(group_a.dropna(), group_b.dropna())
    equal_var = levene_p > alpha
    
    t_stat, p_val = stats.ttest_ind(group_a.dropna(), group_b.dropna(), equal_var=equal_var)
    return {
        "t_statistic": round(t_stat, 4),
        "p_value": round(p_val, 6),
        "equal_variance_assumed": equal_var,
        "significant": p_val < alpha
    }

def chi_square_independence(df: pd.DataFrame, col1: str, col2: str, alpha: float = 0.05) -> dict:
    """Executes Chi-Square test of independence on categorical pairs."""
    contingency_tab = pd.crosstab(df[col1], df[col2])
    chi2, p_val, dof, _ = stats.chi2_contingency(contingency_tab)
    return {
        "chi2_statistic": round(chi2, 4),
        "p_value": round(p_val, 6),
        "degrees_of_freedom": dof,
        "significant": p_val < alpha
    }
