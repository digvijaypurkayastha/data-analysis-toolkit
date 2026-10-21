# Data Analysis Toolkit

A modular Python utility library providing automated exploratory data analysis (EDA), data audit pipelines, and inferential statistical testing wrappers.

## Features
- **Exploratory Profiling:** Instant cardinality checks, null distribution metrics, and IQR-based anomaly detection.
- **Multicollinearity Checks:** Automated extraction of correlated numeric feature pairs above configurable thresholds.
- **Hypothesis Testing:** Wrappers for two-sample independent t-tests (with Levene variance verification) and Chi-Square categorical independence tests.

## Directory Structure
```text
├── toolkit/
│   ├── eda.py            # Feature profiling, IQR outlier detection, correlation audits
│   └── stats.py          # Parametric and non-parametric hypothesis testing routines
├── requirements.txt      # Library dependencies
└── README.md             # Usage documentation

## Quickstart
from toolkit.eda import summarize_dataframe, detect_outliers_iqr
from toolkit.stats import run_two_sample_ttest
import pandas as pd

df = pd.read_csv("sample_data.csv")

# Generate data audit
summary = summarize_dataframe(df)
print(summary)
