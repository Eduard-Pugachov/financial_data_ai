# Import necessary libraries and modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib_inline as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Read financial dataset and assess
df = pd.read_csv("finance_economics_dataset.csv")
print(df.head())
print(df.info())
print(df.describe())
# NOTES
#   - 3000 non-null objects in all 24 columns
#   - 15 float64, 7 int64, 2 objects ('Date' and 'Stock Index' are Strings)
