import numpy as np
import pandas as pd

df = pd.read_csv('County Sheet Finalized.csv')
df.head()

# Dropping the unnamed columns
df = df.loc[:, ~df.columns.str.contains("Unnamed")]

df.head()

df.columns = df.columns.str.strip()

df.head()

df = df.rename(columns={
    '% FI ≤ SP Threshold': 'Percent FI Below SNAP',
    '% FI > SP Threshold': 'Percent FI Above SNAP',
    'Food Insecurity Rate among Black Persons (all ethnicities)': 'FI Rate Black',
    'Food Insecurity Rate among Hispanic Persons (any race)': 'FI Rate Hispanic',
    'Food Insecurity Rate among White, non-Hispanic Persons': 'FI Rate White',
    '# of Food Insecure Persons Overall': 'FI Persons Overall',
    'Overall Food Insecurity Rate': 'FI Rate Overall',
    'Weighted Annual Food Budget Shortfall': 'Annual Budget Shortfall'
})

df.head()

def clean_percent_safe(col):
    return col.astype(str).str.replace('%', '', regex=False).astype(float)

def clean_dollar(col):
    return col.replace('[\$,]', '', regex=True).astype(float)

df.head()

df['FI Rate Overall'] = clean_percent_safe(df['FI Rate Overall'])
df['FI Persons Overall'] = df['FI Persons Overall'].astype(str).str.replace(',', '', regex=True).astype(float)
df['FI Rate Black'] = clean_percent_safe(df['FI Rate Black'])
df['FI Rate Hispanic'] = clean_percent_safe(df['FI Rate Hispanic'])
df['FI Rate White'] = clean_percent_safe(df['FI Rate White'])
df['SP Threshold'] = clean_percent_safe(df['SP Threshold'])
df['Percent FI Below SNAP'] = clean_percent_safe(df['Percent FI Below SNAP'])
df['Percent FI Above SNAP'] = clean_percent_safe(df['Percent FI Above SNAP'])
df['Annual Budget Shortfall'] = clean_dollar(df['Annual Budget Shortfall'])
df['Cost per Meal'] = clean_dollar(df['Cost per Meal'])
df['Weighted Weekly $ Needed by FI'] = clean_dollar(df['Weighted Weekly $ Needed by FI'])

df.head()

df = df.dropna(subset=['Annual Budget Shortfall'])

print(df.info())
df.head(25)

