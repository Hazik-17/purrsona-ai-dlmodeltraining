import pandas as pd
df = pd.read_csv('output/labels.csv')
print(df['breed'].value_counts())
