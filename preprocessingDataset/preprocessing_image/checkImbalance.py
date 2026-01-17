import pandas as pd

# This script reads the labels file and shows how many images per breed
# Input a csv file with filename and breed columns
# Output a printed list of breed counts
df = pd.read_csv('output/labels.csv')
print(df['breed'].value_counts())
