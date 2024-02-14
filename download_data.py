import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv('td001_dpe-clean.csv.gz', compression='gzip', header=0, sep=',', quotechar='"')

# Split the data into a training and test sample
public, private = train_test_split(data, test_size=0.3, random_state=42)

train, test = train_test_split(public, test_size=0.3, random_state=42)
train.to_csv('data/public/train.csv', index=False)
test.to_csv('data/public/test.csv', index=False)

train, test = train_test_split(private, test_size=0.3, random_state=42)
train.to_csv('data/train.csv', index=False)
test.to_csv('data/test.csv', index=False)