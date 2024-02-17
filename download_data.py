import gzip
import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split



with gzip.open('data/raw/td001_dpe-clean.csv.gz', 'rt') as csv_file:
    csv_data = csv_file.read()
    with open('data/raw/td001_dpe-clean.csv', 'wt') as out_file:
        out_file.write(csv_data)

data = pd.read_csv('data/raw/td001_dpe-clean.csv', header=0)

# Split the data into a training and test sample
private, public = train_test_split(data, test_size=0.3, random_state=42)

if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.mkdir(OUT_DIR)

# Save the data
if os.path.exists('data'):
    shutil.rmtree('data')
os.mkdir('data')

if os.path.exists('data/public'):
    shutil.rmtree('data/public')
os.mkdir('data/public')

train, test = train_test_split(public, test_size=0.3, random_state=42)
train.to_csv('data/public/train.csv', index=False, sep=',',
             header=True, quotechar='"')
test.to_csv('data/public/test.csv', index=False, sep=',',
            header=True, quotechar='"')

train, test = train_test_split(private, test_size=0.3, random_state=42)
train.to_csv('data/train.csv', index=False, sep=',',
             header=True, quotechar='"')
test.to_csv('data/test.csv', index=False, sep=',',
            header=True, quotechar='"')
