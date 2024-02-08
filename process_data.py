import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv('td001_dpe-clean.csv.gz', compression='gzip', header=0, sep=',', quotechar='"')


# numéro dpe à enlever
# nom_methode dpe à enlever
# version methodes dpe à enlever
# nom methode etude thermique à enlever
# version methode etude thermique à enlever

# date visite diagnostic à garder comme datetime
# date etablissement dpe à garder comme datetime
# date arrete tarif à garder comme datetime

# commentaire à enlever
# explication méthode à enlever

# estimation ges à enlever
# classe ges à enlever
# secteur d activité à garder
# commune à garder
# arrondissement à garder
# code postal à garder et à catégoriser

# nom de rue à enlever
# numero de rue à enlever
# type de voie à enlever
# batiment à enlever
# escalier à enlever
# etage à enlever
# porte à enlever
# code postal à garder
# code insee à garder
# code insee actualisé à garder

# numero lot à enlever
# surface commerciale à garder
# partie bâtiment à enlever

# organisme certificateur à garder
# adresse organisme certificateur à enlever
# date de reception dpe à garder comme datetime






# Split the data into a training and test se
public, private = train_test_split(data, test_size=0.3, random_state=42)

train, test = train_test_split(public, test_size=0.3, random_state=42)
train.to_csv('data/public/train.csv', index=False)
test.to_csv('data/public/test.csv', index=False)

train, test = train_test_split(private, test_size=0.3, random_state=42)
train.to_csv('data/train.csv', index=False)
test.to_csv('data/test.csv', index=False)