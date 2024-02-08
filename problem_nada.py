#Importation des modules nécessaires
import os
import pandas as pd
import rampwf as rw
from sklearn.model_selection import StratifiedShuffleSplit

#Titre
problem_title="Energetic class prediction for housing in Paris"

 
_prediction_label_names = [A, B, C, D, E, F, G]
Predictions = rw.prediction_types.make_multiclass(label_names=_prediction_label_names)
workflow = rw.workflows.FeatureExtractorClassifier()