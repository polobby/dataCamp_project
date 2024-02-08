# Importation des modules nécessaires
import os
import pandas as pd
import rampwf as rw
from sklearn.model_selection import StratifiedShuffleSplit

# Titre
problem_title = "Energetic class prediction for housing in Paris"
_prediction_label_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

# Correspondance entre catégoriel et int8
int_to_cat = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
}
# Mapping categories to int
cat_to_int = {v: k for k, v in int_to_cat.items()}
_prediction_label_int = list(int_to_cat)

Predictions = rw.prediction_types.make_multiclass(label_names=_prediction_label_int)

# Implémentation du workflow
workflow = rw.workflows.Classifier()

score_types = [
    rw.score_types.ROCAUC(name='auc'),
    rw.score_types.BalancedAccuracy(name='balanced acc'),
    rw.score_types.Accuracy(name="acc", precision=3)
    # rw.score_types.NegativeLogLikelihood(name='nll'),
    ]

# Construction des données
_target_column_name = 'classe_consommation_energie'
_ignore_column_names = ['id',]  # TO DO


def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name))

    X_df = data.drop([_target_column_name] + _ignore_column_names, axis=1)

    y_array = data[_target_column_name]
    y_array = y_array.map(cat_to_int).fillna(-1).astype("int8").values
    return X_df, y_array


def get_train_data(path='.'):
    f_name = 'train.csv'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'test.csv'
    return _read_data(path, f_name)


def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=8, test_size=0.2, random_state=57)
    return cv.split(X, y)
