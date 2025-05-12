import time
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as DecisionTreeSklearn
from sklearn.ensemble import RandomForestClassifier as RandomForestSklearn
from sklearn.ensemble import GradientBoostingClassifier as GradientBoostingSklearn
from sklearn.neural_network import MLPClassifier as MLPSklearn
from concrete.ml.sklearn import DecisionTreeClassifier as DecisionTreeConcrete
from concrete.ml.sklearn import RandomForestClassifier as RandomForestConcrete
from concrete.ml.sklearn import XGBClassifier as XGBConcrete
from concrete.ml.sklearn import NeuralNetClassifier as NeuralNetConcrete
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from joblib import parallel_backend
from torch import nn
import numpy as np


data = pd.read_csv('20240208_120000_lbl.csv')
X = data[['observation_hour', 'speed', 'rpm', 'acceleration', 'throttle_position', 'engine_temperature', 'engine_load_value', 'heart_rate',
          'current_weather', 'visibility', 'precipitation', 'accidents_onsite',	'design_speed',	'accidents_time']]
y = data[['risk_level']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
X_temp_train, X_temp_test, y_temp_train, y_temp_test = train_test_split(X_test, y_test, test_size=1, random_state=42)
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)

# Decision tree concrete
model = DecisionTreeConcrete(random_state=42, n_bits=4)
param_grid = {'max_features': ['sqrt', 'log2', None],
              'ccp_alpha': [0.0, 0.1, .01, .001],
              'max_depth': [1, 2, 5, 8, 10, 15, 20, 30, 50, None],
              'min_samples_leaf': [1, 2, 5, 8, 10],
              'min_samples_split': [2, 5, 8, 10],
              'criterion': ['gini', 'entropy', 'log_loss']}

# Random forest concrete
# model = RandomForestConcrete(random_state=42, n_bits=4, n_jobs=-1)
# param_grid = {'n_estimators': [4, 5, 10, 20, 50], 'max_features': ['sqrt', 'log2', None],
#              'ccp_alpha': [0.1, .01, .001], 'max_depth': [1, 2, 5, 8, 10, 15, 20, 30, 50, None],
#              'min_samples_leaf': [1, 2, 5, 8, 10, 20, 50, 80, 100],
#              'min_samples_split': [1, 2, 5, 8, 10, 20, 50, 80, 100], 'criterion': ['gini', 'entropy', 'log_loss']}

# Gradient boosting concrete
# model = XGBConcrete(random_state=42, n_bits=4, n_jobs=1)
# param_grid = {'n_estimators': [4, 5, 10, 20, 50], 'max_features': ['sqrt', 'log2', None],
#              'max_depth': [1, 2, 5, 8, 10, 15, 20, 30, 50, None], 'learning_rate': [0.1, 0.05, 0.01],
#              'subsample': [0.6, 0.8, 1.0], 'min_samples_leaf': [1, 2, 5, 8, 10, 20, 50, 80, 100],
#              'min_samples_split': [2, 5, 8, 10, 20, 50, 80, 100]}

le = LabelEncoder()
y_train = le.fit_transform(y_train)

# Hyperparameter tuning using GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, scoring="accuracy")

with parallel_backend('multiprocessing'):
    grid.fit(X_train, y_train)

print(f"Best parameters found: {grid.best_params_}")
print(f"Best estimator: {grid.best_estimator_}")
print(f"Best score: {grid.best_score_}")
print(model.get_params())

# Default configuration
model = DecisionTreeConcrete(random_state=42, n_bits=4)
# Optimal configuration
model = DecisionTreeConcrete(max_depth=15, criterion='gini', random_state=42, n_bits=4)
# Hyperparameter tuning
model = DecisionTreeConcrete(max_depth=15, ccp_alpha=0.0, max_features=None, criterion='gini', random_state=42, n_bits=4)

# Default configuration
# model = RandomForestConcrete(random_state=42, n_bits=4, n_jobs=1)
# Optimal configuration
# model = RandomForestConcrete(max_depth=15, criterion='entropy', random_state=42, n_estimators=10, n_bits=4, n_jobs=1)
# Hyperparameter tuning
# model = RandomForestConcrete(max_depth=20, n_estimators=10, criterion='entropy', random_state=42, n_bits=4, n_jobs=1)

# Default configuration
# model = XGBConcrete(random_state=42, n_bits=4, n_jobs=1)
# Optimal configuration
# model = XGBConcrete(learning_rate=0.1, n_estimators=10, max_depth=15, random_state=42, n_bits=4, n_jobs=1)

# Default configuration
# params = {
#    "module__n_layers": 2
# }
# Optimal configuration
# params = {
#    "module__n_layers": 3,
#    "module__activation_function": nn.ReLU,
#    "max_epochs":100,
#    "module__n_w_bits": 4,
#    "module__n_a_bits": 4
# }

# model = NeuralNetConcrete(**params)
# params = model.get_params()
# print(params)
# model = DecisionTreeSklearn(**params)

encrypted = False

# Calculating training, compiling, key generation times
if encrypted:
    print("Encrypted model")
    print("Training model...")
    print("Length of X_test:{}".format(len(X_test)))
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time
    print("Training time: {:.3f} s".format(training_time))

    print("Compiling model...")
    start_time = time.time()
    fhe_circuit = model.compile(X_train)
    end_time = time.time()
    compiling_time = end_time - start_time
    print("Compiling time: {:.3f} s".format(compiling_time))

    print("Key generation...")
    start_time = time.time()
    fhe_circuit.client.keygen(force=True)
    end_time = time.time()
    key_generation_time = end_time - start_time
    print("Key generation time: {:.3f} s".format(key_generation_time))

    print("Inferring...")
    start_time = time.time()
    # y_pred = model.predict(X_test, fhe="simulate")
    y_pred = model.predict(X_temp_test, fhe="execute")
    end_time = time.time()
    inference_time = end_time - start_time
    print("Inferring time: {:.3f} s".format(inference_time))
else:
    print("Unencrypted model")
    print("Training model...")
    print("Length of X_test:{}".format(len(X_test)))
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time
    print("Training time: {:.3f} s".format(training_time))

    print("Inferring...")
    start_time = time.time()
    y_pred = model.predict(X_test)
    end_time = time.time()
    inference_time = end_time - start_time
    print("Inferring time: {:.3f} s".format(inference_time))

print("Inferring time / sample: {:.6f} s".format(inference_time / len(y_pred)))
accuracy = accuracy_score(y_test, y_pred)
# accuracy = accuracy_score(y_temp_test, y_pred)
print("Accuracy: {:.3f}".format(accuracy))
