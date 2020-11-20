import sys

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

import model_utils

print(sys.executable)

# Grid Search: no time to do needs to do!
# https://machinelearningmastery.com/hyperparameter-optimization-with-random-search-and-grid-search/
# https://scikit-learn.org/stable/modules/grid_search.html

vehicles = None
model_utils.first_run()

# ===================================================
# Regression
train0, test0, train_target0, test_target0 = model_utils.load_dataset_file()

if any(x is None for x in [train0, test0, train_target0, test_target0]):
    if vehicles is None:
        vehicles = model_utils.load_dataset_frame()
    train0, test0, train_target0, test_target0 = model_utils.normalize_dataset_frame(vehicles)

k_fold_5 = KFold(n_splits=5, shuffle=True, random_state=2)

for name, rgs in zip(model_utils.regressor_names, model_utils.regressors):
    print("\n## Training for {} starting ****".format(name))
    r2_scores, rmse = [], []
    for train_index, test_index in k_fold_5.split(train0):
        train_x = np.array(train0[train_index][:])
        test_x = np.array(train0[test_index][:])

        train_y = train_target0[train_index][:]
        test_y = train_target0[test_index][:]

        acc_train_r2_num, acc_train_rmse_num = model_utils.regression_accuracy_model(rgs, train_x, test_x, train_y,
                                                                                     test_y)

        r2_scores.append(acc_train_r2_num)
        rmse.append(acc_train_rmse_num)

    print("\nAvg R2 Score:", round(np.mean(r2_scores), 3))
    rmse_mean = np.mean(rmse)
    print("Avg RMSE (normalized): {} & in $ value: {}".format(
        round(rmse_mean, 3), round(rmse_mean * model_utils.data_info["car_price_mean"] / 100, 3)))

    model_utils.model_pickle(name, rgs, train0, test0, train_target0, test_target0)

# ===================================================
# Classification
train0, test0, train_target0, test_target0 = model_utils.load_dataset_file('classification')

if any(x is None for x in [train0, test0, train_target0, test_target0]):
    if vehicles is None:
        vehicles = model_utils.load_dataset_frame()
    train0, test0, train_target0, test_target0 = model_utils.normalize_dataset_frame(vehicles, 'classification')

k_fold_5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)

print("# Starting 5 Fold Cross Validation for Classifiers. Number of Class:", model_utils.data_info['num_class'])
for name, clf in zip(model_utils.classifier_names, model_utils.classifiers):
    print("\n## Training for {} starting ****".format(name))
    accuracy_scores = []
    for train_index, test_index in k_fold_5.split(train0, train_target0):
        train_x = train0[train_index][:]
        test_x = train0[test_index][:]

        train_y = train_target0[train_index][:]
        test_y = train_target0[test_index][:]

        accuracy_scores.append(model_utils.classification_accuracy_model(clf, train_x, test_x, train_y, test_y))
    print("\nAvg Accuracy Score:", round(np.mean(accuracy_scores), 3))

    model_utils.model_pickle(name, clf, train0, test0, train_target0, test_target0, selector='classification',
                             balanced=True)
