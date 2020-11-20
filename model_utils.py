import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor, \
    ExtraTreesRegressor, RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
# models
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, balanced_accuracy_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

import NpEncoder

data_info = {}
regressor_names = ["Linear Regression", "Linear SVR", "SVR", "SGD Regressor", "Decision Tree Regressor",
                   "Random Forest Regressor", "Extra Trees Regressor", "Bagging Regressor",
                   "Gradient Boosting Regressor", "Ada Boost Regressor"]
regressors = [
    LinearRegression(n_jobs=-1),
    LinearSVR(max_iter=10000),
    SVR(max_iter=10000),
    SGDRegressor(early_stopping=True),
    DecisionTreeRegressor(),
    RandomForestRegressor(n_jobs=-1),
    ExtraTreesRegressor(n_jobs=-1),
    BaggingRegressor(n_jobs=-1),
    GradientBoostingRegressor(),
    AdaBoostRegressor(RandomForestRegressor(max_depth=10), n_estimators=100)]

classifier_names = ["Nearest Neighbors Classifier",
                    # "Linear SVM Classifier", "RBF SVM Classifier",
                    # "Gaussian Process Classifier",
                    "Decision Tree Classifier", "Random Forest Classifier", "Extra Trees Classifier",
                    "Neural Net Classifier", "AdaBoost Classifier",
                    "Naive Bayes Classifier", "Quadratic Discriminant Analysis (QDA) Classifier"]
classifiers = [
    KNeighborsClassifier(n_jobs=-1),
    # SVC(kernel="linear", C=0.025),
    # SVC(gamma=2, C=1),
    # GaussianProcessClassifier(1.0 * RBF(1.0), n_jobs=-1),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_jobs=-1),
    ExtraTreesClassifier(n_jobs=-1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(RandomForestClassifier(max_depth=10), n_estimators=100),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]


# #####################################################
def first_run():
    global data_info
    json_file = "notebooks/tmp/data_info.json"

    if data_info is None:
        data_info = {}
    if len(data_info) == 0:
        temp_info = json_info(json_file, dump=False)

        if temp_info is not None:
            data_info = temp_info

    data_info['csv_file'] = "data/true_car_listings.csv"
    data_info['json_file'] = json_file

    # regression file
    data_info['regression'] = "notebooks/tmp/regression"

    if not os.path.exists(data_info['regression']):
        os.makedirs(data_info['regression'])
    if not os.path.isdir(data_info['regression']):
        raise Exception("Sorry, {} is not a directory!".format(data_info['regression']))

    for name in regressor_names:
        data_info[name] = os.path.join(data_info['regression'], name + ".pkl")

    # classification file
    data_info['classification'] = "notebooks/tmp/classification"

    if not os.path.exists(data_info['classification']):
        os.makedirs(data_info['classification'])
    if not os.path.isdir(data_info['classification']):
        raise Exception("Sorry, {} is not a directory!".format(data_info['classification']))

    for name in classifier_names:
        data_info[name] = os.path.join(data_info['classification'], name + ".pkl")

    data_info['num_class'] = 10

    data_info['label_encoder'] = "notebooks/tmp/label_encoder.pkl"
    data_info['scaler'] = "notebooks/tmp/scaler.pkl"

    json_info(data_info['json_file'])


def json_info(json_file, dump=True):
    if dump:
        with open(json_file, 'w') as output_file:
            json.dump(data_info, output_file, cls=NpEncoder.NpEncoder, ensure_ascii=False, indent=2)
    else:
        data = None
        if os.path.exists(json_file):
            with open(json_file) as f:
                data = json.load(f)
        return data


def update_info(key, value):
    global data_info

    data_info[key] = value
    json_info(data_info['json_file'])


# #####################################################
def acc_d(y_meas, pred_y):
    # Relative error between predicted y_pred and measured y_meas values
    return mean_absolute_error(y_meas, pred_y) * len(y_meas) / sum(abs(y_meas))


def acc_rmse(y_meas, pred_y):
    # RMSE between predicted y_pred and measured y_meas values
    return (mean_squared_error(y_meas, pred_y)) ** 0.5


def regression_accuracy(test_y, pred_y):
    acc_train_r2_num = round(r2_score(test_y, pred_y) * 100, 2)
    print('acc(r2_score) for training =', acc_train_r2_num)

    acc_train_d_num = round(acc_d(test_y, pred_y) * 100, 2)
    print('acc(relative error) for training =', acc_train_d_num)

    acc_train_rmse_num = round(acc_rmse(test_y, pred_y) * 100, 2)
    print('acc(rmse) for training =', acc_train_rmse_num)

    return acc_train_r2_num, acc_train_rmse_num


def regression_accuracy_model(rgs, train_x, test_x, train_y, test_y):
    rgs.fit(train_x, train_y)

    pred_train = rgs.predict(train_x)
    pred_y = rgs.predict(test_x)

    print("\n### training performance")
    regression_accuracy(train_y, pred_train)

    print("### Test performance")
    return regression_accuracy(test_y, pred_y)


def classification_accuracy(test_y, pred_y, balanced=True):
    acc_train_r2_num = round(accuracy_score(test_y, pred_y) * 100, 2)
    print('Accuracy score for training =', acc_train_r2_num)
    if not balanced:
        acc_train_r2_num = round(balanced_accuracy_score(test_y, pred_y) * 100, 2)
        print('Balanced accuracy for training =', acc_train_r2_num)
    return acc_train_r2_num


def classification_accuracy_model(clf, train_x, test_x, train_y, test_y, balanced=True):
    clf.fit(train_x, train_y)

    pred_train = clf.predict(train_x)
    pred_y = clf.predict(test_x)

    print("\n### training performance")
    classification_accuracy(train_y, pred_train, balanced)

    print("### Test performance")
    return classification_accuracy(test_y, pred_y, balanced)


def handle_pickle(file_location, model=None, dump=True):
    if dump:
        pickle.dump(model, open(file_location, 'wb'))
    else:
        if os.path.exists(file_location):
            return pickle.load(open(file_location, 'rb'))
        else:
            return None


def model_pickle(model_name, model, train_x, test_x, train_y, test_y, selector='regression', balanced=True):
    model.fit(train_x, train_y)
    pred_y = model.predict(test_x)

    print("\n### Test performance before pickling")
    if selector == 'regression':
        regression_accuracy(test_y, pred_y)
    else:
        classification_accuracy(test_y, pred_y, balanced)
    handle_pickle(data_info[model_name], model, dump=True)

    loaded_model = handle_pickle(data_info[model_name], model=None, dump=False)
    pred_y = loaded_model.predict(test_x)

    print("### Test performance after pickling")
    if selector == 'regression':
        regression_accuracy(test_y, pred_y)
    else:
        classification_accuracy(test_y, pred_y, balanced)


def save_dataset_file(train0, test0, train_target0, test_target0, selector='regression'):
    # https://stackoverflow.com/a/3685339/2049763
    np.savetxt(os.path.join(data_info[selector], 'train0.txt'), train0)
    np.savetxt(os.path.join(data_info[selector], 'test0.txt'), test0)

    np.savetxt(os.path.join(data_info[selector], 'train_target0.txt'), train_target0)
    np.savetxt(os.path.join(data_info[selector], 'test_target0.txt'), test_target0)


def load_dataset_file(selector='regression'):
    train0, test0, train_target0, test_target0 = None, None, None, None

    if os.path.exists(os.path.join(data_info[selector], 'train0.txt')):
        train0 = np.loadtxt(os.path.join(data_info[selector], 'train0.txt'))

    if os.path.exists(os.path.join(data_info[selector], 'train0.txt')):
        test0 = np.loadtxt(os.path.join(data_info[selector], 'test0.txt'))

    if os.path.exists(os.path.join(data_info[selector], 'train0.txt')):
        train_target0 = np.loadtxt(os.path.join(data_info[selector], 'train_target0.txt'))

    if os.path.exists(os.path.join(data_info[selector], 'train0.txt')):
        test_target0 = np.loadtxt(os.path.join(data_info[selector], 'test_target0.txt'))

    return train0, test0, train_target0, test_target0


def load_dataset_frame():
    vehicles = pd.read_csv(data_info['csv_file'])

    vehicles = vehicles.loc[(vehicles.Year >= 1970) & (vehicles.Price >= 1000) & (
            vehicles.Price <= 50000) & (vehicles.Mileage <= 300000)]
    vehicles = vehicles.drop(['Vin'], axis=1)

    vehicles = vehicles.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    for col in ['City', 'State', 'Make', 'Model']:
        vehicles[col] = vehicles[col].apply(lambda x: x.lower())

    # lets drop null rows
    vehicles = vehicles.dropna()

    columns = list(vehicles.columns)
    columns.remove('Price')
    update_info('columns', columns)

    return vehicles


def normalize_dataset_frame(vehicles, selector='regression', balanced=True):
    """

    :type vehicles: object pandas dataframe
    :param selector: str 'regression' or 'classification'
    :param balanced: bool balanced class true or false

    :returns trainX, testX, trainY, test_Y
    """

    if selector == 'classification':
        if balanced:
            vehicles['Price'], price_bins = pd.qcut(vehicles.Price, q=data_info['num_class'], labels=False,
                                                    retbins=True)
            update_info('price_bins', price_bins)
        else:
            vehicles['Price'] = pd.cut(vehicles.Price, bins=data_info['num_class'], labels=False, right=False)

    else:
        car_price_mean = vehicles['Price'].mean()
        update_info('car_price_mean', car_price_mean)
        vehicles['Price'] = (vehicles['Price'] / car_price_mean).astype('float64')

    car_year_min = vehicles['Year'].min()
    update_info('car_year_min', car_year_min)
    vehicles['Year'] = (vehicles['Year'] - car_year_min).astype(int)

    car_mileage_mean = vehicles['Mileage'].mean()
    update_info('car_mileage_mean', car_mileage_mean)
    vehicles['Mileage'] = (vehicles['Mileage'] / car_mileage_mean).astype('float64')

    # ======================================================
    label_encoder_flag = False
    label_encoder = handle_pickle(data_info['label_encoder'], dump=False)
    if label_encoder is None:
        label_encoder = {}
        label_encoder_flag = True

    for col in ['City', 'State', 'Make', 'Model']:
        if col not in label_encoder:
            label_encoder[col] = LabelEncoder()

        label_encoder[col].fit(list(vehicles[col].astype(str).values))
        vehicles[col] = label_encoder[col].transform(list(vehicles[col].astype(str).values))
        print(label_encoder[col].get_params())

    if label_encoder_flag:
        handle_pickle(data_info['label_encoder'], label_encoder)
    # ======================================================

    target_name = 'Price'
    train_target = vehicles[target_name]

    vehicles = vehicles.drop([target_name], axis=1)
    vehicles.sample(5)

    train0, test0, train_target0, test_target0 = train_test_split(vehicles, train_target, test_size=0.2,
                                                                  random_state=0)

    # ======================================================
    scaler_flag = False
    scaler = handle_pickle(data_info['scaler'], dump=False)
    if scaler is None:
        scaler_flag = True
        scaler = StandardScaler()

    train0 = pd.DataFrame(scaler.fit_transform(train0), columns=train0.columns)
    test0 = pd.DataFrame(scaler.transform(test0), columns=test0.columns)

    if scaler_flag:
        handle_pickle(data_info['scaler'], scaler)

    # https://stackoverflow.com/a/54508052/2049763
    train0, train_target0 = train0.to_numpy(), train_target0.to_numpy()
    test0, test_target0 = test0.to_numpy(), test_target0.to_numpy()

    save_dataset_file(train0, test0, train_target0, test_target0, selector)
    return train0, test0, train_target0, test_target0
