from sys import path
from os.path import dirname as dir
project_path = dir(path[0]).split('src', 1)[0]
path.append(project_path)

from src import classifier, data_loading


def test_loss():
    cvd_data_path = f'{project_path}/data/kaggle_heart_disease.csv'
    cvd_data = data_loading.load_data(cvd_data_path)
    model = classifier.NNModel(cvd_data)
    model.cv_train_test()
    assert isinstance(model.get_loss, float)


def test_accuracy():
    cvd_data_path = f'{project_path}/data/kaggle_heart_disease.csv'
    cvd_data = data_loading.load_data(cvd_data_path)
    model = classifier.NNModel(cvd_data)
    model.cv_train_test()
    assert model.get_accuracy <= 1.0


def test_hyperparameters():
    cvd_data_path = f'{project_path}/data/kaggle_heart_disease.csv'
    cvd_data = data_loading.load_data(cvd_data_path)
    model = classifier.NNModel(cvd_data)
    model.cv_train_test()
    assert 'l2_layer_1' in model.get_best_hyperparameters


def test_new_predictions():
    cvd_data_path = f'{project_path}/data/kaggle_heart_disease.csv'
    cvd_data = data_loading.load_data(cvd_data_path)
    cvd_train_data = cvd_data.iloc[:300]
    model = classifier.NNModel(cvd_train_data)
    model.cv_train_test()

    # Using just a couple of rows from the original data set as input to make predictions
    new_data = cvd_data.iloc[300:]
    new_data.pop('target')
    predictions = model.predict(new_data)
    assert all(p >= 0 for p in predictions)
    assert all(p <= 1 for p in predictions)


if __name__ == '__main__':
    test_loss()
    test_accuracy()
    test_hyperparameters()
    test_new_predictions()
