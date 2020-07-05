import argparse
from sys import path
from os.path import dirname as dir
project_path = dir(path[0]).split('src', 1)[0]
path.append(project_path)

from src import classifier, data_loading

# Note: Due to not being containerized, the sys path modification above allows for easier importing from this project,
# regardless of where it's being run. The same path is used below to ensure that data can be loaded without
# modifying this script.


def run_model_with_predictions(new_data_path=''):
    cvd_data_path = f'{project_path}/data/kaggle_heart_disease.csv'
    cvd_data = data_loading.load_data(cvd_data_path)
    cvd_train_data = cvd_data.iloc[:300]

    model = classifier.NNModel(cvd_train_data)
    model.cv_train_test()
    print('Test loss: ', model.get_loss)
    print('Test accuracy: ', model.get_accuracy)
    print('Best hyperparameters selected: ', model.get_best_hyperparameters)

    if new_data_path:
        new_data = data_loading.load_data(new_data_path)
        if 'target' in new_data.columns:
            new_data.pop('target')
    else:
        # Using just a couple of rows from the original data set as input to make predictions
        new_data = cvd_data.iloc[300:]
        new_data.pop('target')
    predictions = model.predict(new_data)
    print('New data predictions: ', predictions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get new data for predictions.')
    parser.add_argument('-p', '--path_to_new_data', default='', required=False, type=str)
    parsed_args = parser.parse_args()
    run_model_with_predictions(new_data_path=parsed_args.path_to_new_data)
