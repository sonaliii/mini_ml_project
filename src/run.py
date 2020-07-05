from src import classifier, data_loading, text_processor


if __name__ == '__main__':
    cvd_data_path = '~/Desktop/mini_ml_project/data/kaggle_heart_disease.csv'
    cvd_data = data_loading.load_data(cvd_data_path)
    print(cvd_data.head())
    print(cvd_data['target'].unique())

    model = classifier.NNModel(cvd_data)
    model.cv_train_test()

