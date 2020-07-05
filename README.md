# Mini Machine Learning Project

## The purpose of this project is to incorporate critical features of project structure and organization into a demo.

### Running the model
```run.py``` contains code for building the entire model - loading data, generating features, building a model, and making new predictions.
If you want to run the model and make predictions on the small training data set, just run:
```python3 run.py```

If you have new data stored in `new_data.csv`, then run the following:
```python3 run.py --path_to_new_data new_data.csv```

### Data Loading
```data_loading.py``` contains code for loading data from a locally saved csv file into a pandas DataFrame.

### Model Building
```classifier.py``` contains code for building a keras/tensorflow deep learning binary classifier model.


### Feature Engineering, Data Cleaning, etc.
This project was built using a small Kaggle data set that had processed/numerical features already. 
Feature generation and data cleaning are not within the scope of this mini-project.
