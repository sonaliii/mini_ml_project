import IPython
import kerastuner as kt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import feature_column, keras


class ClearTrainingOutput(tf.keras.callbacks.Callback):
    """
    Clear output during keras hyperparameter tuning step
    """
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait=True)


class NNModel:
    """
    Build deep learning binary classifier model using tensorflow/keras for predicting cardiovascular disease in patients
    """
    def __init__(self, training_df):
        self.training_df = training_df
        self.feature_columns = [col for col in self.training_df.columns if col != 'target']
        self.scaler = StandardScaler(copy=True)

        # Initialize model input variables used for training, validation, and testing
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None
        self.tuner = None

        # Initialize variables output by the model
        self.trained_model = None
        self.best_hyperparameters = None
        self.y_predictions = None
        self.test_loss = None
        self.test_accuracy = None

    def _tune_model(self, hp):
        """
        Use keras tuner to select hyperparameters for deep neural network.
        This neural network is a binary classifier for predicting cardiovascular disease.
        Using Adam optimizer with a binary cross-entropy loss function, this model is optimized for accuracy.
        Hyperparameter tuning is used here for learning rate, l2 regularization, and hidden layer sizes.
        :param hp: keras hyperparameter object
        :return: keras NN model with optimal hyperparameters selected
        """
        tf.keras.backend.set_floatx('float64')

        feature_columns = [feature_column.numeric_column(col) for col in self.feature_columns]
        input_shape = len(feature_columns)

        initializer = keras.initializers.GlorotNormal()
        inputs = keras.Input(shape=(input_shape,))
        hp_l2_1 = hp.Float('l2_1', min_value=0.001, max_value=0.1)
        hp_units_1 = hp.Int('units_1', min_value=32, max_value=256, step=32)
        dense = keras.layers.Dense(units=hp_units_1, activation='relu',
                                   kernel_regularizer=keras.regularizers.l2(hp_l2_1),
                                   kernel_initializer=initializer)
        x = dense(inputs)
        hp_units_2 = hp.Int('units_2', min_value=32, max_value=128, step=32)
        hp_l2_2 = hp.Float('l2_2', min_value=0.001, max_value=0.1)
        x = keras.layers.Dense(units=hp_units_2, activation='relu',
                               kernel_regularizer=keras.regularizers.l2(hp_l2_2))(x)
        hp_units_3 = hp.Int('units_3', min_value=32, max_value=64, step=32)
        hp_l2_3 = hp.Float('l2_3', min_value=0.001, max_value=0.1)
        x = keras.layers.Dense(units=hp_units_3, activation='relu',
                               kernel_regularizer=keras.regularizers.l2(hp_l2_3))(x)
        outputs = keras.layers.Dense(units=1, activation='sigmoid')(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name='heart_disease')
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        return model

    def _find_best_hyperparameters(self):
        """
        Apply keras hyperparameter tuning to training and validation data after splitting into train/validation/test.
        keras tuner will create a directory 'hp_tuning' in this src folder.
        """
        train, test = train_test_split(self.training_df, test_size=0.2, random_state=123)
        train, val = train_test_split(train, test_size=0.2)
        self.x_train, self.y_train = self._df_to_numpy(train)
        self.x_val, self.y_val = self._df_to_numpy(val)
        self.x_test, self.y_test = self._df_to_numpy(test)
        tuner = kt.Hyperband(self._tune_model,
                             objective='val_accuracy',
                             max_epochs=25,
                             factor=3,
                             directory='hp_tuning',
                             project_name='heart_disease')

        tuner.search(self.x_train,
                     self.y_train,
                     epochs=20,
                     validation_data=(self.x_val, self.y_val),
                     callbacks=[ClearTrainingOutput()])

        # Get the optimal hyperparameters
        self.tuner = tuner.hypermodel
        self.best_hyperparameters = tuner.get_best_hyperparameters(num_trials=3)[0]

    def cv_train_test(self):
        """
        Use keras tuner to select best hyperparameters, and train the optimized model on the full train data set.
        """
        self._find_best_hyperparameters()
        model = self.tuner.build(self.best_hyperparameters)
        model.fit(self.x_train,
                  self.y_train,
                  validation_data=(self.x_val, self.y_val),
                  epochs=100,
                  batch_size=64)
        self.test_loss, self.test_accuracy = model.evaluate(self.x_test, self.y_test, verbose=2)
        self.trained_model = model

    def predict(self, x_new_df):
        """
        Predict heart disease outcome y, given x. First scale x to zero mean, unit variance. Then use trained model
        to make predictions.
        :param x_new_df: pandas DataFrame containing input variables
        :return: numpy array of predictions
        """
        x_new_array = x_new_df.to_numpy()
        x_scaled = self.scaler.fit_transform(x_new_array)
        y_predictions = self.trained_model.predict(x_scaled)

        return y_predictions

    def _df_to_numpy(self, df):
        """
        Convert a pandas DataFrame containing both features and target variable into separate x and y numpy arrays.
        Scale x to zero mean, unit variance.
        :param df: pandas DataFrame containing both y ('target') and x (all input features)
        :return: tuple of numpy arrays x (normalized) and y
        """
        df = df.copy()
        y_arr = np.array(df.pop('target'))
        x_arr = np.array(df)
        # normalizing here makes the cost function faster to optimize
        x_scaled_arr = self.scaler.fit_transform(x_arr)

        return x_scaled_arr, y_arr

    @property
    def get_accuracy(self):
        return self.test_accuracy

    @property
    def get_loss(self):
        return self.test_loss

    @property
    def get_best_hyperparameters(self):
        """
        Restructure selected hyperparameters into a dictionary for easier inspection.
        :return: dict of selected optimal hyperparameters
        """
        best_hp_dict = {
            'l2_layer_1': self.best_hyperparameters.get('l2_1'),
            'l2_layer_2': self.best_hyperparameters.get('l2_2'),
            'l2_layer_3': self.best_hyperparameters.get('l2_3'),
            'hidden_units_layer_1': self.best_hyperparameters.get('units_1'),
            'hidden_units_layer_2': self.best_hyperparameters.get('units_2'),
            'hidden_units_layer_3': self.best_hyperparameters.get('units_3'),
            'learning_rate': self.best_hyperparameters.get('learning_rate')
        }
        return best_hp_dict
