import IPython
import kerastuner as kt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import feature_column, keras


class NNModel:
    def __init__(self, training_df):
        self.training_df = training_df
        self.feature_columns = [col for col in self.training_df.columns if col != 'target']
        self.scaler = StandardScaler(copy=True)

        self.best_params = None
        self.trained_model = None
        self.trained_model_path = None
        self.y_preds = None

        self.input_shape = None
        self.best_hps = None
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None
        self.tuner = None

    def _tune_model(self, hp):
        tf.keras.backend.set_floatx('float64')

        feature_columns = [feature_column.numeric_column(col) for col in self.feature_columns]
        self.input_shape = len(feature_columns)

        inputs = keras.Input(shape=(self.input_shape,))
        hp_units_1 = hp.Int('units_1', min_value=32, max_value=512, step=32)
        dense = keras.layers.Dense(units=hp_units_1, activation="relu")
        x = dense(inputs)
        hp_units_2 = hp.Int('units_2', min_value=32, max_value=512, step=32)
        x = keras.layers.Dense(units=hp_units_2, activation="relu")(x)
        hp_units_3 = hp.Int('units_3', min_value=32, max_value=512, step=32)
        x = keras.layers.Dense(units=hp_units_3, activation="relu")(x)
        hp_units_4 = hp.Int('units_4', min_value=32, max_value=512, step=32)
        x = keras.layers.Dense(units=hp_units_4, activation="relu")(x)
        outputs = keras.layers.Dense(1)(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="heart_disease")
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        return model

    def _find_best_hyperparameters(self):
        train, test = train_test_split(self.training_df, test_size=0.25, random_state=123)
        train, val = train_test_split(train, test_size=0.2)
        self.x_train, self.y_train = self._df_to_numpy(train)
        self.x_val, self.y_val = self._df_to_numpy(val)
        self.x_test, self.y_test = self._df_to_numpy(test)
        tuner = kt.Hyperband(self._tune_model,
                             objective='val_accuracy',
                             max_epochs=5,
                             factor=3,
                             directory='my_dir',
                             project_name='heart_disease')

        tuner.search(self.x_train, self.y_train, epochs=5, validation_data=(self.x_val, self.y_val), callbacks=[ClearTrainingOutput()])

        # Get the optimal hyperparameters
        self.tuner = tuner.hypermodel
        self.best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    def cv_train_test(self):
        """
        Given the full DataFrame with all data points, split into training and holdout data, run through k-fold CV and
        grid search CV before training on full train set and testing on full holdout set
        :return: list of features sorted by importance
        """
        self._find_best_hyperparameters()
        model = self.tuner.build(self.best_hps)
        print(model.summary())
        model_fit = model.fit(self.x_train, self.y_train,
                              validation_data=(self.x_val, self.y_val),
                              epochs=100,
                              batch_size=64)
        loss, accuracy = model.evaluate(self.x_test, self.y_test, verbose=2)
        print('Test loss: ', loss)
        print('Test accuracy: ', accuracy)

        self.trained_model = model

    def predict(self, x_new_df):
        x_new_array = x_new_df.to_numpy()
        x_scaled = self.scaler.fit_transform(x_new_array)
        y_prediction = self.trained_model.predict(x_scaled)

        return y_prediction

    def _df_to_numpy(self, df):
        df = df.copy()
        y_arr = np.array(df.pop('target'))
        x_arr = np.array(df)
        x_scaled_arr = self.scaler.fit_transform(x_arr)

        return x_scaled_arr, y_arr


class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait=True)
