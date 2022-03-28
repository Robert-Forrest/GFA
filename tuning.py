import numpy as np
import pandas as pd
import kerastuner as kt  # pylint: disable=import-error
from sklearn.model_selection import StratifiedKFold  # pylint: disable=import-error
import tensorflow as tf  # pylint: disable=import-error

import neuralnets
import features
import params


class HyperModel(kt.HyperModel):
    def __init__(self, train_features):
        self.train_features = train_features

    def build(self, hp):

        num_shared_layers = hp.Int('num_shared_layers', min_value=2,
                                   max_value=10, step=1)
        num_specific_layers = hp.Int('num_specific_layers', min_value=2,
                                     max_value=10, step=1)

        # units_per_layer = hp.Int('units_per_layer', min_value=32,
        #                         max_value=512, step=32)
        units_per_layer = hp.Choice('units_per_layer', values=[
                                    8, 16, 32, 64, 128, 256, 512])

        model = neuralnets.build_model(
            train_features=self.train_features,
            num_shared_layers=num_shared_layers,
            num_specific_layers=num_specific_layers,
            units_per_layer=units_per_layer,
            #regularizer=hp.Choice('regularizer', values=["l1", "l2", "l1l2"]),
            # regularizer_rate=hp.Choice('regularizer_rate', values=[
            #    1e-2, 1e-3, 1e-4, 1e-5]),
            #dropout=hp.Choice('dropout', values=[0.1, 0.2, 0.3, 0.4, 0.5]),
            # learning_rate=1e-3,
            # activation=hp.Choice('activation', values=[
            #    'relu', 'tanh', 'sigmoid', 'elu']),
            #max_norm=hp.Choice('max_norm', values=[1, 3, 5]),
        )
        return model


# class Tuner(kt.engine.tuner.Tuner):
#     def setup(self, allData, train_features, train_labels, test_ds, predictableFeatures, numFolds=2):
#         self.allData = allData
#         self.train_features = train_features
#         self.train_labels = train_labels
#         self.test_ds = test_ds
#         self.predictableFeatures = predictableFeatures
#         self.numFolds = numFolds

#     def run_trial(self, trial, x, y, batch_size=features.batch_size, epochs=1):

#         val_losses = []
#         # skf = StratifiedKFold(n_splits=self.numFolds)
#         # for train_index, test_index in skf.split(np.zeros(len(self.trainingData['GFA'])), self.trainingData['GFA']):

#         #     train_tmp = self.trainingData.iloc[train_index]
#         #     test_tmp = self.trainingData.iloc[test_index]

#         model = self.hypermodel.build(trial.hyperparameters)

#         xTrain = {}
#         for feature in self.train_features:
#             xTrain[feature] = self.train_features[feature]

#         yTrain = {}
#         for feature in self.predictableFeatures:
#             yTrain[feature] = self.train_labels[feature]

#         model.fit(xTrain, yTrain,
#                   batch_size=batch_size,
#                   epochs=epochs,
#                   verbose=2)
#         val_losses.append(model.evaluate(self.test_ds))

#         self.oracle.update_trial(
#             trial.trial_id, {'val_loss': np.mean(val_losses)})
#         self.save_model(trial.trial_id, model)


# tuner = kt.tuners.Hyperband(
#     hypermodel,
#     objective='loss',
#     factor=3,
#     max_epochs=100,
#     hyperband_iterations=2,
#     directory=output_directory + '/models',
#     project_name='GFA')

# tuner.search_space_summary()

# if tune:
#     numFolds = 2

#     tuner.search(
#         x=xTrain,
#         y=yTrain,
#         batch_size=features.batch_size,
#         verbose=2,
#         sample_weight=sampleWeight)

def tune(train_features, train_labels, sampleWeight, test_features=None, test_labels=None, sampleWeightTest=None, tuner="bayesian"):

    if tuner == "bayesian":
        tuner = kt.tuners.BayesianOptimization(
            HyperModel(train_features),
            objective='loss',
            max_trials=1000,
            directory=params.output_directory + '/models',
            project_name='GFA')

    elif tuner == "hyperband":
        tuner = kt.tuners.Hyperband(
            HyperModel(train_features),
            objective='loss',
            factor=3,
            max_epochs=1000,
            hyperband_iterations=1,
            directory=params.output_directory + '/models',
            project_name='GFA')

    patience = 100
    min_delta = 0.01

    xTrain = {}
    for feature in train_features:
        xTrain[feature] = train_features[feature]

    yTrain = {}
    for feature in features.predictableFeatures:
        yTrain[feature] = train_labels[feature]

    if test_features is not None:
        xTest = {}
        for feature in test_features:
            xTest[feature] = test_features[feature]

        yTest = {}
        for feature in features.predictableFeatures:
            yTest[feature] = test_labels[feature]

        tuner.search(
            x=xTrain,
            y=yTrain,
            batch_size=features.batch_size,
            epochs=1000,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='loss',
                    patience=patience,
                    min_delta=min_delta,
                    mode="auto",
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="loss",
                    factor=0.75,
                    patience=patience // 3,
                    mode="auto",
                    min_delta=min_delta * 10,
                    cooldown=patience // 4,
                    min_lr=0
                )
            ],
            verbose=2,
            validation_data=(xTest, yTest, sampleWeightTest),
            sample_weight=sampleWeight)

    else:

        tuner.search(
            x=xTrain,
            y=yTrain,
            batch_size=features.batch_size,
            epochs=1000,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='loss',
                    patience=patience,
                    min_delta=min_delta,
                    mode="auto",
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="loss",
                    factor=0.75,
                    patience=patience // 3,
                    mode="auto",
                    min_delta=min_delta * 10,
                    cooldown=patience // 4,
                    min_lr=0
                )
            ],
            verbose=2,
            sample_weight=sampleWeight)
