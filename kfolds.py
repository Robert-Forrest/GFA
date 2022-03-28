# pylint: disable=no-member
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import datetime

import tensorflow as tf  # pylint: disable=import-error
import pandas as pd  # pylint: disable=import-error
from sklearn.model_selection import StratifiedKFold  # pylint: disable=import-error
import numpy as np  # pylint: disable=import-error

import features
import neuralnets
import plots
import params

numFolds = 5


def kfolds_split(data, numFolds):
    data = data.copy()

    unique_composition_spaces = {}
    for _, row in data.iterrows():
        composition = features.parse_composition(row['composition'])
        sorted_composition = sorted(list(composition.keys()))
        composition_space = "".join(sorted_composition)

        if composition_space not in unique_composition_spaces:
            unique_composition_spaces[composition_space] = []

        unique_composition_spaces[composition_space].append(row)

    shuffled_unique_compositions = list(unique_composition_spaces.keys())
    np.random.shuffle(shuffled_unique_compositions)

    foldSize = int(np.floor(len(shuffled_unique_compositions) / numFolds))

    folds = []
    for i in range(numFolds):
        trainingSetCompositions = []
        if i > 0:
            trainingSetCompositions = shuffled_unique_compositions[0:i*foldSize]

        testSetCompositions = shuffled_unique_compositions[i*foldSize:(
            i+1)*foldSize]

        if i < numFolds - 1:
            trainingSetCompositions.extend(
                shuffled_unique_compositions[(i+1)*foldSize:len(shuffled_unique_compositions)-1])

        trainingSet = []
        testSet = []
        for composition in trainingSetCompositions:
            trainingSet.extend(unique_composition_spaces[composition])
        for composition in testSetCompositions:
            testSet.extend(unique_composition_spaces[composition])

        folds.append([pd.DataFrame(trainingSet), pd.DataFrame(testSet)])

    return folds


def kfolds(originalData, save=False, plot=False):

    MADs = {}
    RMSDs = {}
    for feature in features.predictableFeatures:
        if feature != 'GFA':
            MADs[feature] = plots.meanAbsoluteDeviation(
                plots.filter_masked(originalData[feature]))
            RMSDs[feature] = plots.rootMeanSquareDeviation(
                plots.filter_masked(originalData[feature]))

    MAEs = {}
    RMSEs = {}
    accuracies = {}
    f1s = {}
    for feature in features.predictableFeatures:
        if feature != 'GFA':
            MAEs[feature] = []
            RMSEs[feature] = []
        else:
            accuracies[feature] = []
            f1s[feature] = []

    fold_test_labels = []
    fold_test_predictions = []

    folds = kfolds_split(originalData, numFolds)

    for foldIndex in range(numFolds):

        train_tmp = folds[foldIndex][0]
        test_tmp = folds[foldIndex][1]

        train_compositions = train_tmp.pop('composition')
        test_compositions = test_tmp.pop('composition')

        train_ds, test_ds, train_features, test_features, train_labels, test_labels, sampleWeight, sampleWeightTest = features.create_datasets(
            originalData, train=train_tmp, test=test_tmp)

        model = neuralnets.train_model(train_features, train_labels,
                                       sampleWeight,
                                       test_features=test_features, test_labels=test_labels, sampleWeight_test=sampleWeightTest,
                                       plot=plot, maxEpochs=1000, model_name=foldIndex)
        if save and not plot:
            neuralnets.save(
                model,
                params.output_directory +
                '/model_' +
                str(foldIndex))

        train_predictions, test_predictions = neuralnets.evaluate_model(
            model, train_ds, train_labels, test_ds, test_labels, plot=plot, model_name=foldIndex)

        fold_test_labels.append(test_labels)
        fold_test_predictions.append(test_predictions)

        for feature in features.predictableFeatures:
            featureIndex = features.predictableFeatures.index(feature)
            if feature != 'GFA':

                test_labels_masked, test_predictions_masked = plots.filter_masked(
                    test_labels[feature], test_predictions[featureIndex].flatten())

                MAEs[feature].append(plots.calc_MAE(
                    test_labels_masked, test_predictions_masked))
                RMSEs[feature].append(plots.calc_RMSE(
                    test_labels_masked, test_predictions_masked))
            else:
                test_labels_masked, test_predictions_masked = plots.filter_masked(
                    test_labels[feature], test_predictions[featureIndex])

                accuracies[feature].append(plots.calc_accuracy(
                    test_labels_masked, test_predictions_masked))
                f1s[feature].append(plots.calc_f1(
                    test_labels_masked, test_predictions_masked))

    with open(params.output_directory + '/validation.dat', 'w') as validationFile:
        for feature in features.predictableFeatures:
            if feature != 'GFA':
                validationFile.write('# ' + feature + '\n')
                validationFile.write('# MAD RMSD\n')
                validationFile.write(
                    str(MADs[feature]) + ' ' + str(RMSDs[feature]) + '\n')
                validationFile.write('# foldId MAE RMSE\n')
                for i in range(len(MAEs[feature])):
                    validationFile.write(
                        str(i) + ' ' + str(MAEs[feature][i]) + ' ' + str(RMSEs[feature][i]) + '\n')
                validationFile.write('# Mean \n')
                validationFile.write(
                    str(np.mean(MAEs[feature])) + ' ' + str(np.mean(RMSEs[feature])) + '\n')
                validationFile.write('# Standard Deviation \n')
                validationFile.write(
                    str(np.std(MAEs[feature])) + ' ' + str(np.std(RMSEs[feature])) + '\n\n')
            else:
                validationFile.write('# ' + feature + '\n')
                validationFile.write('# foldId accuracy f1\n')
                for i in range(len(accuracies[feature])):
                    validationFile.write(
                        str(i) + ' ' + str(accuracies[feature][i]) + ' ' + str(f1s[feature][i]) + '\n')
                validationFile.write('# Mean \n')
                validationFile.write(
                    str(np.mean(accuracies[feature])) + ' ' + str(np.mean(f1s[feature])) + '\n')
                validationFile.write('# Standard Deviation \n')
                validationFile.write(
                    str(np.std(accuracies[feature])) + ' ' + str(np.std(f1s[feature])) + '\n\n')

    plots.plot_results_regression_heatmap(
        fold_test_labels, fold_test_predictions)


def kfoldsEnsemble(originalData):
    #kfolds(originalData, save=True, plot=True)

    compositions = originalData.pop('composition')

    train_ds, train_features, train_labels, sampleWeight = features.create_datasets(
        originalData)

    inputs = neuralnets.build_input_layers(train_features)
    outputs = []
    losses, lossWeights, metrics = neuralnets.setup_losses()
    # lossWeights = {
    #     'Tl': 1,
    #     'Tg': 1,
    #     'Tx': 1,
    #     'deltaT': 1,
    #     'Dmax': 1,
    #     'GFA': 1
    # }

    submodel_outputs = []
    for k in range(numFolds):
        submodel = neuralnets.load(
            params.output_directory + '/model_' + str(k))
        submodel._name = "ensemble_" + str(k)
        for layer in submodel.layers:
            layer.trainable = False

        submodel_outputs.append(submodel(inputs))

    for i in range(len(features.predictableFeatures)):

        submodel_output = [output[i] for output in submodel_outputs]

        submodels_merged = tf.keras.layers.concatenate(submodel_output)

        hidden = tf.keras.layers.Dense(64, activation="relu")(submodels_merged)

        output = None
        if features.predictableFeatures[i] in ["GFA"]:
            activation = "softmax"
            numNodes = 3
        else:
            activation = "softplus"
            numNodes = 1

        output = tf.keras.layers.Dense(
            numNodes,
            activation=activation,
            name=features.predictableFeatures[i])(hidden)
        outputs.append(output)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    tf.keras.utils.plot_model(
        model, to_file=params.image_directory + 'model_ensemble.png', rankdir='LR')

    learning_rate = 0.01
    optimiser = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimiser,
        loss=losses,
        loss_weights=lossWeights,
        metrics=metrics)

    model, history = neuralnets.fit(
        model, train_features, train_labels, sampleWeight, maxEpochs=1000)
    neuralnets.save(model, params.output_directory + '/model')

    plots.plot_training(history)
    train_predictions = neuralnets.evaluate_model(
        model, train_ds, train_labels, train_compositions=compositions, bayesian=False)
