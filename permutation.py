# pylint: disable=no-member
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np  # pylint: disable=import-error

import features
import neuralnets
import plots
import params
numPermutations = 5


def permutation():

    model_name = "kfoldsEnsemble"
    params.setup(model_name, existingModel=True)

    open(params.output_directory + '/permutedFeatures.dat', 'w')

    model = neuralnets.load(params.output_directory+'/model')
    originalData = features.load_data(model=model)

    permutationImportance = {}
    for permutedFeature in ['none'] + list(originalData.columns):
        if permutedFeature not in features.predictableFeatures and permutedFeature != 'composition':

            permutationImportance[permutedFeature] = {}
            for feature in features.predictableFeatures:
                permutationImportance[permutedFeature][feature] = []

            for k in range(numPermutations):

                data = originalData.copy()
                if permutedFeature != 'none':
                    data[permutedFeature] = np.random.permutation(
                        data[permutedFeature].values)

                train_ds, train_features, labels, sampleWeight = features.create_datasets(
                    data)

                predictions = neuralnets.evaluate_model(
                    model, train_ds, labels, plot=False)

                for feature in features.predictableFeatures:
                    featureIndex = features.predictableFeatures.index(feature)
                    if feature != 'GFA':

                        labels_masked, predictions_masked = plots.filter_masked(
                            labels[feature], predictions[featureIndex].flatten())

                        permutationImportance[permutedFeature][feature].append(plots.calc_MAE(
                            labels_masked, predictions_masked))

                    else:

                        labels_masked, predictions_masked = plots.filter_masked(
                            labels[feature], predictions[featureIndex])

                        permutationImportance[permutedFeature][feature].append(plots.calc_accuracy(
                            labels_masked, predictions_masked))
                if permutedFeature == 'none':
                    for feature in features.predictableFeatures:
                        permutationImportance[permutedFeature][feature] = permutationImportance[permutedFeature][feature][0]
                    break

            if permutedFeature != 'none':
                for feature in features.predictableFeatures:
                    averageScore = 0
                    for i in range(numPermutations):
                        averageScore += permutationImportance[permutedFeature][feature][i]
                    averageScore /= numPermutations
                    if feature != 'GFA':
                        permutationImportance[permutedFeature][feature] = max(0, averageScore -
                                                                              permutationImportance['none'][feature])
                    else:
                        permutationImportance[permutedFeature][feature] = max(
                            0, permutationImportance['none'][feature] - averageScore)

            with open(params.output_directory + '/permutedFeatures.dat', 'a') as resultsFile:
                for feature in features.predictableFeatures:
                    resultsFile.write(permutedFeature + ' ' + feature + ' ' +
                                      " " + str(permutationImportance[permutedFeature][feature]) + '\n')

    del permutationImportance['none']
    plots.plot_feature_permutation(permutationImportance)
