# pylint: disable=no-member
from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import argparse
import tensorflow as tf
import xlrd

import params
import plots
import features
import neuralnets
import tuning
import kfolds
import permutation
import composition_scan


parser = argparse.ArgumentParser()
parser.add_argument('task', default="train", nargs='?', type=str)
parser.add_argument('compositions', default=None, nargs='*', type=str)

args = parser.parse_args()

if args.task in ['train', 'kfolds', 'kfoldsEnsemble',
                 'tune', 'featurePermutation', 'compositionScan']:

    if tf.test.gpu_device_name() != '/device:GPU:0':
        print('WARNING: GPU device not found.')
    else:
        print('SUCCESS: Found GPU: {}'.format(tf.test.gpu_device_name()))

    if args.task == 'train':
        model_type = "neuralnet"
        trainPercentage = 1.0
        maxEpochs = 100
        bayesian = False

        params.setup(model_type+"_train"+str(trainPercentage))

        originalData = features.load_data()

        if trainPercentage < 1.0:

            train, test = features.train_test_split(
                originalData, trainPercentage)

            train_compositions = list(train.pop('composition'))
            test_compositions = list(test.pop('composition'))

            train_ds, test_ds, train_features, test_features, train_labels, test_labels, sampleWeight, sampleWeightTest = features.create_datasets(
                originalData, train, test)

            if model_type == 'neuralnet':
                model = neuralnets.train_model(
                    train_features, train_labels, sampleWeight,
                    test_features=test_features, test_labels=test_labels,
                    sampleWeight_test=sampleWeightTest, maxEpochs=maxEpochs, bayesian=bayesian)

            train_predictions = neuralnets.evaluate_model(
                model, train_ds, train_labels, test_ds=test_ds, test_labels=test_labels,
                train_compositions=train_compositions, test_compositions=test_compositions, bayesian=bayesian)

        else:
            compositions = list(originalData.pop('composition'))

            train_ds, train_features, train_labels, sampleWeight = features.create_datasets(
                originalData)

            model = neuralnets.train_model(
                train_features, train_labels, sampleWeight, maxEpochs=maxEpochs)

            train_predictions = neuralnets.evaluate_model(
                model, train_ds, train_labels, train_compositions=compositions, bayesian=bayesian)

    elif args.task == 'compositionScan':
        composition_scan.run(compositions=args.compositions)
    else:
        if args.task != 'featurePermutation':

            params.setup(args.task)
            originalData = features.load_data()

            if args.task == 'kfolds':
                kfolds.kfolds(originalData)

            elif args.task == 'kfoldsEnsemble':
                kfolds.kfoldsEnsemble(originalData)

            elif args.task == 'tune':

                train_ds, train_features, train_labels, sampleWeight = features.create_datasets(
                    originalData)

                tuning.tune(train_features, train_labels, sampleWeight)
        else:
            permutation.permutation()

else:
    print("Unknown task", args.task)
