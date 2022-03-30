# pylint: disable=no-member
from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import xlrd
from omegaconf import OmegaConf

import cerebral as cb

import composition_scan

if __name__ == '__main__':

    conf = OmegaConf.load('config.yaml')

    cb.setup()

    if conf.task in ['simple', 'kfolds', 'kfoldsEnsemble',
                     'tune', 'featurePermutation', 'compositionScan']:

        if conf.task == 'simple':
            train_percentage = conf.train.get("train_percentage", 1.0)
            max_epochs = conf.train.get("max_epochs", 100)

            originalData = cb.io.load_data()

            if train_percentage < 1.0:

                train, test = cb.features.train_test_split(
                    originalData, train_percentage)

                train_compositions = list(train.pop('composition'))
                test_compositions = list(test.pop('composition'))

                train_ds, test_ds, train_features, test_features, train_labels, test_labels, sampleWeight, sampleWeightTest = cb.features.create_datasets(
                    originalData, train, test)

                model = cb.models.train_model(
                    train_features, train_labels, sampleWeight,
                    test_features=test_features, test_labels=test_labels,
                    sampleWeight_test=sampleWeightTest, maxEpochs=max_epochs)

                train_predictions = cb.models.evaluate_model(
                    model, train_ds, train_labels, test_ds=test_ds, test_labels=test_labels,
                    train_compositions=train_compositions, test_compositions=test_compositions)

            else:
                compositions = list(originalData.pop('composition'))

                train_ds, train_features, train_labels, sampleWeight = cb.features.create_datasets(
                    originalData)

                model = cb.models.train_model(
                    train_features, train_labels, sampleWeight, maxEpochs=max_epochs)

                train_predictions = cb.models.evaluate_model(
                    model, train_ds, train_labels, train_compositions=compositions)

        elif conf.task == 'compositionScan':
            composition_scan.run(compositions=conf.compositions)

        else:
            if conf.task != 'featurePermutation':

                originalData = cb.io.load_data()

                if conf.task == 'kfolds':
                    kfolds.kfolds(originalData)

                elif conf.task == 'kfoldsEnsemble':
                    kfolds.kfoldsEnsemble(originalData)

                elif conf.task == 'tune':

                    train_ds, train_features, train_labels, sampleWeight = cb.features.create_datasets(
                        originalData)

                    tuning.tune(train_features, train_labels, sampleWeight)
            else:
                permutation.permutation()

    else:
        print("Unknown task", conf.task)
