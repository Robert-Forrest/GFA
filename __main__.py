import argparse

from omegaconf import OmegaConf
import cerebral as cb

import data
import composition_scan

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("config", default="config.yaml", nargs="?", type=str)

    args = parser.parse_args()

    conf = OmegaConf.load(args.config)

    cb.setup(config=args.config)

    if conf.task in [
        "simple",
        "kfolds",
        "kfoldsEnsemble",
        "tune",
        "feature_permutation",
        "compositionScan",
    ]:

        if conf.task == "simple":
            train_percentage = conf.train.get("train_percentage", 1.0)
            max_epochs = conf.train.get("max_epochs", 100)

            originalData = cb.io.load_data(postprocess=data.ensure_default_values)

            if train_percentage < 1.0:

                train, test = cb.features.train_test_split(
                    originalData, train_percentage
                )

                train_compositions = list(train.pop("composition"))
                test_compositions = list(test.pop("composition"))

                (
                    train_ds,
                    test_ds,
                    train_features,
                    test_features,
                    train_labels,
                    test_labels,
                    sampleWeight,
                    sampleWeightTest,
                ) = cb.features.create_datasets(originalData, conf.targets, train, test)

                model = cb.models.train_model(
                    train_features,
                    train_labels,
                    sampleWeight,
                    test_features=test_features,
                    test_labels=test_labels,
                    sampleWeight_test=sampleWeightTest,
                    maxEpochs=max_epochs,
                )

                train_predictions = cb.models.evaluate_model(
                    model,
                    train_ds,
                    train_labels,
                    test_ds=test_ds,
                    test_labels=test_labels,
                    train_compositions=train_compositions,
                    test_compositions=test_compositions,
                )

            else:
                compositions = list(originalData.pop("composition"))

                (
                    train_ds,
                    train_features,
                    train_labels,
                    sampleWeight,
                ) = cb.features.create_datasets(originalData)

                model = cb.models.train_model(
                    train_features, train_labels, sampleWeight, maxEpochs=max_epochs
                )

                train_predictions = cb.models.evaluate_model(
                    model, train_ds, train_labels, train_compositions=compositions
                )

        elif conf.task == "compositionScan":
            composition_scan.run(compositions=conf.compositions)

        else:
            if conf.task != "feature_permutation":

                originalData = cb.io.load_data(postprocess=data.ensure_default_values)

                if conf.task == "kfolds":
                    cb.kfolds.kfolds(originalData)

                elif conf.task == "kfoldsEnsemble":
                    cb.kfolds.kfoldsEnsemble(originalData)

                elif conf.task == "tune":

                    (
                        train_ds,
                        train_features,
                        train_labels,
                        sampleWeight,
                    ) = cb.features.create_datasets(originalData)

                    cb.tuning.tune(train_features, train_labels, sampleWeight)
            else:
                cb.permutation.permutation(postprocess=data.ensure_default_values)

    else:
        print("Unknown task", conf.task)
