import argparse

import seaborn as sns
from omegaconf import OmegaConf
import cerebral as cb

import composition_scan

sns.set()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("config", default="config.yaml", nargs="?", type=str)

    args = parser.parse_args()

    conf = OmegaConf.load(args.config)

    if conf.task in [
        "simple",
        "kfolds",
        "kfoldsEnsemble",
        "tune",
        "feature_permutation",
        "composition_scan",
    ]:

        if conf.task == "simple":
            cb.setup(conf)

            train_percentage = conf.train.get("train_percentage", 1.0)

            data = cb.features.load_data(
                postprocess=cb.GFA.ensure_default_values_glass,
                drop_correlated_features=conf.get("drop_correlated_features", True),
            )

            if train_percentage < 1.0:

                model, history, train_ds, test_ds = cb.models.train_model(data)

                (
                    train_results,
                    test_results,
                    metrics,
                ) = cb.models.evaluate_model(model, train_ds, test_ds=test_ds)

            else:
                model, history, train_ds = cb.models.train_model(data)

                train_results, metrics = cb.models.evaluate_model(model, train_ds)

            print(metrics)

        elif conf.task == "composition_scan":
            composition_scan.run(
                model=conf.model_path,
                compositions=conf.compositions,
                properties=conf.properties,
                uncertainty=conf.get("model_uncertainty", False),
            )

        elif conf.task != "feature_permutation":
            cb.setup(conf)
            originalData = cb.features.load_data(
                postprocess=cb.GFA.ensure_default_values_glass
            )

            if conf.task == "kfolds":
                cb.kfolds.kfolds(originalData)

            elif conf.task == "kfoldsEnsemble":
                cb.kfolds.kfoldsEnsemble(originalData)

            elif conf.task == "tune":

                train_ds = cb.features.create_datasets(
                    originalData, targets=cb.conf.targets
                )

                cb.tuning.tune(train_ds)
        else:
            cb.permutation.permutation(postprocess=cb.GFA.ensure_default_values_glass)

    else:
        raise NotImplementedError("Unknown task: " + conf.task)
