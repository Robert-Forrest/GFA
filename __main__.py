import os
import argparse

from omegaconf import OmegaConf
import cerebral as cb
import tensorflow as tf

import composition_scan

if __name__ == "__main__":

    # tf.keras.backend.set_floatx("float64")
    # physical_devices = tf.config.list_physical_devices("GPU")
    # try:
    #     tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # except:
    #     print("Could not set memory growth")

    # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # tf.get_logger().setLevel("INFO")

    parser = argparse.ArgumentParser()
    parser.add_argument("config", default="config.yaml", nargs="?", type=str)

    args = parser.parse_args()

    conf = OmegaConf.load(args.config)

    cb.setup(conf)

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

        elif conf.task == "compositionScan":
            composition_scan.run(compositions=conf.compositions)

        elif conf.task != "feature_permutation":

            originalData = cb.features.load_data(
                postprocess=cb.GFA.ensure_default_values_glass
            )

            if conf.task == "kfolds":
                cb.kfolds.kfolds(originalData)

            elif conf.task == "kfoldsEnsemble":
                cb.kfolds.kfoldsEnsemble(originalData)

            elif conf.task == "tune":

                (
                    train_ds,
                    train_features,
                    sampleWeight,
                ) = cb.features.create_datasets(originalData)

                cb.tuning.tune(train_features, sampleWeight)
        else:
            cb.permutation.permutation(postprocess=cb.GFA.ensure_default_values_glass)

    else:
        raise NotImplementedError("Unknown task: " + conf.task)
