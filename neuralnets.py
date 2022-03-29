import datetime
import os
import features
import plots
import params
import tensorflow_probability as tfp   # pylint: disable=import-error
import numpy as np
import tensorflow as tf   # pylint: disable=import-error
K = tf.keras.backend
tfd = tfp.distributions
# import tensorflow_model_analysis as tfma
# import tensorflow_addons as tfa
tf.keras.backend.set_floatx('float64')
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    print("Could not set memory growth")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.get_logger().setLevel('INFO')


def tprPerClass(y_true, y_pred, class_index=0):

    pred = K.argmax(y_pred)
    true = K.squeeze(y_true, axis=-1)

    mask = K.cast(K.not_equal(true, features.maskValue), 'int64')

    pp = K.cast(K.equal(pred, class_index), 'int64') * mask
    p = K.cast(K.equal(true, class_index), 'int64')
    tp = K.dot(K.reshape(pp, (1, -1)), K.reshape(p, (-1, 1)))

    return K.cast(K.sum(tp), 'float64') / (K.cast(
        K.sum(p), 'float64') + K.epsilon())


def truePositiveRate(y_true, y_pred):
    return (tprPerClass(y_true, y_pred, 0) + tprPerClass(y_true,
                                                         y_pred, 1) + tprPerClass(y_true, y_pred, 2)) / 3


def ppvPerClass(y_true, y_pred, class_index=0):

    pred = K.argmax(y_pred)
    true = K.squeeze(y_true, axis=-1)

    mask = K.cast(K.not_equal(true, features.maskValue), 'int64')

    pp = K.cast(K.equal(pred, class_index), 'int64') * mask
    p = K.cast(K.equal(true, class_index), 'int64')
    tp = K.dot(K.reshape(pp, (1, -1)), K.reshape(p, (-1, 1)))

    return K.cast(K.sum(tp), 'float64') / (K.cast(
        K.sum(pp), 'float64') + K.epsilon())


def positivePredictiveValue(y_true, y_pred):
    return (ppvPerClass(y_true, y_pred, 0) + ppvPerClass(y_true,
                                                         y_pred, 1) + ppvPerClass(y_true, y_pred, 2)) / 3


def f1_score(y_true, y_pred):
    positivePredictiveValue_val = positivePredictiveValue(y_true, y_pred)
    truePositiveRate_val = truePositiveRate(y_true, y_pred)
    return (2 * positivePredictiveValue_val * truePositiveRate_val) / \
        (positivePredictiveValue_val +
         truePositiveRate_val +
         tf.keras.backend.epsilon())


def tnrPerClass(y_true, y_pred, class_index=0):

    pred = K.argmax(y_pred)
    true = K.squeeze(y_true, axis=-1)

    mask = K.cast(K.not_equal(true, features.maskValue), 'int64')

    pn = K.cast(K.not_equal(pred, class_index), 'int64') * mask
    n = K.cast(K.not_equal(true, class_index), 'int64')
    tn = K.dot(K.reshape(pn, (1, -1)), K.reshape(n, (-1, 1)))

    return K.cast(K.sum(tn), 'float64') / (K.cast(
        K.sum(n), 'float64') + K.epsilon())


def trueNegativeRate(y_true, y_pred):
    return (tnrPerClass(y_true, y_pred, 0) + tnrPerClass(y_true,
                                                         y_pred, 1) + tnrPerClass(y_true, y_pred, 2)) / 3


def npvPerClass(y_true, y_pred, class_index=0):

    pred = K.argmax(y_pred)
    true = K.squeeze(y_true, axis=-1)

    mask = K.cast(K.not_equal(true, -1), 'int64')

    pn = K.cast(K.not_equal(pred, class_index), 'int64') * mask
    n = K.cast(K.not_equal(true, class_index), 'int64')
    tn = K.dot(K.reshape(pn, (1, -1)), K.reshape(n, (-1, 1)))

    return K.cast(K.sum(tn), 'float64') / (K.cast(
        K.sum(pn), 'float64') + K.epsilon())


def negativePredictiveValue(y_true, y_pred):
    return (npvPerClass(y_true, y_pred, 0) + npvPerClass(y_true,
                                                         y_pred, 1) + npvPerClass(y_true, y_pred, 2)) / 3


def informedness(y_true, y_pred):
    return truePositiveRate(y_true, y_pred) + \
        trueNegativeRate(y_true, y_pred) - 1


def markedness(y_true, y_pred):
    return positivePredictiveValue(
        y_true, y_pred) + negativePredictiveValue(y_true, y_pred) - 1


def accuracy(y_true, y_pred):
    pred = K.argmax(y_pred)
    true = K.cast(K.squeeze(y_true, axis=-1), 'int64')

    mask = K.cast(K.not_equal(true, features.maskValue), 'int64')
    matches = K.cast(K.equal(true, pred), 'int64') * mask

    return K.sum(matches) / K.maximum(K.sum(mask), 1)


def balancedAccuracy(y_true, y_pred):
    return 0.5 * (truePositiveRate(y_true, y_pred) +
                  trueNegativeRate(y_true, y_pred))

# def prevelanceThreshold(y_true, y_pred):
#     TPR = truePositiveRate(y_true, y_pred)
#     TNR = specificity(y_true, y_pred)
#     return ((TPR * (1 - TNR))**(0.5) + TNR - 1) / (TPR + TNR - 1)


def masked_MSE(y_true, y_pred):
    mask = K.not_equal(y_true, features.maskValue)

    squared_error = tf.where(
        mask, tf.math.square(tf.subtract(y_true, y_pred)), 0)

    # return tf.divide(tf.reduce_sum(squared_error), K.cast(tf.math.count_nonzero(K.cast(mask, 'int32')), 'float64'))
    return squared_error


def masked_MAE(y_true, y_pred):
    mask = K.not_equal(y_true, features.maskValue)

    abs_error = tf.where(
        mask, tf.math.abs(tf.subtract(y_true, y_pred)), 0)

    return abs_error
    # return tf.divide(tf.reduce_sum(abs_error), K.cast(tf.math.count_nonzero(K.cast(mask, 'int32')), 'float64'))


def masked_PseudoHuber(y_true, y_pred):
    mask = K.not_equal(y_true, features.maskValue)

    error = tf.where(mask, tf.subtract(y_true, y_pred), 0)

    huber = tf.math.subtract(tf.math.sqrt(
        tf.math.add(K.cast(1.0, 'float64'), tf.math.square(error))), K.cast(1.0, 'float64'))

    return huber


def masked_Huber(y_true, y_pred):
    mask = K.not_equal(y_true, features.maskValue)

    error = tf.where(mask, tf.abs(tf.subtract(y_true, y_pred)), 0)

    delta = K.cast(1.0, 'float64')

    huber = tf.where(tf.abs(error) > delta,
                     tf.add(K.cast(0.5*delta**2, 'float64'), tf.multiply(delta,
                                                                         tf.subtract(error, delta))),
                     tf.multiply(K.cast(0.5, 'float64'), tf.square(error)))

    return huber


def masked_sparse_categorical_crossentropy(y_true, y_pred):

    mask = K.not_equal(K.squeeze(y_true, axis=-1), features.maskValue)

    scce = tf.where(
        mask, tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred), 0)

    return scce


def negloglik(y_true, y_pred):
    mask = K.not_equal(y_true, features.maskValue)
    log_prob = -y_pred.log_prob(y_true)

    mask = tf.reshape(mask, [-1])
    nonzero = tf.math.count_nonzero(mask)

    return K.sum(tf.where(mask, log_prob, 0)) / tf.cast(nonzero, 'float64')


def get_dense(units, activation, regularizer, regularizer_rate, max_norm,
              bayesian=False):
    if bayesian:
        return tfp.layers.DenseVariational(units,
                                           make_prior_fn=prior, make_posterior_fn=posterior,
                                           kl_weight=1 /
                                           (20 * params.batch_size),
                                           activation='sigmoid')
    else:
        if regularizer == 'l1':
            regularizer = tf.keras.regularizers.l1(regularizer_rate)
        elif regularizer == 'l2':
            regularizer = tf.keras.regularizers.l2(regularizer_rate)
        elif regularizer == 'l1l2':
            regularizer = tf.keras.regularizers.L1L2(regularizer_rate)

        return tf.keras.layers.Dense(units, activation=activation,
                                     # activity_regularizer=regularizer,
                                     kernel_regularizer=regularizer,
                                     # bias_regularizer=regularizer,
                                     kernel_constraint=tf.keras.constraints.max_norm(
                                         max_norm),
                                     bias_constraint=tf.keras.constraints.max_norm(max_norm))


def build_base_model(inputs, num_shared_layers, regularizer, regularizer_rate, max_norm,
                     dropout, activation, units_per_layer,
                     bayesian=False):

    baseModel = None
    for i in range(num_shared_layers):
        if(i == 0):
            baseModel = get_dense(units_per_layer, activation,
                                  regularizer, regularizer_rate, max_norm,
                                  bayesian=bayesian)(inputs)
        else:
            baseModel = get_dense(units_per_layer, activation,
                                  regularizer, regularizer_rate,  max_norm,
                                  bayesian=bayesian)(baseModel)
        if dropout > 0:
            if i < num_shared_layers - 1:
                baseModel = tf.keras.layers.Dropout(dropout)(baseModel)
            # else:
            #     baseModel = tf.keras.layers.Dropout(dropout / 2)(baseModel)
    return baseModel


def build_ensemble(feature, ensemble_size, num_layers, units_layer,
                   max_norm, activation, regularizer, regularizer_rate, dropout, inputs,
                   bayesian=False):

    ensemble = []
    for m in range(ensemble_size):
        x = None
        for i in range(num_layers):
            if(i == 0):
                x = get_dense(units_layer, activation, regularizer, regularizer_rate,
                              max_norm, bayesian=bayesian)(inputs)
            else:
                x = get_dense(units_layer, activation, regularizer, regularizer_rate,
                              max_norm, bayesian=bayesian)(x)
            if dropout > 0:
                if i < num_layers - 1:
                    x = tf.keras.layers.Dropout(dropout)(x)
                else:
                    x = tf.keras.layers.Dropout(dropout / 5)(x)

        if feature == 'GFA':
            if ensemble_size > 1:
                ensemble.append(tf.keras.layers.Dense(
                    3, activation='softmax',
                    name=feature + '_' + str(m))(x))
            else:
                ensemble.append(tf.keras.layers.Dense(
                    3, activation='softmax',
                    name=feature)(x))
        else:
            if ensemble_size > 1:
                if bayesian:
                    ensemble.append(tf.keras.layers.Dense(
                        1, activation='softplus',
                        name=feature + '_' + str(m))(x))
                else:

                    ensemble.append(tf.keras.layers.Dense(
                        1, activation='softplus', name=feature + '_' + str(m))(x))
            else:
                if bayesian:
                    distribution_params = tf.keras.layers.Dense(2)(x)
                    ensemble.append(tfp.layers.IndependentNormal(
                        1, name=feature)(distribution_params))
                else:
                    ensemble.append(tf.keras.layers.Dense(
                        1, activation='softplus', name=feature)(x))
    return ensemble


def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = tf.keras.Sequential(
        [
            # tfp.layers.VariableLayer(
            # tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            # ),
            # tfp.layers.MultivariateNormalTriL(n),
            # tfp.layers.DistributionLambda(
            #   lambda t: tfp.distributions.MultivariateNormalDiag(
            #        loc=tf.zeros(n, dtype=dtype), scale_diag=tf.ones(n, dtype=dtype)
            #    )
            # )
            tfp.layers.VariableLayer(n, dtype=dtype),
            tfp.layers.DistributionLambda(lambda t: tfd.Independent(
                tfd.Normal(loc=t, scale=1),
                reinterpreted_batch_ndims=1)),
        ]
    )
    return prior_model


def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.))
    posterior_model = tf.keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
            # tfp.layers.VariableLayer(2 * n, dtype=dtype),
            # tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            #     tfd.Normal(loc=t[..., :n],
            #                scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
            #     reinterpreted_batch_ndims=1)),
        ]
    )
    return posterior_model


def setup_losses(bayesian=False):
    losses = {}
    metrics = {}
    lossWeights = {}

    for feature in features.predictableFeatures:
        if feature in ['Tl', 'Tg', 'Tx', 'deltaT', 'Dmax']:
            metrics[feature] = [masked_MSE, masked_MAE,
                                masked_Huber, masked_PseudoHuber]
        elif feature == 'GFA':
            metrics[feature] = [accuracy, truePositiveRate, trueNegativeRate, positivePredictiveValue,
                                negativePredictiveValue, balancedAccuracy, f1_score, informedness, markedness]

        if bayesian:
            losses[feature] = negloglik
        else:
            if feature in ['Tl', 'Tg', 'Tx', 'deltaT', 'Dmax']:
                losses[feature] = masked_Huber  # masked_MSE
            elif feature == 'GFA':
                losses[feature] = masked_sparse_categorical_crossentropy

        # if feature in ['Tg', 'Tx']:
        #     lossWeights[feature] = 0.0001
        # elif feature == 'Tl':
        #     lossWeights[feature] = 0.0001
        # elif feature == 'deltaT':
        #     lossWeights[feature] = 0.001
        # elif feature == 'Dmax':
        #     lossWeights[feature] = 10
        # elif feature == 'GFA':
        #     lossWeights[feature] = 0.5

        if feature in ['Tg', 'Tx']:
            lossWeights[feature] = 1
        elif feature == 'Tl':
            lossWeights[feature] = 0.1
        elif feature == 'Dmax' or feature == 'deltaT':
            lossWeights[feature] = 100
        elif feature == 'GFA':
            lossWeights[feature] = 1

    print("LOSS WEIGHTS", lossWeights)

    # maxWeight = max(list(lossWeights.values()))
    # for feature in lossWeights:
    #     lossWeights[feature] /= maxWeight

    return losses, lossWeights, metrics


def build_input_layers(train_features):
    inputs = []
    for feature in train_features.columns:
        inputs.append(
            tf.keras.Input(
                shape=(1,),
                name=feature,
                dtype='float64')
        )

    return inputs


class MaskedNormalise(tf.keras.layers.Layer):
    def __init__(self, mask_value=-1, **kwargs):
        super(MaskedNormalise, self).__init__(**kwargs)
        self.mask_value = mask_value

    def call(self, inputs):
        mask = K.not_equal(inputs, self.mask_value)

        masked_max = K.max(tf.where(mask, inputs, 0))
        masked_min = K.min(tf.where(mask, inputs, np.Inf))

        normalised = tf.where(mask, tf.divide(tf.subtract(inputs, masked_min),
                                              tf.subtract(masked_max, masked_min)), -1)

        return normalised

    def get_config(self):
        config = {'mask_value': self.mask_value}
        base_config = super(MaskedNormalise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def build_model(train_features, train_labels, num_shared_layers,
                num_specific_layers, units_per_layer, regularizer="l2",
                regularizer_rate=0.001, dropout=0.3, learning_rate=0.01,
                ensemble_size=1, activation="elu", max_norm=3, bayesian=False):

    inputs = build_input_layers(train_features)

    normalized_inputs = []
    for input_layer in inputs:
        normalizer = tf.keras.layers.experimental.preprocessing.Normalization(
            axis=None)
        normalizer.adapt(train_features[input_layer.name])
        normalized_inputs.append(
            normalizer(input_layer)
        )
        # normalized_inputs.append(
        #    MaskedNormalise(mask_value=features.maskValue)(input_layer)
        # )

    concatenated_inputs = tf.keras.layers.concatenate(
        normalized_inputs, name="Inputs")

    if num_shared_layers > 0 and len(features.predictableFeatures) > 1:
        baseModel = build_base_model(concatenated_inputs,
                                     num_shared_layers, regularizer, regularizer_rate, max_norm,
                                     dropout, activation,
                                     units_per_layer, bayesian=False)

    else:
        baseModel = concatenated_inputs

    # baseModel = tf.keras.layers.LayerNormalization()(baseModel)

    losses, lossWeights, metrics = setup_losses(bayesian=bayesian)
    outputs = []
    normalized_outputs = []

    for feature in features.predictableFeatures:

        if len(outputs) > 0:
            model_branch = tf.keras.layers.concatenate(
                [baseModel] + normalized_outputs)
        else:
            model_branch = baseModel

        ensemble = build_ensemble(feature, ensemble_size, num_specific_layers,
                                  units_per_layer, max_norm, activation,
                                  regularizer, regularizer_rate, dropout, model_branch, bayesian=bayesian)

        if(len(ensemble) > 1):
            outputs.append(tf.keras.layers.average(ensemble, name=feature))
        else:
            outputs.append(ensemble[0])

        normalized_outputs.append(
            tf.keras.layers.LayerNormalization()(outputs[-1])
            # outputs[-1]
        )

    if "Tx" in features.predictableFeatures and "Tg" in features.predictableFeatures and 'deltaT' not in features.predictableFeatures:
        outputs.append(tf.keras.layers.Subtract(name="deltaT")([
            outputs[features.predictableFeatures.index("Tx")],
            outputs[features.predictableFeatures.index("Tg")]
        ]))

    # optimiser = tf.keras.optimizers.SGD(
    #     learning_rate=learning_rate,
    #     momentum=0.9,
    #     nesterov=True
    # )

    optimiser = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(loss=losses, metrics=metrics, loss_weights=lossWeights,
                  optimizer=optimiser, run_eagerly=False)

    tf.keras.utils.plot_model(
        model, to_file=params.image_directory + 'model.png', rankdir='LR')
    return model


def save(model, path):
    model.save(path)


def load(path):
    return tf.keras.models.load_model(path, custom_objects={
        'accuracy': accuracy,
        'truePositiveRate': truePositiveRate,
        'trueNegativeRate': trueNegativeRate,
        'positivePredictiveValue': positivePredictiveValue,
        'negativePredictiveValue': negativePredictiveValue,
        'balancedAccuracy': balancedAccuracy,
        'f1_score': f1_score,
        'informedness': informedness,
        'markedness': markedness,
        'masked_MSE': masked_MSE,
        'masked_MAE': masked_MAE,
        'masked_Huber': masked_Huber,
        'masked_PseudoHuber': masked_PseudoHuber,
        'masked_sparse_categorical_crossentropy': masked_sparse_categorical_crossentropy,
        'negloglik': negloglik,
    })


def load_weights(model, path):
    model.load_weights(path + "/model")


def compile_and_fit(train_features, train_labels, sampleWeight,
                    test_features=None, test_labels=None,
                    sampleWeight_test=None, maxEpochs=1000,
                    bayesian=False):

    model = build_model(
        train_features,
        train_labels,
        num_shared_layers=3,  # 7,
        num_specific_layers=5,
        units_per_layer=64,
        regularizer='l2',
        regularizer_rate=0.001,
        dropout=0.1,
        learning_rate=1e-2,
        # learning_rate=tf.keras.optimizers.schedules.InverseTimeDecay(
        #     0.01,
        #     decay_steps=1,
        #     decay_rate=0.01,
        #     staircase=False),
        activation='elu',
        max_norm=5.0,
        ensemble_size=1,
        bayesian=bayesian
    )

    return fit(model, train_features, train_labels, sampleWeight,
               test_features, test_labels, sampleWeight_test, maxEpochs)


def fit(model, train_features, train_labels, sampleWeight, test_features=None,
        test_labels=None, sampleWeight_test=None, maxEpochs=1000):
    patience = 100
    min_delta = 0.001

    xTrain = {}
    for feature in train_features:
        xTrain[feature] = train_features[feature]

    yTrain = {}
    for feature in features.predictableFeatures:
        if feature in train_labels:
            yTrain[feature] = train_labels[feature]

    monitor = "loss"

    testData = None
    if test_features is not None:
        xTest = {}
        for feature in test_features:
            xTest[feature] = test_features[feature]

        yTest = {}
        for feature in features.predictableFeatures:
            if feature in test_labels:
                yTest[feature] = test_labels[feature]

        testData = (xTest, yTest, sampleWeight_test)

        monitor = "val_loss"

    history = model.fit(
        x=xTrain,
        y=yTrain,
        batch_size=params.batch_size,
        epochs=maxEpochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=patience,
                min_delta=min_delta,
                mode="auto",
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=monitor,
                factor=0.5,
                patience=patience // 3,
                mode="auto",
                min_delta=min_delta * 10,
                cooldown=patience // 4,
                min_lr=0
            ),
            tf.keras.callbacks.TensorBoard(log_dir=params.output_directory+"/logs/fit/" +
                                           datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), histogram_freq=1)
        ],
        sample_weight=sampleWeight,
        validation_data=testData,
        verbose=2
    )
    return model, history


def train_model(train_features, train_labels, sampleWeight,
                test_features=None, test_labels=None,
                sampleWeight_test=None, plot=True, maxEpochs=1000,
                bayesian=False, model_name=None):

    model, history = compile_and_fit(train_features, train_labels,
                                     sampleWeight,
                                     test_features=test_features,
                                     test_labels=test_labels,
                                     sampleWeight_test=sampleWeight_test,
                                     maxEpochs=maxEpochs,
                                     bayesian=bayesian)

    if plot:
        plots.plot_training(history, model_name=model_name)
        if model_name is not None:
            save(model, params.output_directory + '/model_'+str(model_name))
        else:
            save(model, params.output_directory + '/model')
    return model


def evaluate_model(model, train_ds, train_labels, test_ds=None,
                   test_labels=None, plot=True,
                   train_compositions=None, test_compositions=None,
                   bayesian=False, model_name=None):

    train_errorbars = None
    test_errorbars = None

    train_predictions = []
    test_predictions = []

    predictionNames = features.getModelPredictionFeatures(model)

    if bayesian:

        train_errorbars = []

        for i in range(len(train_labels.columns)):
            train_predictions.append([])
            train_errorbars.append([])

        for b in range(len(list(train_ds))):
            train_inputs, tmp_train_labels = list(train_ds)[b]
            train_distribution = model(train_inputs)

            for i in range(len(train_distribution)):

                train_predictions[i].extend(
                    train_distribution[i].mean().numpy().flatten())
                train_errorbars[i].extend(
                    (1.96 * train_distribution[i].stddev().numpy()).flatten())

        if test_ds is not None:
            test_errorbars = []

            for i in range(len(test_labels.columns)):
                test_predictions.append([])
                test_errorbars.append([])

            for b in range(len(list(test_ds))):
                test_inputs, tmp_test_labels = list(test_ds)[b]
                test_distribution = model(test_inputs)

                for i in range(len(test_distribution)):
                    test_predictions[i].extend(
                        test_distribution[i].mean().numpy().flatten())
                    test_errorbars[i].extend(
                        (1.96 * test_distribution[i].stddev().numpy()).flatten())

    else:
        train_predictions = model.predict(train_ds)

        if len(predictionNames) == 1:
            if predictionNames[0] != 'GFA':
                train_predictions = [train_predictions.flatten()]
        else:
            for i in range(len(train_predictions)):
                if predictionNames[i] != 'GFA':
                    train_predictions[i] = train_predictions[i].flatten()

        if test_ds:
            test_predictions = model.predict(test_ds)
            if len(predictionNames) == 1:
                if predictionNames[0] != 'GFA':
                    test_predictions = [test_predictions.flatten()]
            else:
                for i in range(len(test_predictions)):
                    if predictionNames[i] != 'GFA':
                        test_predictions[i] = test_predictions[i].flatten()

    if plot:
        if test_ds is not None:
            plots.plot_results_classification(
                train_labels, train_predictions, test_labels, test_predictions, model_name=model_name)
            plots.plot_results_regression(train_labels,
                                          train_predictions,
                                          test_labels,
                                          test_predictions,
                                          train_compositions=train_compositions,
                                          test_compositions=test_compositions,
                                          train_errorbars=train_errorbars,
                                          test_errorbars=test_errorbars,
                                          model_name=model_name)
        else:
            plots.plot_results_classification(
                train_labels, train_predictions, model_name=model_name)
            plots.plot_results_regression(train_labels,
                                          train_predictions,
                                          train_compositions=train_compositions,
                                          test_compositions=test_compositions,
                                          train_errorbars=train_errorbars, model_name=model_name)

    if test_ds is not None:
        return train_predictions, test_predictions
    else:
        return train_predictions
