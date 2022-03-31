import re

import cerebral as cb
import eyeglass as eg

import data


def run(compositions=None):

    if compositions is None:
        print("Error: No compositions entered")
        exit()
    else:
        for i in range(len(compositions)):
            compositions[i] = re.findall('[A-Z][^A-Z]*', compositions[i])

    onlyPredictions = True
    plotExamples = True

    model = cb.models.load(cb.conf.output_directory + '/model')

    x_features = cb.conf.x_features
    y_features = cb.conf.y_features

    if plotExamples:
        originalData = cb.io.load_data(
            model=model, plot=False,
            dropCorrelatedFeatures=False,
            additionalFeatures=y_features,
            postprocess=data.ensure_default_values)
    else:
        originalData = None

    for composition in compositions:
        if len(composition) == 2:
            eg.plots.plot_binary(
                composition, model,
                originalData=originalData,
                x_features=x_features,
                y_features=y_features)

        elif len(composition) == 3:
            eg.plots.plot_ternary(
                composition, model,
                originalData=originalData,
                y_features=y_features)

        elif len(composition) == 4:
            eg.plots.plot_quaternary(
                composition, model,
                onlyPredictions=onlyPredictions,
                originalData=originalData,
                y_features=y_features)
