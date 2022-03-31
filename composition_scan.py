import re

import cerebral as cb

import data
import plots

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

    inspect_features = cb.conf.x_features
    additionalFeatures = cb.conf.y_features

    if plotExamples:
        originalData = cb.io.load_data(
            model=model, plot=False,
            dropCorrelatedFeatures=False,
            additionalFeatures=additionalFeatures,
            postprocess=data.ensure_default_values)
    else:
        originalData = None

    for composition in compositions:
        if len(composition) == 2:
            plots.plot_binary(composition, model, onlyPredictions=onlyPredictions,
                                 originalData=originalData, inspect_features=inspect_features, additionalFeatures=additionalFeatures)
        elif len(composition) == 3:
            plots.plot_ternary(composition, model, onlyPredictions=onlyPredictions,
                                  originalData=originalData, additionalFeatures=additionalFeatures)
        elif len(composition) == 4:
            plots.plot_quaternary(composition, model, onlyPredictions=onlyPredictions,
                                     originalData=originalData, additionalFeatures=additionalFeatures)
