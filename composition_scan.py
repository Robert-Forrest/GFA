import re
import plots
import neuralnets
import features
import params


def run(compositions=None):

    if compositions is None:
        print("Error: No compositions entered")
        exit()
    else:
        for i in range(len(compositions)):
            compositions[i] = re.findall('[A-Z][^A-Z]*', compositions[i])

    model_name = 'kfoldsEnsemble'
    onlyPredictions = True
    plotExamples = True

    params.setup(model_name, existingModel=True)

    model = neuralnets.load(params.output_directory + '/model')

    inspect_features = ['percentage', 'mixing_Gibbs_free_energy', 'PHSS', 'mixing_entropy', 'mixing_enthalpy',
                        'wigner_seitz_electron_density_deviation', 'series_deviation', 'radius_deviation', 'density_linearmix', 'p_valence', 'd_valence']
    additionalFeatures = [
        'price', 'wigner_seitz_electron_density', 'mixing_enthalpy', 'mixing_Gibbs_free_energy', 'radius', 'p_valence', 'd_valence']

    if plotExamples:
        originalData = features.load_data(
            model=model, plot=False, dropCorrelatedFeatures=False,  additionalFeatures=additionalFeatures)
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
