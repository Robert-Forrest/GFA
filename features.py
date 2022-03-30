import re
import os
import json
from collections import defaultdict, OrderedDict
import numpy as np  # pylint: disable=import-error
import scipy.stats as stats
from scipy.cluster import hierarchy
import pandas as pd  # pylint: disable=import-error
import tensorflow as tf  # pylint: disable=import-error
from sklearn.feature_selection import VarianceThreshold  # pylint: disable=import-error
from sklearn import preprocessing
from sklearn.utils.class_weight import compute_class_weight
from decimal import Decimal

import elementy
import metallurgy as mg

import params
import extra_data
import plots

predictableFeatures = ['Tl', 'Tg', 'Tx', 'deltaT', 'GFA', 'Dmax']

maskValue = -1
idealGasConstant = 8.31

units = {
    'Dmax': 'mm',
    'Tl': 'K',
    'Tg': 'K',
    'Tx': 'K',
    'deltaT': 'K',
    'price_linearmix': "\\$/kg",
    'price': "\\$/kg",
    'mixing_enthalpy': 'kJ/mol',
    'mixing_Gibbs_free_energy': 'kJ/mol'
}
inverse_units = {}
for feature in units:
    if "/" not in units[feature]:
        inverse_units[feature] = "1/" + units[feature]
    else:
        split_units = units[feature].split('/')
        inverse_units[feature] = split_units[1] + "/" + split_units[0]


def load_data(calculate_extra_features=True, use_composition_vector=False, plot=True,
              dropCorrelatedFeatures=True, model=None, tmp=False, additionalFeatures=[]):
    global predictableFeatures

    if not os.path.exists(
            './data/calculated_features.csv') or model is not None or tmp:
        # data_files = ['data.csv', 'BMG_dataset_Yohan.xlsx']
        data_files = ['data.csv']
        # data_files = ['BMG_dataset_Yohan.xlsx']

        data = []
        for data_file in data_files:
            if '.csv' in data_file:
                rawData = pd.read_csv('./data/' + data_file)
            elif '.xls' in data_file:
                rawData = pd.read_excel('./data/' + data_file, 'CD')
                if 'deltaT' in predictableFeatures:
                    rawData = pd.concat([rawData, pd.read_excel(
                        './data/' + data_file, 'SLR')])

            rawData = rawData.loc[:, ~rawData.columns.str.contains('^Unnamed')]

            if 'composition' not in rawData:
                rawData = calculate_compositions(rawData)

            data.append(rawData)

        data = pd.concat(data, ignore_index=True)

        if model is None:
            data = calculate_features(data, calculate_extra_features=calculate_extra_features,
                                      use_composition_vector=use_composition_vector, plot=plot,
                                      dropCorrelatedFeatures=dropCorrelatedFeatures, additionalFeatures=additionalFeatures)
        else:
            modelInputs = []
            for inputLayer in model.inputs:
                modelInputs.append(inputLayer.name)
            data = calculate_features(data, requiredFeatures=modelInputs, calculate_extra_features=calculate_extra_features,
                                      use_composition_vector=use_composition_vector, plot=plot,
                                      dropCorrelatedFeatures=dropCorrelatedFeatures, additionalFeatures=additionalFeatures)

        data = data.fillna(maskValue)

        if 'GFA' in data:
            data['GFA'] = data['GFA'].astype('int64')

        if not tmp:
            data.to_csv('./data/calculated_features.csv')
    else:
        data = pd.read_csv('./data/calculated_features.csv')
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

    if 'deltaT' not in predictableFeatures and 'deltaT' in data:
        data = data.drop('deltaT', axis='columns')

    newPredictableFeatures = []
    for feature in predictableFeatures:
        if feature in data:
            newPredictableFeatures.append(feature)

    predictableFeatures = newPredictableFeatures

    return data


def train_test_split(data, trainPercentage=0.75):
    data = data.copy()

    unique_composition_spaces = {}
    for _, row in data.iterrows():
        composition = mg.alloy.parse_composition(row['composition'])
        sorted_composition = sorted(list(composition.keys()))
        composition_space = "".join(sorted_composition)

        if composition_space not in unique_composition_spaces:
            unique_composition_spaces[composition_space] = []

        unique_composition_spaces[composition_space].append(row)

    numTraining = np.ceil(
        int(trainPercentage * len(unique_composition_spaces)))

    trainingSet = []
    testSet = []

    shuffled_unique_compositions = list(unique_composition_spaces.keys())
    np.random.shuffle(shuffled_unique_compositions)

    for i in range(len(shuffled_unique_compositions)):
        compositions = unique_composition_spaces[shuffled_unique_compositions[i]]
        if i < numTraining:
            trainingSet.extend(compositions)
        else:
            testSet.extend(compositions)

    return pd.DataFrame(trainingSet), pd.DataFrame(testSet)


def getModelPredictionFeatures(model):
    predictions = []
    for node in model.outputs:
        predictions.append(node.name.split('/')[0])
    return predictions


def calculate_compositions(data):
    compositions = []
    columns_to_drop = []
    for _, row in data.iterrows():
        composition = {}
        for column in data.columns:
            if column not in predictableFeatures:
                if column not in columns_to_drop:
                    columns_to_drop.append(column)
                if row[column] > 0:
                    composition[column] = row[column] / 100.0
        composition = mg.Alloy(composition)
        compositions.append(composition.to_string())

    data['composition'] = compositions
    for column in columns_to_drop:
        data = data.drop(column, axis='columns')
    return data


def camelCaseToSentence(string):
    if(string == string.upper()):
        return string
    else:
        tmp = re.sub(r'([A-Z][a-z])', r" \1",
                     re.sub(r'([A-Z]+)', r"\1", string))
        return tmp[0].upper() + tmp[1:]


def prettyName(feature):
    name = ""
    if feature not in ['Tl', 'Tx', 'Tg', 'Dmax', 'deltaT']:
        featureParts = feature.split('_')
        if len(featureParts) > 1:
            if featureParts[-1] == 'linearmix':
                name = r'$\Sigma$ '
            elif featureParts[-1] == 'reciprocalMix':
                name = r'$\Sigma^{-1}$ '
            elif featureParts[-1] == 'deviation':
                name = r'$\sigma$ '
            elif featureParts[-1] == 'deviation':
                name = r'$\delta$ '
            elif featureParts[-1] == "_percent":
                name = r'$\%$'

            name += " ".join(featureParts[0:-1])
    else:
        if feature == 'Tl':
            name = r'$T_l$'
        elif feature == 'Tg':
            name = r'$T_g$'
        elif feature == 'Tx':
            name = r'$T_x$'
        elif feature == 'Dmax':
            name = r'$D_{max}$'
        elif feature == 'deltaT':
            name = r'$\Delta T$'

    return name


droppedFeatures = []


def ensure_default_values(row, i, data):
    try:
        _ = data.at[i, 'Dmax']
        hasDmax = True
    except BaseException:
        hasDmax = False

    if(hasDmax):
        if not np.isnan(data.at[i, 'Dmax']):
            if row['Dmax'] == 0:
                data.at[i, 'GFA'] = 0
            elif row['Dmax'] <= 0.15:
                data.at[i, 'GFA'] = 1
            else:
                data.at[i, 'GFA'] = 2
        else:
            data.at[i, 'Dmax'] = maskValue
    else:
        data.at[i, 'Dmax'] = maskValue

    try:
        _ = data.at[i, 'GFA']
        hasGFA = True
    except BaseException:
        hasGFA = False

    if(hasGFA):
        if not np.isnan(data.at[i, 'GFA']):
            if(int(data.at[i, 'GFA']) == 0):
                data.at[i, 'Dmax'] = 0
            elif(int(data.at[i, 'GFA']) == 1):
                data.at[i, 'Dmax'] = 0.15
            elif(int(data.at[i, 'GFA']) == 2):
                if('Dmax' in row):
                    if(np.isnan(data.at[i, 'Dmax']) or data.at[i, 'Dmax'] == 0 or data.at[i, 'Dmax'] is None):
                        data.at[i, 'Dmax'] = maskValue
                else:
                    data.at[i, 'Dmax'] = maskValue
            else:
                data.at[i, 'Dmax'] = maskValue
        else:
            data.at[i, 'GFA'] = maskValue
    else:
        data.at[i, 'GFA'] = maskValue

    if 'Tx' in row and 'Tg' in row:
        if not np.isnan(row['Tx']) and not np.isnan(row['Tg']):
            data.at[i, 'deltaT'] = row['Tx'] - row['Tg']


def calculate_features(data, calculate_extra_features=True, use_composition_vector=False,
                       dropCorrelatedFeatures=True, plot=False, additionalFeatures=[], requiredFeatures=[], merge_duplicates=True):

    global droppedFeatures

    # calculate_extra_features = True
    # use_composition_vector = False
    if not calculate_extra_features:
        dropCorrelatedFeatures = False

    basicFeatures = ['atomic_number', 'periodic_number', 'mass', 'group',
                     'radius', 'atomic_volume',
                     'period', 'protons', 'neutrons', 'electrons', 'valence_electrons',
                     'valence', 'electron_affinity', 'ionisation_energies',
                     'wigner_seitz_electron_density', 'work_function',
                     'mendeleev_universal_sequence', 'chemical_scale',
                     'mendeleev_pettifor', 'mendeleev_modified', 'electronegativity_pauling',
                     'electronegativity_miedema', 'electronegativity_mulliken', 'melting_temperature',
                     'boiling_temperature', 'fusion_enthalpy',
                     'vaporisation_enthalpy', 'molar_heat_capacity',
                     'thermal_conductivity', 'thermal_expansion',
                     'density', 'cohesive_energy', 'debye_temperature',
                     'chemical_hardness', 'chemical_potential']

    complexFeatures = ['theoreticalDensity', 'atomic_volume_deviation',
                       's_valence', 'p_valence', 'd_valence', 'f_valence',
                       'structure_deviation', 'ideal_entropy',
                       'ideal_entropy_xia', 'mismatch_entropy',
                       'mixing_entropy', 'mixing_enthalpy',
                       'mixing_Gibbs_free_energy',
                       'block_deviation', 'series_deviation', 'viscosity',
                       'lattice_distortion', 'EsnPerVec', 'EsnPerMn',
                       'mismatch_PHS', 'mixing_PHS', 'PHSS', 'price']

    for additional in additionalFeatures:
        if additional not in basicFeatures and additional not in complexFeatures:
            basicFeatures.append(additional)

    if len(requiredFeatures) > 0:
        dropCorrelatedFeatures = False

        for feature in requiredFeatures:
            if feature.endswith("_percent"):
                use_composition_vector = True

            elif "_linearmix" in feature:
                calculate_extra_features = True
                actualFeature = feature.split("_linearmix")[0]
                if actualFeature not in basicFeatures and actualFeature not in complexFeatures and feature not in complexFeatures:
                    basicFeatures.append(actualFeature)

            elif "_deviation" in feature:
                calculate_extra_features = True
                actualFeature = feature.split("_deviation")[0]
                if actualFeature not in basicFeatures and actualFeature not in complexFeatures and feature not in complexFeatures:
                    basicFeatures.append(actualFeature)

            else:
                calculate_extra_features = True
                if feature not in complexFeatures:
                    complexFeatures.append(feature)

    # compositionPercentages = {}
    # for element in elementData:
    #     if element not in compositionPercentages:
    #         compositionPercentages[element] = []

    featureValues = {}
    complexFeatureValues = {}

    for feature in basicFeatures:
        featureValues[feature] = {
            'linearmix': [],
            'deviation': []
        }
        units[feature + '_deviation'] = "%"

    for feature in complexFeatures:
        complexFeatureValues[feature] = []

    if('GFA' in data.columns):
        data['GFA'] = data['GFA'].map({'Crystal': 0, 'Ribbon': 1, 'BMG': 2})
        data['GFA'] = data['GFA'].fillna(maskValue)
        data['GFA'] = data['GFA'].astype(np.int64)

    import cProfile
    import io
    import pstats
    from pstats import SortKey
    pr = cProfile.Profile()
    pr.enable()

    for i, row in data.iterrows():

        composition = mg.alloy.parse_composition(row['composition'])

        ensure_default_values(row, i, data)

        if use_composition_vector:
            for element in compositionPercentages:
                if element in composition:
                    compositionPercentages[element].append(
                        composition[element])
                else:
                    compositionPercentages[element].append(0)

        if calculate_extra_features:
            for feature in basicFeatures:

                if 'linearmix' in featureValues[feature]:
                    featureValues[feature]['linearmix'].append(
                        mg.linear_mixture(composition, feature))

                if 'deviation' in featureValues[feature]:
                    featureValues[feature]['deviation'].append(
                        mg.deviation(composition, feature))

            if 'theoreticalDensity' in complexFeatureValues:
                complexFeatureValues['theoreticalDensity'].append(
                    mg.density.calculate_theoretical_density(composition))

            if 'sValence' in complexFeatureValues:
                complexFeatureValues['sValence'].append(
                    mg.valence.calculate_valence_proportion(composition, 's'))
            if 'pValence' in complexFeatureValues:
                complexFeatureValues['pValence'].append(
                    mg.valence.calculate_valence_proportion(composition, 'p'))
            if 'dValence' in complexFeatureValues:
                complexFeatureValues['dValence'].append(
                    mg.valence.calculate_valence_proportion(composition, 'd'))
            if 'fValence' in complexFeatureValues:
                complexFeatureValues['fValence'].append(
                    mg.valence.calculate_valence_proportion(composition, 'f'))

            if 'ideal_entropy' in complexFeatureValues:
                complexFeatureValues['ideal_entropy'].append(
                    mg.entropy.calculate_ideal_entropy(composition))

            if 'ideal_entropy_Xia' in complexFeatureValues:
                complexFeatureValues['ideal_entropy_Xia'].append(
                    mg.entropy.calculate_ideal_entropy_xia(composition))

            if 'mismatch_entropy' in complexFeatureValues:
                complexFeatureValues['mismatch_entropy'].append(
                    mg.entropy.calculate_mismatch_entropy(composition))

            if 'mixing_entropy' in complexFeatureValues:
                complexFeatureValues['mixing_entropy'].append(
                    mg.entropy.calculate_mixing_entropy(composition))

            if 'structure_deviation' in complexFeatureValues:
                complexFeatureValues['structure_deviation'].append(
                    mg.structures.calculate_structure_mismatch(composition))

            if 'block_deviation' in complexFeatureValues:
                complexFeatureValues['block_deviation'].append(
                    mg.deviation(composition, 'block'))

            if 'series_deviation' in complexFeatureValues:
                complexFeatureValues['series_deviation'].append(
                    mg.deviation(composition, 'series'))

            if 'mixing_enthalpy' in complexFeatureValues:
                complexFeatureValues['mixing_enthalpy'].append(
                    mg.enthalpy.calculate_mixing_enthalpy(composition))

            if 'price' in complexFeatureValues:
                complexFeatureValues['price'].append(
                    mg.price.calculate_price(composition))

            if 'mixing_Gibbs_free_energy' in complexFeatureValues:
                complexFeatureValues['mixing_Gibbs_free_energy'].append(
                    mg.enthalpy.calculate_mixing_Gibbs_free_energy(composition,
                                                                   complexFeatureValues['mixing_enthalpy'][-1],
                                                                   featureValues['melting_temperature']['linearmix'][-1],
                                                                   complexFeatureValues['mixing_entropy'][-1]))

            if 'mismatch_PHS' in complexFeatureValues:
                complexFeatureValues['mismatch_PHS'].append(
                    mg.enthalpy.calculate_mismatch_PHS(
                        composition,
                        complexFeatureValues['mixing_enthalpy'][-1],
                        complexFeatureValues['mismatch_entropy'][-1])
                    )

            if 'mixing_PHS' in complexFeatureValues:
                complexFeatureValues['mismatch_PHS'].append(
                    mg.enthalpy.calculate_mixing_PHS(
                        composition,
                        complexFeatureValues['mixing_enthalpy'][-1],
                        complexFeatureValues['mixing_entropy'][-1])
                    )
                

            if 'PHSS' in complexFeatureValues:
                complexFeatureValues['PHSS'].append(
                    mg.enthalpy.calculate_mixing_PHS(
                        composition,
                        complexFeatureValues['mixing_enthalpy'][-1],
                        complexFeatureValues['mixing_entropy'][-1],
                        complexFeatureValues['mismatch_entropy'][-1])
                    )

            if 'viscosity' in complexFeatureValues:
                complexFeatureValues['viscosity'].append(
                    mg.viscosity.calculate_viscosity(composition, complexFeatureValues['mixing_enthalpy'][-1]))

            if 'radius_gamma' in complexFeatureValues:
                complexFeatureValues['radius_gamma'].append(
                    mg.radii.calculate_radius_gamma(composition))

            if 'lattice_distortion' in complexFeatureValues:
                complexFeatureValues['lattice_distortion'].append(
                    mg.radii.calculate_lattice_distortion(composition))

            if 'EsnPerVec' in complexFeatureValues:
                complexFeatureValues['EsnPerVec'].append(
                    mg.ratios.shell_to_valence_electron_concentration(
                        featureValues['period']['linearmix'][-1],
                        featureValues['valence_electrons']['linearmix'][-1])
                    )

            if 'EsnPerMn' in complexFeatureValues:
                complexFeatureValues['EsnPerMn'].append(
                    mg.ratios.shell_to_valence_electron_concentration(
                        featureValues['period']['linearmix'][-1],
                        featureValues['mendeleev_universal_sequence']['linearmix'][-1])
                    )


            if 'thermodynamic_factor' in complexFeatureValues:
                complexFeatureValues['thermodynamic_factor'].append(
                    mg.enthalpy.calculate_thermodynamic_factor(
                        featureValues['melting_temperature']['linearmix'][-1],
                        complexFeatureValues['mixing_entropy'][-1],
                        complexFeatureValues['mixing_enthalpy'][-1]
                    )
                )
                

    if use_composition_vector:
        for element in compositionPercentages:
            data[element + '_percent'] = compositionPercentages[element]

    if calculate_extra_features:
        for feature in featureValues:
            for kind in featureValues[feature]:
                if len(featureValues[feature][kind]) == len(data.index):
                    data[feature + '_' + kind] = featureValues[feature][kind]
        for feature in complexFeatures:
            if len(complexFeatureValues[feature]) == len(data.index):
                data[feature] = complexFeatureValues[feature]

    data = data.drop_duplicates()
    data = data.fillna(maskValue)

    if merge_duplicates:
        to_drop = []
        seen_compositions = []
        duplicate_compositions = {}
        for i, row in data.iterrows():
            composition = mg.Alloy(row['composition']).to_string()

            if(not mg.alloy.valid_composition(row['composition'])):
                print("Invalid composition:", row['composition'], i)
                to_drop.append(i)
            elif(composition in seen_compositions):
                if composition not in duplicate_compositions:
                    duplicate_compositions[composition] = [
                        data.iloc[seen_compositions.index(composition)]
                    ]
                duplicate_compositions[composition].append(row)
                to_drop.append(i)
            seen_compositions.append(composition)

        data = data.drop(to_drop)

        to_drop = []
        for i, row in data.iterrows():
            composition = mg.Alloy(row['composition']).to_string()

            if composition in duplicate_compositions:
                to_drop.append(i)

        data = data.drop(to_drop)

        deduplicated_rows = []
        for composition in duplicate_compositions:

            maxClass = -1
            for i in range(len(duplicate_compositions[composition])):
                maxClass = max(
                    [duplicate_compositions[composition][i]['GFA'], maxClass])

            averaged_features = {}
            num_contributions = {}
            for feature in duplicate_compositions[composition][0].keys():
                if feature != 'composition' and "_percent" not in feature:
                    averaged_features[feature] = 0
                    num_contributions[feature] = 0

            for i in range(len(duplicate_compositions[composition])):
                if duplicate_compositions[composition][i]['GFA'] == maxClass:
                    for feature in averaged_features:
                        if duplicate_compositions[composition][i][feature] != maskValue and not pd.isnull(
                                duplicate_compositions[composition][i][feature]):

                            averaged_features[feature] += duplicate_compositions[composition][i][feature]
                            num_contributions[feature] += 1

            for i in range(len(duplicate_compositions[composition])):
                for feature in averaged_features:
                    if num_contributions[feature] == 0:
                        if duplicate_compositions[composition][i][feature] != maskValue and not pd.isnull(
                                duplicate_compositions[composition][i][feature]):

                            averaged_features[feature] += duplicate_compositions[composition][i][feature]
                            num_contributions[feature] += 1

            for feature in averaged_features:
                if num_contributions[feature] == 0:
                    averaged_features[feature] = maskValue
                elif num_contributions[feature] > 1:
                    averaged_features[feature] /= num_contributions[feature]

            averaged_features['composition'] = composition
            for feature in duplicate_compositions[composition][0].keys():
                if "_percent" in feature:
                    averaged_features[feature] = duplicate_compositions[composition][0][feature]

            deduplicated_rows.append(
                pd.DataFrame(averaged_features, index=[0]))

        if(len(deduplicated_rows) > 0):
            deduplicated_data = pd.concat(deduplicated_rows, ignore_index=True)
            data = pd.concat([data, deduplicated_data], ignore_index=True)

    if plot:
        plots.plot_correlation(data)
        plots.plot_feature_variation(data)

    if dropCorrelatedFeatures:

        staticFeatures = []
        varianceCheckData = data.drop('composition', axis='columns')
        for feature in data.columns:
            if feature in predictableFeatures or "_percent" in feature:
                varianceCheckData = varianceCheckData.drop(
                    feature, axis='columns')

        quartileDiffusions = {}
        for feature in varianceCheckData.columns:

            Q1 = np.percentile(varianceCheckData[feature], 25)
            Q3 = np.percentile(varianceCheckData[feature], 75)

            coefficient = 0
            if np.abs(Q1 + Q3) > 0:
                coefficient = np.abs((Q3 - Q1) / (Q3 + Q1))
            quartileDiffusions[feature] = coefficient

            if coefficient < 0.1:
                staticFeatures.append(feature)

        print("Dropping static features:", staticFeatures)
        for feature in staticFeatures:
            varianceCheckData = varianceCheckData.drop(
                feature, axis='columns')

        correlation = np.array(varianceCheckData.corr())

        correlatedDroppedFeatures = []
        for i in range(len(correlation) - 1):
            if varianceCheckData.columns[i] not in correlatedDroppedFeatures:
                for j in range(i + 1, len(correlation)):
                    if varianceCheckData.columns[j] not in correlatedDroppedFeatures:
                        if np.abs(correlation[i][j]) >= params.correlation_threshold:

                            if sum(np.abs(correlation[i])) < sum(
                                    np.abs(correlation[j])):
                                print(varianceCheckData.columns[j],
                                      sum(np.abs(correlation[j])), "beats",
                                      varianceCheckData.columns[i],
                                      sum(np.abs(correlation[i])))
                                correlatedDroppedFeatures.append(
                                    varianceCheckData.columns[i])
                                break
                            else:
                                print(varianceCheckData.columns[i], sum(np.abs(correlation[i])),
                                      "beats", varianceCheckData.columns[j], sum(np.abs(correlation[j])))
                                correlatedDroppedFeatures.append(
                                    varianceCheckData.columns[j])

        droppedFeatures = staticFeatures + correlatedDroppedFeatures

    if len(droppedFeatures) > 0:
        for feature in droppedFeatures:
            if feature in data.columns:
                data = data.drop(feature, axis='columns')

        if plot:
            plots.plot_correlation(data, suffix="filtered")
            plots.plot_feature_variation(data, suffix="filtered")

    if len(requiredFeatures) > 0:

        for feature in data.columns:
            trueFeatureName = feature.split(
                '_linearmix')[0].split('_deviation')[0]
            if feature not in requiredFeatures and feature != 'composition' and feature not in predictableFeatures and trueFeatureName not in additionalFeatures:
                print("Dropping", feature)
                data = data.drop(feature, axis='columns')

    for i, row in data.iterrows():
        ensure_default_values(row, i, data)

    return data.copy()


def df_to_dataset(dataframe):
    dataframe = dataframe.copy()

    labelNames = []
    for feature in predictableFeatures:
        if feature in dataframe.columns:
            labelNames.append(feature)

    if len(labelNames) > 0:
        labels = pd.concat([dataframe.pop(x)
                            for x in labelNames], axis=1)
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    else:
        ds = tf.data.Dataset.from_tensor_slices(dict(dataframe))

    ds = ds.batch(params.batch_size)
    ds = ds.prefetch(params.batch_size)
    ds = ds.cache()

    return ds


def generate_sample_weights(labels, classWeights):
    sampleWeight = []
    for _, row in labels.iterrows():
        if 'GFA' in row:
            if row['GFA'] in [0, 1, 2]:
                sampleWeight.append(classWeights[int(row['GFA'])])
            else:
                sampleWeight.append(1)
        else:
            sampleWeight.append(1)
    return np.array(sampleWeight)


def create_datasets(data, train=[], test=[]):

    if (len(train) == 0):
        train = data.copy()

    train_ds = df_to_dataset(train)
    train_features = train.copy()
    train_labels = {}
    for feature in predictableFeatures:
        if feature in train_features:
            train_labels[feature] = train_features.pop(feature)
    train_labels = pd.DataFrame(train_labels)

    if 'GFA' in data:
        unique = pd.unique(data['GFA'])
        classes = [0, 1, 2]

        counts = data['GFA'].value_counts()
        numSamples = 0
        for c in classes:
            if c in counts:
                numSamples += counts[c]

        classWeights = []
        for c in classes:
            if c in counts:
                classWeights.append(numSamples / (2 * counts[c]))
            else:
                classWeights.append(1.0)
    else:
        classWeights = [1]

    sampleWeight = generate_sample_weights(train_labels, classWeights)

    if len(test) > 0:
        test_ds = df_to_dataset(test)
        test_features = test.copy()
        test_labels = {}
        for feature in predictableFeatures:
            if feature in test_features:
                test_labels[feature] = test_features.pop(feature)
        test_labels = pd.DataFrame(test_labels)

        sampleWeightTest = generate_sample_weights(test_labels, classWeights)

        return train_ds, test_ds, train_features, test_features, train_labels, test_labels, sampleWeight, sampleWeightTest
    else:
        return train_ds, train_features, train_labels, sampleWeight
