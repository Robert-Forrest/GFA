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
from urllib3.util.retry import Retry  # pylint: disable=import-error
import requests  # pylint: disable=import-error
from requests.adapters import HTTPAdapter  # pylint: disable=import-error

import params
import extra_data
import plots

predictableFeatures = ['Tl', 'Tg', 'Tx', 'deltaT', 'GFA', 'Dmax']

elementData = {}

maskValue = -1

units = {
    'Dmax': 'mm',
    'Tl': 'K',
    'Tg': 'K',
    'Tx': 'K',
    'deltaT': 'K',
    'price_linearMix': "\\$/kg",
    'price': "\\$/kg",
    'mixingEnthalpy': 'kJ/mol',
    'mixingGibbsFreeEnergy': 'kJ/mol'
}
inverse_units = {}
for feature in units:
    if "/" not in units[feature]:
        inverse_units[feature] = "1/" + units[feature]
    else:
        split_units = units[feature].split('/')
        inverse_units[feature] = split_units[1] + "/" + split_units[0]


def requests_session():
    session = requests.Session()
    retry = Retry(
        total=100,
        read=100,
        connect=100,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        method_whitelist=frozenset(['GET', 'PATCH', 'POST'])
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


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
        composition = parse_composition(row['composition'])
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
        composition = composition_to_string(composition)
        compositions.append(composition)

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
            if featureParts[1] == 'linearMix':
                name = r'$\Sigma$ '
            elif featureParts[1] == 'reciprocalMix':
                name = r'$\Sigma^{-1}$ '
            elif featureParts[1] == 'deviation':
                name = r'$\sigma$ '
            elif featureParts[1] == 'discrepancy':
                name = r'$\delta$ '
            elif featureParts[1] == "_percent":
                name = r'$\%$'

        name += camelCaseToSentence(featureParts[0])
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


def prettyComposition(composition):
    numbers = re.compile(r'(\d+)')
    return numbers.sub(r'$_{\1}$', composition_to_string(composition))


def mulliken(element, elementData):
    if('electronAffinity' in elementData[element] and 'ionisationEnergies' in elementData[element]):
        if elementData[element]['electronAffinity'] is not None:
            m = 0.5 * np.abs(elementData[element]['electronAffinity'] +
                             (elementData[element]['ionisationEnergies'][0] / 96.485))
            return m
    return None


def extract_data(composition, featureName):
    data = {}
    for element in composition:
        if(featureName in elementData[element]):
            if(elementData[element][featureName] is not None):
                data[element] = elementData[element][featureName]
        elif(featureName in elementData[element]['stpProperties']):
            if(elementData[element]['stpProperties'] is not None):
                if(elementData[element]['stpProperties'][featureName] is not None):
                    data[element] = elementData[element]['stpProperties'][featureName]
        elif(featureName in elementData[element]['electronegativity']):
            if(elementData[element]['electronegativity'] is not None):
                if(elementData[element]['electronegativity'][featureName] is not None):
                    data[element] = elementData[element]['electronegativity'][featureName]
        elif(featureName in elementData[element]['mendeleevNumber']):
            if(elementData[element]['mendeleevNumber'] is not None):
                if(elementData[element]['mendeleevNumber'][featureName] is not None):
                    data[element] = elementData[element]['mendeleevNumber'][featureName]

    return data


def linear_mixture(composition, featureName=None, data=None):
    if data is None:
        data = extract_data(composition, featureName)

    mixed_property = 0
    for element in composition:
        if(element in data):
            mixed_property += composition[element] * data[element]
        else:
            mixed_property = None
            break

    return mixed_property


def reciprocal_mixture(composition, featureName, data=None):
    if data is None:
        data = extract_data(composition, featureName)

    mixed_property = 0
    for element in composition:
        if(element in data):
            mixed_property += composition[element] / data[element]
        else:
            mixed_property = None
            break

    return 1.0 / mixed_property


def discrepancy(composition, data):
    if(len(composition) > 1):
        mean = 0
        for element in composition:
            if(element in data):
                mean += composition[element] * data[element]
            else:
                return None

        tmp = 0
        if np.abs(mean) != 0:
            for element in composition:
                tmp += composition[element] * \
                    ((1 - (data[element] / mean))**2)
            return np.sqrt(tmp)
        else:
            for element in composition:
                tmp += composition[element] * \
                    ((data[element] - mean)**2)
            return np.sqrt(tmp)

    else:
        return 0


def deviation(composition, data):
    if(len(composition) > 1):
        mean = 0
        for element in composition:
            if(element in data):
                mean += composition[element] * data[element]
            else:
                return None

        tmp = 0
        for element in composition:
            tmp += composition[element] * \
                ((data[element] - mean)**2)
        return np.sqrt(tmp)

#        return np.tmp([data[element] for element in data])

    else:
        return 0


def parse_composition(composition_string):

    composition = {}
    if('(' in composition_string):
        major_composition = composition_string.split(')')[0].split('(')[1]

        major_composition_percentage = float(re.split(
            r'(\d+(?:\.\d+)?)', composition_string.split(')')[1])[1]) / 100.0

        split_major_composition = re.findall(
            r'[A-Z][^A-Z]*', major_composition)

        for element_percentage in split_major_composition:
            split_element_percentage = re.split(
                r'(\d+(?:\.\d+)?)', element_percentage)
            composition[split_element_percentage[0]] = (float(
                split_element_percentage[1]) / 100.0) * major_composition_percentage

        minor_composition = composition_string.split(
            ')')[1][len(str(int(major_composition_percentage * 100))):]
        split_minor_composition = re.findall(
            r'[A-Z][^A-Z]*', minor_composition)
        for element_percentage in split_minor_composition:
            split_element_percentage = re.split(
                r'(\d+(?:\.\d+)?)', element_percentage)

            decimal_places = 2
            if '.' in str(split_element_percentage[1]):
                decimal_places += len(
                    str(split_element_percentage[1]).split('.')[1])

            composition[split_element_percentage[0]] = round(
                float(split_element_percentage[1]) / 100.0, decimal_places)

    else:

        split_composition = re.findall(
            r'[A-Z][^A-Z]*', composition_string)

        for element_percentage in split_composition:
            split_element_percentage = re.split(
                r'(\d+(?:\.\d+)?)', element_percentage)

            decimal_places = 2
            if '.' in str(split_element_percentage[1]):
                decimal_places += len(
                    str(split_element_percentage[1]).split('.')[1])

            composition[split_element_percentage[0]] = round(
                float(split_element_percentage[1]) / 100.0, decimal_places)

    filtered_composition = {}
    for element in composition:
        if(composition[element] > 0):
            filtered_composition[element] = composition[element]

    ordered_composition = OrderedDict()
    elements = filtered_composition.keys()
    for element in sorted(elements):
        ordered_composition[element] = filtered_composition[element]

    return ordered_composition


def composition_to_string(composition):
    if isinstance(composition, (str, np.str_, np.string_)):
        composition = parse_composition(composition)

    sorted_composition = sorted(zip([composition[e] for e in composition],
                                    [e for e in composition]))[::-1]

    composition_str = ""
    for e in sorted_composition:
        percentage_str = str(e[0] * 100)

        split_str = percentage_str.split('.')
        decimal = split_str[1]
        if decimal == '0':
            percentage_str = split_str[0]
        else:
            decimal_places = len(str(e[0]).split('.')[1])
            percentage_str = str(round(float(percentage_str), decimal_places))

            split_str = percentage_str.split('.')
            decimal = split_str[1]
            if decimal == '0':
                percentage_str = split_str[0]

        composition_str += e[1] + percentage_str

    return composition_str


def valid_composition(composition_string):
    composition = parse_composition(composition_string)

    total = 0
    for element in composition:
        total += composition[element]
    if(1 - total > 0.05):
        return False
    return True


def calc_range(composition, data):
    if(len(composition) > 1):
        maxVal = -np.inf
        minVal = np.inf
        for element in composition:
            if(element in data):
                value = data[element] * composition[element]
                if value > maxVal:
                    maxVal = value
                if value < minVal:
                    minVal = value

        return np.abs(maxVal - minVal)

    else:
        return 0


def calculate_discrepancy(composition, featureName):

    data = extract_data(composition, featureName)
    return discrepancy(composition, data)


def calculate_deviation(composition, featureName):

    data = extract_data(composition, featureName)
    return deviation(composition, data)


def calculate_range(composition, featureName):

    data = extract_data(composition, featureName)
    return calc_range(composition, data)


def calculate_ideal_entropy(composition):

    ideal_entropy = 0
    for element in composition:
        ideal_entropy += composition[element] * np.log(composition[element])

    return -ideal_entropy


def calculate_ideal_entropy_xia(composition):

    cube_sum = 0
    for element in composition:
        cube_sum += composition[element] * calculate_radius(
            element, composition)**3

    ideal_entropy = 0
    for element in composition:
        ideal_entropy += composition[element] * np.log((composition[element] * calculate_radius(
            element, composition)**3) / cube_sum)

    return -ideal_entropy


def calculate_mismatch_entropy(composition):

    diameters = {}
    for element in composition:
        diameters[element] = calculate_radius(
            element, composition) * 2

    sigma_2 = 0
    for element in composition:
        sigma_2 += composition[element] * \
            (diameters[element]**2)

    sigma_3 = 0
    for element in composition:
        sigma_3 += composition[element] * \
            (diameters[element]**3)

    y_3 = (sigma_2**3) / (sigma_3**2)

    y_1 = 0
    y_2 = 0

    elements = list(composition.keys())

    for i in range(len(elements) - 1):
        for j in range(i + 1, len(elements)):
            element = elements[i]
            otherElement = elements[j]

            y_1 += (diameters[element] + diameters[otherElement]) * (
                (diameters[element] - diameters[otherElement])**2) * composition[element] * composition[otherElement]

            y_2 += diameters[element] * diameters[otherElement] * (
                (diameters[element] - diameters[otherElement])**2) * composition[element] * composition[otherElement]

    y_1 /= sigma_3

    y_2 *= (sigma_2 / (sigma_3**2))

    packing_fraction = 0.64
    zeta = 1.0 / (1 - packing_fraction)

    mismatch_entropy = (((3.0 / 2.0) * ((zeta**2) - 1) * y_1) + ((3.0 / 2.0) * (
        (zeta - 1)**2) * y_2) - (1 - y_3) * (0.5 * (zeta - 1) * (zeta - 3) + np.log(zeta)))  # * boltzmann

    return mismatch_entropy


def calculate_structure_mismatch(composition):
    structures = {}
    for element in composition:
        if(elementData[element]['stpProperties']['phase'] == 'solid'):
            structure = elementData[element]['stpProperties']['crystalStructure']
        else:
            structure = elementData[element]['stpProperties']['phase']
        if structure not in structures:
            structures[structure] = 0
        structures[structure] += composition[element]

    if(len(structures) > 1):
        shannonEntropy = 0
        for structure in structures:
            shannonEntropy -= structures[structure] * \
                np.log(structures[structure])
        return shannonEntropy
    else:
        return 0


def calculate_composition_evenness(composition):

    if(len(composition) > 1):
        shannonEntropy = 0
        for element in composition:
            shannonEntropy -= composition[element] * \
                np.log(composition[element])
        return shannonEntropy / np.log(len(composition))
    else:
        return 1


def Gamma(elementA, elementB):

    Q, P, R = calculate_QPR(elementA, elementB)
    return calculate_electronegativity_enthalpy_component(
        elementA, elementB, P) + calculate_WS_enthalpy_component(elementA, elementB, Q) - R


def calculate_QPR(elementA, elementB):
    seriesA = elementData[elementA]['series']
    if(elementA == 'Ca' or elementA == 'Sr' or elementA == 'Ba'):
        seriesA = 'nonTransitionMetal'

    seriesB = elementData[elementB]['series']
    if(elementB == 'Ca' or elementB == 'Sr' or elementB == 'Ba'):
        seriesB = 'nonTransitionMetal'

    if seriesA == 'transitionMetal' and seriesB == 'transitionMetal':
        P = 14.1
        R = 0
    elif seriesA != 'transitionMetal' and seriesB != 'transitionMetal':
        P = 10.6
        R = 0
    else:
        P = 12.3
        R = extra_data.Rparams(elementA) * \
            extra_data.Rparams(elementB)

    Q = P * 9.4

    return Q, P, R


def calculate_electronegativity_enthalpy_component(elementA, elementB, P):
    electronegativityDiff = extra_data.electronegativityMiedema(
        elementA) - extra_data.electronegativityMiedema(elementB)

    return -P * (electronegativityDiff**2)


def calculate_WS_enthalpy_component(elementA, elementB, Q):
    return Q * (diffDiscontinuity(elementA, elementB)**2)


def diffDiscontinuity(elementA, elementB):
    discontinuityA = elementData[elementA]['wignerSeitzElectronDensity']
    discontinuityB = elementData[elementB]['wignerSeitzElectronDensity']
    return (discontinuityA**(1. / 3.)) - (discontinuityB**(1. / 3.))


def calculate_molar_volume(element):
    return elementData[element]['mass'] / \
        elementData[element]['stpProperties']['density']


def calculate_surface_concentration(elements, volumes, composition):

    reduced_vol_A = composition[elements[0]] * (volumes[0]**(2.0 / 3.0))
    reduced_vol_B = composition[elements[1]] * (volumes[1]**(2.0 / 3.0))

    return reduced_vol_A / (reduced_vol_A + reduced_vol_B)


def calc_f_AB(surfaceConcentration_A, surfaceConcentration_B):
    gamma = 4
    return surfaceConcentration_B * \
        (1 + gamma * ((surfaceConcentration_A * surfaceConcentration_B)**2))


def calculate_corrected_volume(elementA, elementB, Cs_A):

    pureV = extra_data.volumeMiedema(elementA)

    electronegativityDiff = extra_data.electronegativityMiedema(
        elementA) - extra_data.electronegativityMiedema(elementB)

    a = None
    if(elementA in ['Ca', 'Sr', 'Ba']):
        a = 0.04
    elif(elementA in ['Ru', 'Rh', 'Pd', 'Os', 'Ir', 'Pt', 'Au']):
        a = 0.07

    if a is None:
        if(elementData[elementA]['series'] == 'alkaliMetal'):
            a = 0.14
        elif(elementData[elementA]['valenceElectrons'] == 2):
            a = 0.1
        elif(elementData[elementA]['valenceElectrons'] == 3):
            a = 0.07
        else:
            a = 0.04

    f_AB = 1 - Cs_A

    correctedV = (pureV**(2.0 / 3.0)) * (1 + a * f_AB * electronegativityDiff)

    return correctedV


def calculate_interface_enthalpy(elementA, elementB, volumeA):

    discontinuityA = elementData[elementA]['wignerSeitzElectronDensity']
    discontinuityB = elementData[elementB]['wignerSeitzElectronDensity']

    return 2 * volumeA * Gamma(elementA, elementB) / \
        (discontinuityA**(-1. / 3.) + discontinuityB**(-1. / 3.))


def calculate_topological_enthalpy(composition):
    topological_enthalpy = 0
    for element in composition:
        topological_enthalpy += elementData[element]['stpProperties']['fusionEnthalpy'] * \
            composition[element]

    return topological_enthalpy


def calculate_mixing_enthalpy(composition):
    if(len(composition) > 1):

        elements = []
        for element in composition:
            elements.append(element)
        elementPairs = [(a, b) for idx, a in enumerate(elements)
                        for b in elements[idx + 1:]]

        total_mixing_enthalpy = 0
        for pair in elementPairs:
            tmpComposition = {}
            subComposition = 0
            for element in pair:
                subComposition += composition[element]
            for element in pair:
                tmpComposition[element] = composition[element] / \
                    subComposition

            Cs_A = None
            V_A_alloy, V_B_alloy = extra_data.volumeMiedema(
                pair[0]), extra_data.volumeMiedema(pair[1])
            for _ in range(10):

                Cs_A = calculate_surface_concentration(
                    pair, [V_A_alloy, V_B_alloy], tmpComposition)

                V_A_alloy = calculate_corrected_volume(
                    pair[0], pair[1], Cs_A)

                V_B_alloy = calculate_corrected_volume(
                    pair[1], pair[0], 1 - Cs_A)

            chemical_enthalpy = composition[pair[0]] * composition[pair[1]] * (
                (1 - Cs_A) * calculate_interface_enthalpy(pair[0], pair[1], V_A_alloy) +
                Cs_A *
                calculate_interface_enthalpy(pair[1], pair[0], V_B_alloy)
            )

            total_mixing_enthalpy += chemical_enthalpy

    else:
        total_mixing_enthalpy = 0

    return total_mixing_enthalpy


def calculate_ionisation_mix(composition):

    ionisation_energies = {}
    for element in composition:
        if 'ionisationEnergies' in elementData[element]:
            ionisation_energies[element] = elementData[element]['ionisationEnergies'][0]
        else:
            return None

    return linear_mixture(composition, 'ionisationEnergies',
                          data=ionisation_energies)


def calculate_ionisation_discrepancy(composition):

    ionisation_energies = {}
    for element in composition:
        if 'ionisationEnergies' in elementData[element]:
            ionisation_energies[element] = elementData[element]['ionisationEnergies'][0]
        else:
            return None

    return discrepancy(composition, ionisation_energies)


def calculate_ionisation_range(composition):

    ionisation_energies = {}
    for element in composition:
        if 'ionisationEnergies' in elementData[element]:
            ionisation_energies[element] = elementData[element]['ionisationEnergies'][0]
        else:
            return None

    return calc_range(composition, ionisation_energies)


def calculate_categorical_mismatch(composition, featureName):
    data = extract_data(composition, featureName)

    values = {}
    for element in composition:
        value = data[element]
        if value not in values:
            values[value] = 0
        values[value] += composition[element]

    if(len(values) > 1):
        shannonEntropy = 0
        for value in values:
            shannonEntropy -= values[value] * \
                np.log(values[value])
        return shannonEntropy
    else:
        return 0


plankConstant = 6.63e-34
avogadroNumber = 6.022e23
idealGasConstant = 8.31
boltzmann = 1.38064852e-23


def calculate_viscosity(composition, mixingEnthalpy,
                        averageMeltingTemperature):

    const = 3.077e-3
    elementalViscosity = {}
    for element in composition:
        elementalViscosity[element] = const * np.sqrt((elementData[element]['mass'] / 1000) * elementData[element]
                                                      ['stpProperties']['meltingTemperature']) / (calculate_molar_volume(element) * 1.0E-6)

    sum_aG = 0
    for element in composition:
        sum_aG += elementData[element]['stpProperties']['meltingTemperature'] * composition[element] * np.log(
            (elementalViscosity[element] * (elementData[element]['mass'] / 1000)) / (plankConstant * avogadroNumber * (elementData[element]['stpProperties']['density']) * 1000))
    sum_aG *= idealGasConstant

    averageMolarVolume = 0
    for element in composition:
        averageMolarVolume += composition[element] * \
            (calculate_molar_volume(element) * 1.0E-6)

    viscosity = ((plankConstant * avogadroNumber) / (averageMolarVolume)) * np.exp((sum_aG - 0.155 * mixingEnthalpy) /
                                                                                   (idealGasConstant * averageMeltingTemperature))

    return viscosity


def calculate_average_size_ratio(composition):
    if(len(composition) > 1):
        elements = []
        for element in composition:
            elements.append(element)
        elementPairs = [(a, b) for idx, a in enumerate(elements)
                        for b in elements[idx + 1:]]

        average_ratio = 0
        for pair in elementPairs:
            tmpComposition = {}
            subComposition = 0
            for element in pair:
                subComposition += composition[element]
            for element in pair:
                tmpComposition[element] = composition[element] / subComposition

            radii = [tmpComposition[pair[0]] * calculate_radius(pair[0], tmpComposition),
                     tmpComposition[pair[1]] * calculate_radius(pair[0], tmpComposition)]
            maxR = max(radii)
            minR = min(radii)
            ratio = 1 - np.abs((maxR - minR) / maxR)

            average_ratio += subComposition * ratio

        return average_ratio / len(elementPairs)
    else:
        return 1


def calculate_num_elements(composition):
    if(len(composition) > 1):
        minComposition = 1
        for element in composition:
            minComposition = min(minComposition, composition[element])

        sumComposition = 0
        maxComposition = 0
        for element in composition:
            tmp = composition[element] / minComposition
            sumComposition += tmp
            maxComposition = max(maxComposition, tmp)
        numElements = sumComposition / maxComposition
    else:
        numElements = 1
    return numElements


def calculate_radius_gamma(composition):
    maxR = 0
    minR = 1000

    meanR = 0
    for element in composition:
        if(element in elementData):
            meanR += composition[element] * \
                calculate_radius(element, composition)

    for element in composition:
        r = calculate_radius(element, composition)
        if r > maxR:
            maxR = r
        if r < minR:
            minR = r

    rMinAvSq = (minR + meanR)**2
    rMaxAvSq = (maxR + meanR)**2
    rAvSq = meanR**2

    numerator = 1.0 - np.sqrt((rMinAvSq - rAvSq) / (rMinAvSq))
    denominator = 1.0 - np.sqrt((rMaxAvSq - rAvSq) / (rMaxAvSq))

    return numerator / denominator


def calculate_lattice_distortion(composition):

    meanR = 0
    for element in composition:
        if(element in elementData):
            meanR += composition[element] * \
                calculate_radius(element, composition)

    lattice_distortion = 0
    elements = list(composition.keys())
    for i in range(len(elements) - 1):
        for j in range(i + 1, len(elements)):
            element = elements[i]
            otherElement = elements[j]

            lattice_distortion += (composition[element] * composition[otherElement] * np.abs(
                calculate_radius(element, composition) + calculate_radius(otherElement, composition) - 2 * meanR)) / (2 * meanR)

    return lattice_distortion


def calculate_radius(element, composition):
    return elementData[element]['radius']

def calculate_radius_mix(composition):

    total = 0
    for element in composition:
        total += composition[element] * \
            calculate_radius(element, composition)

    return total


def calculate_radius_discrepancy(composition):
    radii = {}
    for element in composition:
        radii[element] = calculate_radius(element, composition)
    return discrepancy(composition, radii)


def calculate_radius_range(composition):
    radii = {}
    for element in composition:
        radii[element] = calculate_radius(element, composition)

    return calc_range(composition, radii)


def calculate_atomic_volume_mix(composition):
    total = 0
    for element in composition:
        total += (4. / 3.) * np.pi * composition[element] * \
            (calculate_radius(element, composition)**3)

    return total


def calculate_atomic_volume_discrepancy(composition):
    volumes = {}
    for element in composition:
        volumes[element] = (4. / 3.) * np.pi * \
            calculate_radius(element, composition)**3

    return discrepancy(composition, volumes)


def calculate_atomic_volume_range(composition):
    volumes = {}
    for element in composition:
        volumes[element] = calculate_radius(element, composition)**3

    return calc_range(composition, volumes)


def calc_num_valence(composition, orbital, totalValence):
    orbitalCount = {}
    for element in composition:
        orbitalCount[element] = 0

        i = 0
        electrons = 0
        while electrons < elementData[element]['valenceElectrons']:
            electrons += elementData[element]['orbitals'][-1 - i]['electrons']
            if elementData[element]['orbitals'][-1 -
                                                i]['orbital'][-1] == orbital:
                orbitalCount[element] += elementData[element]['orbitals'][-1 - i]['electrons']
            i += 1

    total = 0
    for element in composition:
        total += orbitalCount[element] * composition[element]

    return total / totalValence


def calculate_theoretical_density(composition):
    massFractions = {}
    totalMass = 0
    for element in composition:
        totalMass += elementData[element]['mass'] * composition[element]

    for element in composition:
        massFractions[element] = composition[element] * \
            elementData[element]['mass'] / totalMass

    total = 0
    for element in composition:
        total += massFractions[element] / \
            elementData[element]['stpProperties']['density']

    return 1 / total


def calculate_price(composition):
    total_weight = linear_mixture(composition, 'mass')

    price = 0
    for element in composition:
        weight_percent = elementData[element]['mass'] * \
            composition[element] / total_weight
        price += weight_percent * elementData[element]['price']

    return price


droppedFeatures = []


def getElementData():
    
    with open("./data/elements.json") as jsonFile:
        elementDataRaw = json.load(jsonFile)
    
    elementData = {}
    for element in elementDataRaw:

        if('USE' in element['radius']):
            radius = element['radius']['USE']
        elif('empirical' in element['radius']):
            radius = element['radius']['empirical']
        elif('metallic' in element['radius']):
            radius = element['radius']['metallic']
        element['radius'] = radius

        if(element['group'] is None):
            if(element['series'] == 'lanthanide' or element['series'] == 'actinide'):
                element['group'] = 3

        elementData[element['symbol']] = element

    for element in elementData:
        elementData[element]['electronegativity']['mulliken'] = mulliken(
            element, elementData)
        elementData[element]['electronegativity']['miedema'] = extra_data.electronegativityMiedema(
            element)

    return elementData


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

    global elementData
    global droppedFeatures

    # calculate_extra_features = True
    # use_composition_vector = False
    if not calculate_extra_features:
        dropCorrelatedFeatures = False

    basicFeatures = ['atomicNumber', 'periodicNumber', 'mass', 'group',
                     'period', 'protons', 'neutrons', 'electrons', 'valenceElectrons',
                     'valence', 'electronAffinity',
                     'wignerSeitzElectronDensity', 'workFunction',
                     'universalSequence', 'chemicalScale',
                     'pettiforMendeleev', 'modifiedMendeleev', 'pauling',
                     'miedema', 'mulliken', 'meltingTemperature',
                     'boilingTemperature', 'fusionEnthalpy',
                     'vaporisationEnthalpy', 'molarHeatCapacity',
                     'thermalConductivity', 'thermalExpansion',
                     'density', 'cohesiveEnergy', 'debyeTemperature',
                     'chemicalHardness', 'chemicalPotential']

    for additional in additionalFeatures:
        if additional not in basicFeatures:
            basicFeatures.append(additional)

    complexFeatures = ['radius_linearMix', 'radius_discrepancy', 'theoreticalDensity',
                       'atomicVolume_linearMix', 'atomicVolume_discrepancy',
                       'sValence', 'pValence', 'dValence', 'fValence',
                       'structure_discrepancy', 'idealEntropy',
                       'idealEntropyXia', 'mismatchEntropy',
                       'mixingEntropy', 'mixingEnthalpy',
                       'mixingGibbsFreeEnergy',
                       #     'rearrangementInhibition',
                       #     'thermodynamicFactor',
                       'ionisation_linearMix', 'ionisation_discrepancy',
                       'block_discrepancy', 'series_discrepancy', 'viscosity',  # 'radiusGamma',
                       'latticeDistortion', 'EsnPerVec', 'EsnPerMn',
                       'mismatchPHS', 'mixingPHS', 'PHSS', 'price']

    if len(requiredFeatures) > 0:
        dropCorrelatedFeatures = False

        for feature in requiredFeatures:
            if feature.endswith("_percent"):
                use_composition_vector = True

            elif "_linearMix" in feature:
                calculate_extra_features = True
                actualFeature = feature.split("_linearMix")[0]
                if actualFeature not in basicFeatures and actualFeature not in complexFeatures and feature not in complexFeatures:
                    basicFeatures.append(actualFeature)

            elif "_discrepancy" in feature:
                calculate_extra_features = True
                actualFeature = feature.split("_discrepancy")[0]
                if actualFeature not in basicFeatures and actualFeature not in complexFeatures and feature not in complexFeatures:
                    basicFeatures.append(actualFeature)

            else:
                calculate_extra_features = True
                if feature not in complexFeatures:
                    complexFeatures.append(feature)

    if(len(elementData) == 0):
        elementData = getElementData()

    compositionPercentages = {}
    for element in elementData:
        if element not in compositionPercentages:
            compositionPercentages[element] = []

    featureValues = {}
    complexFeatureValues = {}

    for feature in basicFeatures:
        featureValues[feature] = {
            'linearMix': [],
            # 'reciprocalMix': [],
            'discrepancy': [],
            #            'deviation': []
        }
        units[feature + '_discrepancy'] = "%"

    for feature in complexFeatures:
        complexFeatureValues[feature] = []

    if('GFA' in data.columns):
        data['GFA'] = data['GFA'].map({'Crystal': 0, 'Ribbon': 1, 'BMG': 2})
        data['GFA'] = data['GFA'].fillna(maskValue)
        data['GFA'] = data['GFA'].astype(np.int64)

    for i, row in data.iterrows():

        composition = parse_composition(row['composition'])

        for element in composition:
            if element not in elementData:
                print("ERROR: UNKNOWN ELEMENT ", element)
                exit()

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
                if 'linearMix' in featureValues[feature]:
                    featureValues[feature]['linearMix'].append(
                        linear_mixture(composition, feature))

                if 'reciprocalMix' in featureValues[feature]:
                    featureValues[feature]['reciprocalMix'].append(
                        reciprocal_mixture(composition, feature))

                if 'discrepancy' in featureValues[feature]:
                    featureValues[feature]['discrepancy'].append(
                        calculate_discrepancy(composition, feature))

                if 'deviation' in featureValues[feature]:
                    featureValues[feature]['deviation'].append(
                        calculate_deviation(composition, feature))

            if 'radius_linearMix' in complexFeatureValues:
                complexFeatureValues['radius_linearMix'].append(
                    calculate_radius_mix(composition))

            if 'radius_discrepancy' in complexFeatureValues:
                complexFeatureValues['radius_discrepancy'].append(
                    calculate_radius_discrepancy(composition))

            if 'radius_range' in complexFeatureValues:
                complexFeatureValues['radius_range'].append(
                    calculate_radius_range(composition))

            if 'atomicVolume_linearMix' in complexFeatureValues:
                complexFeatureValues['atomicVolume_linearMix'].append(
                    calculate_atomic_volume_mix(composition))

            if 'atomicVolume_discrepancy' in complexFeatureValues:
                complexFeatureValues['atomicVolume_discrepancy'].append(
                    calculate_atomic_volume_discrepancy(composition))

            if 'atomicVolume_range' in complexFeatureValues:
                complexFeatureValues['atomicVolume_range'].append(
                    calculate_atomic_volume_range(composition))

            if 'theoreticalDensity' in complexFeatureValues:
                complexFeatureValues['theoreticalDensity'].append(
                    calculate_theoretical_density(composition))

            if 'sValence' in complexFeatureValues:
                complexFeatureValues['sValence'].append(calc_num_valence(
                    composition, 's', featureValues['valenceElectrons']['linearMix'][-1]))
            if 'pValence' in complexFeatureValues:
                complexFeatureValues['pValence'].append(calc_num_valence(
                    composition, 'p', featureValues['valenceElectrons']['linearMix'][-1]))
            if 'dValence' in complexFeatureValues:
                complexFeatureValues['dValence'].append(calc_num_valence(
                    composition, 'd', featureValues['valenceElectrons']['linearMix'][-1]))
            if 'fValence' in complexFeatureValues:
                complexFeatureValues['fValence'].append(calc_num_valence(
                    composition, 'f', featureValues['valenceElectrons']['linearMix'][-1]))

            if 'structure_discrepancy' in complexFeatureValues:
                complexFeatureValues['structure_discrepancy'].append(
                    calculate_structure_mismatch(composition))

            if 'idealEntropy' in complexFeatureValues:
                complexFeatureValues['idealEntropy'].append(
                    calculate_ideal_entropy(composition))

            if 'idealEntropyXia' in complexFeatureValues:
                complexFeatureValues['idealEntropyXia'].append(
                    calculate_ideal_entropy_xia(composition))

            if 'mismatchEntropy' in complexFeatureValues:
                complexFeatureValues['mismatchEntropy'].append(
                    calculate_mismatch_entropy(composition))

            if 'mixingEntropy' in complexFeatureValues:
                complexFeatureValues['mixingEntropy'].append(
                    complexFeatureValues['idealEntropy'][-1] + complexFeatureValues['mismatchEntropy'][-1])

            if 'ionisation_linearMix' in complexFeatureValues:
                complexFeatureValues['ionisation_linearMix'].append(
                    calculate_ionisation_mix(composition))
            if 'ionisation_discrepancy' in complexFeatureValues:
                complexFeatureValues['ionisation_discrepancy'].append(
                    calculate_ionisation_discrepancy(composition))
            if 'ionisation_range' in complexFeatureValues:
                complexFeatureValues['ionisation_range'].append(
                    calculate_ionisation_range(composition))

            if 'block_discrepancy' in complexFeatureValues:
                complexFeatureValues['block_discrepancy'].append(
                    calculate_categorical_mismatch(composition, 'block'))

            if 'series_discrepancy' in complexFeatureValues:
                complexFeatureValues['series_discrepancy'].append(
                    calculate_categorical_mismatch(composition, 'series'))

            if 'mixingEnthalpy' in complexFeatureValues:
                complexFeatureValues['mixingEnthalpy'].append(
                    calculate_mixing_enthalpy(composition))

            if 'price' in complexFeatureValues:
                complexFeatureValues['price'].append(
                    calculate_price(composition))

            if 'mixingGibbsFreeEnergy' in complexFeatureValues:
                complexFeatureValues['mixingGibbsFreeEnergy'].append(
                    (complexFeatureValues['mixingEnthalpy'][-1] * 1e3) - featureValues['meltingTemperature']['linearMix'][-1] * complexFeatureValues['mixingEntropy'][-1] * idealGasConstant)

            if 'mismatchPHS' in complexFeatureValues:
                complexFeatureValues['mismatchPHS'].append(
                    complexFeatureValues['mixingEnthalpy'][-1] * complexFeatureValues['mismatchEntropy'][-1])

            if 'mixingPHS' in complexFeatureValues:
                complexFeatureValues['mixingPHS'].append(
                    complexFeatureValues['mixingEnthalpy'][-1] * complexFeatureValues['mixingEntropy'][-1])

            if 'PHSS' in complexFeatureValues:
                complexFeatureValues['PHSS'].append(
                    complexFeatureValues['mixingEnthalpy'][-1] *
                    complexFeatureValues['mixingEntropy'][-1] *
                    complexFeatureValues['mismatchEntropy'][-1]
                )

            if 'viscosity' in complexFeatureValues:
                complexFeatureValues['viscosity'].append(
                    calculate_viscosity(composition, complexFeatureValues['mixingEnthalpy'][-1], featureValues['meltingTemperature']['linearMix'][-1]))

            if 'radiusGamma' in complexFeatureValues:
                complexFeatureValues['radiusGamma'].append(
                    calculate_radius_gamma(composition))

            if 'latticeDistortion' in complexFeatureValues:
                complexFeatureValues['latticeDistortion'].append(
                    calculate_lattice_distortion(composition))

            if 'EsnPerVec' in complexFeatureValues:
                complexFeatureValues['EsnPerVec'].append(featureValues['period']['linearMix']
                                                         [-1] / featureValues['valenceElectrons']['linearMix'][-1])

            if 'EsnPerMn' in complexFeatureValues:
                complexFeatureValues['EsnPerMn'].append(featureValues['period']['linearMix']
                                                        [-1] / featureValues['universalSequence']['linearMix'][-1])

            if 'Rc' in complexFeatureValues:
                complexFeatureValues['Rc'].append(((featureValues['meltingTemperature']['linearMix'][-1]**2)
                                                   /
                                                   (complexFeatureValues['atomicVolume_linearMix'][-1]
                                                    * complexFeatureValues['viscosity'][-1])) *
                                                  np.exp(complexFeatureValues['mixingGibbsFreeEnergy'][-1]
                                                         / (idealGasConstant *
                                                            featureValues['meltingTemperature']['linearMix'][-1])))

            if 'rearrangementInhibition' in complexFeatureValues:
                complexFeatureValues['rearrangementInhibition'].append(
                    -complexFeatureValues['mixingEntropy'][-1] / ((complexFeatureValues['mixingEnthalpy'][-1] * 1e3) - 1e-10))

            if 'thermodynamicFactor' in complexFeatureValues:
                complexFeatureValues['thermodynamicFactor'].append(
                    (featureValues['meltingTemperature']['linearMix'][-1] * complexFeatureValues['mixingEntropy'][-1]) / (np.abs(complexFeatureValues['mixingEnthalpy'][-1] * 1e3) + 1e-10))

            if 'CompositionL2' in complexFeatureValues:
                complexFeatureValues['CompositionL2'].append(np.linalg.norm(
                    list(composition.values()), ord=2))
            if 'CompositionL3' in complexFeatureValues:
                complexFeatureValues['CompositionL3'].append(np.linalg.norm(
                    list(composition.values()), ord=3))
            if 'CompositionL5' in complexFeatureValues:
                complexFeatureValues['CompositionL5'].append(np.linalg.norm(
                    list(composition.values()), ord=5))
            if 'CompositionL7' in complexFeatureValues:
                complexFeatureValues['CompositionL7'].append(np.linalg.norm(
                    list(composition.values()), ord=7))
            if 'CompositionL10' in complexFeatureValues:
                complexFeatureValues['CompositionL10'].append(np.linalg.norm(
                    list(composition.values()), ord=10))

            # if('Tg' in row and 'Tl' in row and 'Tg' in predictedFeatures and 'Tl' in predictedFeatures):
            #     complexFeatureValues['Trg'].append(row['Tg'] / row['Tl'])
            # if('Tg' in row and 'Tx' in row and 'Tg' in predictedFeatures and 'Tx' in predictedFeatures):
            #     complexFeatureValues['deltaTx'].append(row['Tx'] - row['Tg'])
            # if('Tg' in row and 'Tx' in row and 'Tl' in row and 'Tg' in predictedFeatures and 'Tx' in predictedFeatures and 'Tl' in predictedFeatures):
            #     complexFeatureValues['gammaLu'].append(
            #         row['Tx'] / (row['Tg'] + row['Tl']))

            #     R0 = 5.1e21
            #     gamma0 = 0.427
            #     complexFeatureValues['RcLu'].append(
            # R0 * np.exp(-(np.log(R0) / gamma0) *
            # complexFeatureValues['gammaLu'][-1]))

            #     t0 = 2.8e-7
            #     gamma1 = 0.362
            #     complexFeatureValues['DmaxLu'].append(
            # t0 * np.exp(-(np.log(t0) / gamma1) *
            # complexFeatureValues['gammaLu'][-1]))

            #     complexFeatureValues['gammaDu'].append(
            #         (2 * row['Tx'] - row['Tg']) / row['Tl'])

            #     complexFeatureValues['betaMondal'].append(
            #         (row['Tx'] / row['Tg']) + (row['Tg'] / row['Tl']))

            #     complexFeatureValues['deltaChen'].append(
            #         row['Tx'] / (row['Tl'] - row['Tg']))

            #     if(complexFeatureValues['deltaTx'][-1] > 0):
            #         complexFeatureValues['phiFan'].append(
            #             complexFeatureValues['Trg'][-1] * (complexFeatureValues['deltaTx'][-1] / row['Tg'])**0.143)
            #     else:
            #         complexFeatureValues['phiFan'].append(0)

            # if('Tl' in row and 'Tx' in row and 'Tl' in predictedFeatures and 'Tx' in predictedFeatures):
            # complexFeatureValues['alphaMondal'].append(row['Tx'] / row['Tl'])

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

            composition = composition_to_string(row['composition'])

            if(not valid_composition(row['composition'])):
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
            composition = composition_to_string(row['composition'])

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

            deduplicated_rows.append(pd.DataFrame(averaged_features, index=[0]))

        if(len(deduplicated_rows)>0):
            deduplicated_data = pd.concat(deduplicated_rows, ignore_index=True)
            data = pd.concat([data,deduplicated_data], ignore_index=True)

        
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
                '_linearMix')[0].split('_discrepancy')[0]
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
                # classWeights.append(numSamples / (len(classes)+counts[c]))
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
