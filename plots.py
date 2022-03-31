from PIL import Image
from adjustText import adjust_text
from sklearn.preprocessing import StandardScaler
from scipy.cluster import hierarchy
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, mean_squared_error, mean_absolute_error, roc_curve, auc, accuracy_score, f1_score, recall_score, precision_score, precision_recall_fscore_support
from sklearn import manifold, decomposition, random_projection, discriminant_analysis, neighbors, cluster
import tensorflow as tf  # pylint: disable=import-error
import ternary  # pylint: disable=import-error
import seaborn as sns  # pylint: disable=import-error
import matplotlib.ticker as ticker  # pylint: disable=import-error
import matplotlib.pyplot as plt  # pylint: disable=import-error
import os
import numpy as np  # pylint: disable=import-error
import pandas as pd  # pylint: disable=import-error
import matplotlib as mpl  # pylint: disable=import-error
from mpl_toolkits import axes_grid1
import matplotlib.collections as mcoll
import metallurgy as mg
import cerebral as cb
mpl.use('Agg')
plt.style.use('ggplot')
plt.rc('axes', axisbelow=True)

# mpl.rcParams['text.usetex'] = True


def plot_binary(elements, model, onlyPredictions=False, originalData=None, inspect_features=["percentage"], additionalFeatures=[]):
    if not os.path.exists(cb.conf.image_directory + "compositions"):
        os.makedirs(cb.conf.image_directory + "compositions")
    binary_dir = cb.conf.image_directory + \
        "compositions/" + "_".join(elements)
    if not os.path.exists(binary_dir):
        os.makedirs(binary_dir)
    
    realData = []
    requiredFeatures = None
    if originalData is not None:
        for _, row in originalData.iterrows():
            parsedComposition = mg.alloy.parse_composition(row['composition'])
            if set(elements).issuperset(set(parsedComposition.keys())):
                if elements[0] in parsedComposition:
                    row['percentage'] = parsedComposition[elements[0]] * 100
                else:
                    row['percentage'] = 0
                realData.append(row)
        realData = pd.DataFrame(realData)
        realData = realData.reset_index(drop=True)
        requiredFeatures = list(originalData.columns)

    for feature in inspect_features:
        if feature not in requiredFeatures:
            requiredFeatures.append(feature)

    compositions, percentages = mg.binary.generate_alloys(elements)

    all_features = pd.DataFrame(compositions, columns=['composition'])
    all_features = cb.features.calculate_features(all_features,
                                               dropCorrelatedFeatures=False,
                                               plot=False,
                                               requiredFeatures=requiredFeatures,
                                               additionalFeatures=additionalFeatures)
    all_features = all_features.drop('composition', axis='columns')
    all_features = all_features.fillna(cb.features.maskValue)
    for feature in cb.conf.targets:
        all_features[feature.name] = cb.features.maskValue
    all_features['GFA'] = all_features['GFA'].astype('int64')
    all_features['percentage'] = percentages
    GFA_predictions = []

    prediction_ds = cb.features.df_to_dataset(all_features)
    predictions = model.predict(prediction_ds)
    for i in range(len(cb.conf.targets)):
        if cb.conf.targets[i].name == 'GFA':
            GFA_predictions = predictions[i]
        else:
            all_features[cb.conf.targets[i].name] = predictions[i]

    for inspect_feature in inspect_features:
        if inspect_feature not in all_features.columns:
            continue
        if not os.path.exists(binary_dir+'/'+inspect_feature):
            os.makedirs(binary_dir + '/'+inspect_feature)
        if not os.path.exists(binary_dir+'/'+inspect_feature + '/predictions'):
            os.makedirs(binary_dir+'/'+inspect_feature + '/predictions')
        if not os.path.exists(binary_dir+'/'+inspect_feature + '/features'):
            os.makedirs(binary_dir+'/'+inspect_feature + '/features')

        for feature in all_features.columns:

            xlabel = None
            ylabel = None

            data = []
            labels = []
            scatter_data = None
            use_colorline = False
            
            trueFeatureName = feature.split(
                '_linearmix')[0].split('_discrepancy')[0]
            if (onlyPredictions and feature not in cb.conf.targets and trueFeatureName not in additionalFeatures) or inspect_feature == feature:
                continue
            if feature not in cb.conf.targets and os.path.exists(
                    binary_dir + "/"+inspect_feature + "/" + feature + ".png"):
                continue

            if feature == 'GFA':
                crystal = []
                ribbon = []
                BMG = []
                for prediction in GFA_predictions:
                    crystal.append(prediction[0])
                    ribbon.append(prediction[1])
                    BMG.append(prediction[2])

                data.append(crystal)
                labels.append('Crystal')

                data.append(ribbon)
                labels.append('GR')
                
                data.append(BMG)
                labels.append('BMG')
                
                if len(realData) > 0 and inspect_feature in realData:
                    scatter_data.append({
                        'data':[
                            realData[realData['GFA'] == 0][inspect_feature],
                            [1] * len(realData[realData['GFA'] == 0][inspect_feature])
                        ],
                        'marker':"s",
                        'label':None    
                    })

                    scatter_data.append({
                        'data':[
                            realData[realData['GFA'] == 1][inspect_feature],
                            [1] * len(realData[realData['GFA'] == 1][inspect_feature])
                        ],
                        'marker':"D",
                        'label':None    
                    })

                    scatter_data.append({
                        'data':[
                            realData[realData['GFA'] == 2][inspect_feature],
                            [1] * len(realData[realData['GFA'] == 2][inspect_feature])
                        ],
                        'marker':"o",
                        'label':None    
                    })

            else:

                if len(realData) > 0 and feature in realData and inspect_feature in realData:
                    crystalData, crystalPercentages = cb.features.filter_masked(
                        realData[realData['GFA'] == 0][feature], realData[realData['GFA'] == 0][inspect_feature])
                    ribbonData, ribbonPercentages = cb.features.filter_masked(
                        realData[realData['GFA'] == 1][feature], realData[realData['GFA'] == 1][inspect_feature])
                    bmgData, bmgPercentages = cb.features.filter_masked(
                        realData[realData['GFA'] == 2][feature], realData[realData['GFA'] == 2][inspect_feature])

                    scatter_data = []
                    if len(crystalData) > 0:
                        if len(ribbonData) > 0 or len(bmgData) > 0:
                            scatter_data.append({
                                'data':[crystalPercentages, crystalData],
                                'marker':"s", 'label':"Crystal"
                            })
                    if len(ribbonData) > 0:
                        scatter_data.append({
                            'data':[ribbonPercentages, ribbonData],
                            'marker':"D", 'label':"GR"
                        })
                    if len(bmgData) > 0:
                        scatter_data.append({
                            'data':[bmgPercentages, bmgData],
                            'marker':"o", 'label':"BMG"
                        })
            
                

            if inspect_feature != 'percentage':
                xlabel = cb.features.prettyName(inspect_feature) 
                if inspect_feature in cb.features.units:
                    xlabel += ' (' + cb.features.units[inspect_feature] + ')'
                    
                use_colorline = True
                data = [all_features[inspect_feature],all_features[feature]]
            else:
                data = all_features[feature]

            if feature in cb.features.units:
                ylabel = cb.features.prettyName(feature) +' (' + cb.features.units[feature] + ')'
            elif feature == 'GFA':
                ylabel = "GFA Classification Confidence"
            else:
                ylabel = cb.features.prettyName(feature)

            save_path = ""
            if feature in cb.conf.targets:
                save_path = binary_dir+'/'+inspect_feature + "/predictions/" + feature + ".png"
            else:
                save_path = binary_dir+'/'+inspect_feature + "/features/" + feature + ".png"
    
            mg.plots.binary(
                compositions,
                data,
                scatter_data=scatter_data,
                xlabel=xlabel,
                ylabel=ylabel,
                use_colorline=use_colorline,
                save_path=save_path
            )

