import matplotlib.pyplot as plt  # pylint: disable=import-error
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from scipy.stats.stats import pearsonr

import plots
import neuralnets
import features
import rule_check

plt.style.use('ggplot')


def inoue(numElements, radiusDiscrepancy, mixingEnthalpy):
    if numElements > 1 and radiusDiscrepancy >= 12 and mixingEnthalpy < 0:
        return True
    return False


originalData = features.load_data(
    plot=False, dropCorrelatedFeatures=False, tmp=True)


Dmax = {
    'crystal': [],
    'ribbon': [],
    'BMG': []
}
newDiscrepancy = {
    'crystal': [],
    'ribbon': [],
    'BMG': []
}
oldDiscrepancy = {
    'crystal': [],
    'ribbon': [],
    'BMG': []
}
changeDiscrepancy = {
    'crystal': [],
    'ribbon': [],
    'BMG': []
}

oldClassifications = []
newClassifications = []
trueClassifications = []


for _, row in originalData.iterrows():

    composition = features.parse_composition(row['composition'])

    # print(row['composition'], row['GFA'], row['Dmax'],
    #       row['miedema_discrepancy'], row['wignerSeitzElectronDensity_discrepancy'])

    elements = list(composition.keys())
    # if(len(composition) != 2):
    #     continue

    radii = {}
    wsDensities = {}
    for element in composition:
        # * composition[element]
        radii[element] = features.elementData[element]['radius']
        # * composition[element]
        wsDensities[element] = features.elementData[element]['wignerSeitzElectronDensity']

    # sorted_radii = dict(
    #     sorted(radii.items(), key=lambda item: item[1], reverse=True))
    # sorted_radii_elements = list(sorted_radii.keys())

    # sorted_wsDensities = dict(
    #     sorted(wsDensities.items(), key=lambda item: item[1], reverse=True))
    # sorted_wsDensities_elements = list(sorted_wsDensities.keys())

    # radii_domination = {}
    # hierarchy = {}
    # hierarchy_diff = {}
    # for elementA in radii:
    #     hierarchy[elementA] = 0
    #     hierarchy_diff[elementA] = 0
    #     radii_domination[elementA] = 0
    #     for elementB in radii:
    #         if elementA != elementB:
    #             if radii[elementA] > radii[elementB]:
    #                 radii_domination[elementA] += 1
    #                 if wsDensities[elementA] > wsDensities[elementB]:
    #                     hierarchy[elementA] += 1
    #     hierarchy_diff[elementA] = hierarchy[elementA] - \
    #         radii_domination[elementA]

    # hierarchy_match = 0
    # for element in composition:
    #     hierarchy_match += hierarchy_diff[element]*composition[element]

    # metric = 0
    # # for i in range(len(elements)-1):
    # #elementA = elements[i]
    # for elementA in composition:
    #     elementA_contribution = 0
    #     # for j in range(i+1, len(elements)):
    #     #    elementB = elements[j]
    #     for elementB in composition:
    #         if elementA != elementB:
    #             subcomposition_total = composition[elementA] + \
    #                 composition[elementB]
    #             subcomposition = {
    #                 elementA: composition[elementA]/subcomposition_total, elementB: composition[elementB]/subcomposition_total}

    #             adjustedRadii = {}
    #             adjustedWSDensities = {}
    #             for e in subcomposition:
    #                 adjustedRadii[e] = radii[e]  # * subcomposition[e]
    #                 # * subcomposition[e]
    #                 adjustedWSDensities[e] = wsDensities[e]

    #             deltaR = adjustedRadii[elementA] - adjustedRadii[elementB]
    #             deltaN = adjustedWSDensities[elementA] - \
    #                 adjustedWSDensities[elementB]

    #             combo = deltaR*deltaN
    #             elementA_contribution += combo

    #             # Q, P, R = features.calculate_QPR(elementA, elementB)
    #             # phi = features.calculate_electronegativity_enthalpy_component(
    #             #     elementA, elementB, P)
    #             # n = features.calculate_WS_enthalpy_component(
    #             #     elementA, elementB, Q)

    #             if "Pd" in composition:
    #                 print(elementA, elementB)
    #                 print("r", radii[elementA], radii[elementB], deltaR)
    #                 print("n", wsDensities[elementA],
    #                       wsDensities[elementB], deltaN)
    #                 print("r.n", combo, combo * composition[elementA])
    #             # metric += subcomposition[elementA]*subcomposition[elementB]*n
    #     metric += elementA_contribution*composition[elementA]
    # if "Pd" in composition:
    #     print("total", metric)

    # metric = 0
    # averageR = features.linear_mixture(composition, data=radii)
    # discrepancyR = features.discrepancy(composition, data=radii)
    # averageN = features.linear_mixture(composition, data=wsDensities)
    # print("average:", averageR, averageN)
    # for element in composition:
    #     factor = 0

    #     r = radii[element]  # *composition[element]
    #     n = wsDensities[element]  # *composition[element]
    #     deltaN = n - averageN

    #     if n > averageN:
    #         if r > averageR:
    #             factor += 1
    #         elif r < averageR:
    #             factor -= 1
    #     elif n < averageN:
    #         if r > averageR:
    #             factor -= 1
    #         elif r < averageR:
    #             factor += 1
    #     print(element, r, n,
    #           deltaN, factor*composition[element])
    #     metric += factor*composition[element] * np.abs(deltaN)
    # #metric *= discrepancyR
    # print(metric)
    # print()

    discrepancyOldR, discrepancyNewR = rule_check.calculate_adjusted_radius_discrepancy(
        composition)
    discrepancyOldR *= 100
    discrepancyNewR *= 100
    discrepancyChange = (discrepancyNewR-discrepancyOldR)

    inoueOld = inoue(len(composition), discrepancyOldR, row['mixingEnthalpy'])
    inoueNew = inoue(len(composition), discrepancyNewR, row['mixingEnthalpy'])

    if inoueOld:
        oldClassifications.append(1)
    else:
        oldClassifications.append(0)

    if inoueNew:
        newClassifications.append(1)
    else:
        newClassifications.append(0)

    if row['GFA'] == 0:
        trueClassifications.append(0)
    else:
        trueClassifications.append(1)

    # metric = discrepancyNewR  # - discrepancyOldR

    # phi_total = 0
    # n_total = 0
    # k = 0
    # for i in range(len(elements)-1):
    #     elementA = elements[i]
    #     elementA_phi = 0
    #     elementA_n = 0
    #     for j in range(i+1, len(elements)):
    #         elementB = elements[j]

    #         Q, P, R = features.calculate_QPR(elementA, elementB)
    #         phi = features.calculate_electronegativity_enthalpy_component(
    #             elementA, elementB, P)
    #         n = features.calculate_WS_enthalpy_component(
    #             elementA, elementB, Q)

    #         elementA_phi += phi
    #         elementA_n += n
    #         k += 1
    #     n_total += elementA_n*composition[elementA]
    #     phi_total += elementA_phi*composition[elementA]

    # metric = n_total
    # print(metric)

    metric = 'wignerSeitzElectronDensity_discrepancy'

    if row['GFA'] == 0:
        oldDiscrepancy['crystal'].append(discrepancyOldR)
        newDiscrepancy['crystal'].append(discrepancyNewR)
        changeDiscrepancy['crystal'].append(discrepancyChange)
        Dmax['crystal'].append(row[metric])
    elif row['GFA'] == 1:
        oldDiscrepancy['ribbon'].append(discrepancyOldR)
        newDiscrepancy['ribbon'].append(discrepancyNewR)
        changeDiscrepancy['ribbon'].append(discrepancyChange)
        Dmax['ribbon'].append(row[metric])
    elif row['GFA'] == 2:
        oldDiscrepancy['BMG'].append(discrepancyOldR)
        newDiscrepancy['BMG'].append(discrepancyNewR)
        changeDiscrepancy['BMG'].append(discrepancyChange)
        Dmax['BMG'].append(row[metric])

print("Crystal", np.mean(oldDiscrepancy['crystal']), np.mean(
    newDiscrepancy['crystal']), np.mean(changeDiscrepancy['crystal']))
print("Ribbon", np.mean(oldDiscrepancy['ribbon']), np.mean(
    newDiscrepancy['ribbon']), np.mean(changeDiscrepancy['ribbon']))
print("BMG", np.mean(oldDiscrepancy['BMG']), np.mean(
    newDiscrepancy['BMG']), np.mean(changeDiscrepancy['BMG']))

combinedX = Dmax['crystal']+Dmax['ribbon']+Dmax['BMG']
combinedOld = oldDiscrepancy['crystal'] + \
    oldDiscrepancy['ribbon'] + oldDiscrepancy['BMG']
combinedNew = newDiscrepancy['crystal'] + \
    newDiscrepancy['ribbon'] + newDiscrepancy['BMG']

oldR = pearsonr(combinedX, combinedOld)
newR = pearsonr(combinedX, combinedNew)

zOld = np.polyfit(combinedX, combinedOld, 1)
pOld = np.poly1d(zOld)

zNew = np.polyfit(combinedX, combinedNew, 1)
pNew = np.poly1d(zNew)

print("Dmax Rsq", oldR, newR)

print("Accuracy", accuracy_score(trueClassifications, oldClassifications),
      accuracy_score(trueClassifications, newClassifications))

print("F1", f1_score(trueClassifications, oldClassifications),
      f1_score(trueClassifications, newClassifications))

print("Precision", precision_score(trueClassifications, oldClassifications),
      precision_score(trueClassifications, newClassifications))

print("Recall", recall_score(trueClassifications, oldClassifications),
      recall_score(trueClassifications, newClassifications))


confusionOld = confusion_matrix(trueClassifications, oldClassifications)
confusionNew = confusion_matrix(trueClassifications, newClassifications)

fig = plt.figure()
ax = fig.add_subplot()
confusionPlot = ConfusionMatrixDisplay(
    confusion_matrix=confusionOld, display_labels=['Crystal', 'Glass'])
confusionPlot.plot(colorbar=False, cmap=plt.cm.Blues, ax=ax)
ax.set_xlabel("Predicted class")
ax.set_ylabel("True class")
plt.tight_layout()
plt.savefig("Inoue_confusion_old.png")
plt.cla()
plt.clf()
plt.close()

fig = plt.figure()
ax = fig.add_subplot()
confusionPlot = ConfusionMatrixDisplay(
    confusion_matrix=confusionNew, display_labels=['Crystal', 'Glass'])
confusionPlot.plot(colorbar=False, cmap=plt.cm.Blues, ax=ax)
ax.set_xlabel("Predicted class")
ax.set_ylabel("True class")
plt.tight_layout()
plt.savefig("Inoue_confusion_new.png")
plt.cla()
plt.clf()
plt.close()


indices = np.arange(3)
width = 0.35

plt.bar(indices, [np.mean(oldDiscrepancy[c])
                  for c in oldDiscrepancy], width, label="Standard radii")
plt.bar(indices+width, [np.mean(newDiscrepancy[c])
                        for c in newDiscrepancy], width, label="Adjusted radii")
plt.xticks(indices + width / 2, ('Crystal', 'Ribbon', 'BMG'))
plt.xlabel("GFA classification")
plt.ylabel("Radii discrepancy (%)")
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('discrepancies.png')
plt.cla()
plt.clf()
plt.close()


# plt.scatter(matches, Dmax, marker='x')
# plt.scatter(crystal_x, crystal_y, label="Crystal", s=10)
# plt.scatter(ribbon_x, ribbon_y, label="Ribbon", s=10)
plt.scatter(Dmax['crystal']+Dmax['ribbon']+Dmax['BMG'],  newDiscrepancy['crystal'] +
            newDiscrepancy['ribbon'] + newDiscrepancy['BMG'], label=r"Adjusted radii, R$^2$="+str(round(newR[0], 2)), s=5, lw=0, color="#e86f5c")
plt.scatter(Dmax['crystal']+Dmax['ribbon']+Dmax['BMG'], oldDiscrepancy['crystal'] +
            oldDiscrepancy['ribbon'] + oldDiscrepancy['BMG'], label=r"Standard radii, R$^2$="+str(round(oldR[0], 2)), s=5, lw=0, color="#71aed1")

plt.plot(combinedX, pNew(combinedX), color="#71261a")
plt.plot(combinedX, pOld(combinedX), color="#1f5371")

lgnd = plt.legend(loc="best")
lgnd.legendHandles[0]._sizes = [30]
lgnd.legendHandles[1]._sizes = [30]

plt.ylabel('Radii discrepancy (%)')
plt.xlabel('Wigner-Seitz boundary electron-density discrepancy (%)')
plt.grid(alpha=.4)
plt.tight_layout()
plt.savefig('radii_ws_discrepancy.png')
plt.cla()
plt.clf()
plt.close()
