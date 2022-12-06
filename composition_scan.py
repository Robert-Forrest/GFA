import re

import cerebral as cb
import metallurgy as mg

pretty_labels = {"Dmax": "$D_{\mathrm{max}}$ (mm)"}


def run(model, compositions=None, properties=None, uncertainty=False):

    if compositions is None:
        raise ValueError("Error: No compositions entered")
    else:
        for i in range(len(compositions)):
            compositions[i] = re.findall("[A-Z][^A-Z]*", compositions[i])

    if properties is None:
        raise ValueError("Error: No properties entered")

    mg.set_model(cb.models.load(model))

    for composition in compositions:
        print("".join(composition))

        if len(composition) == 2:
            alloys, percentages = mg.generate.binary(composition, step=2)
        elif len(composition) == 3:
            alloys, percentages = mg.generate.ternary(composition)
        elif len(composition) == 4:
            alloys = mg.generate.quaternary(
                composition, quaternary_percentages=[5, 10, 15, 20]
            )
        else:
            raise NotImplementedError()

        for p in properties:
            if isinstance(alloys[0], mg.Alloy):
                values = mg.calculate(alloys, p, uncertainty=uncertainty)
            elif isinstance(alloys[0], list) and isinstance(alloys[0][0], mg.Alloy):
                values = []
                for i in range(len(alloys)):
                    values.append(mg.calculate(alloys[i], p, uncertainty=uncertainty))
            else:
                raise NotImplementedError()

            if p in pretty_labels:
                label = pretty_labels[p]
            else:
                label = p

            mg.plot(
                alloys,
                values,
                label=label,
                save_path="./" + "_".join(composition) + "_" + p,
            )
