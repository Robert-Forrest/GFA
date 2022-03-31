# Machine-learning improves understanding of glass formation in metallic systems

Code and data associated with the publication "Machine-learning
improves understanding of glass formation in metallic systems".

doi:xyz

## Instructions

The code in this repository utilises a number of other packages to
process data, train neural networks, evaluate those networks, and
visualise predictions.

To run this code, execute the following:

```
git clone https://github.com/Robert-Forrest/GFA
cd GFA
python3 -m pip install -r requirements.txt
python3 __main__.py examples/simple.yaml
```

The examples directory contains configuration files for a number of
situations.

### simple.yaml

`simple.yaml` contains configuration for the `simple` task, which
trains a standard neural-network model. The prediction targets are
defined in the `targets` list.


### kfolds.yaml

`kfolds.yaml` contains configuration for the `kfolds` task, which
performs k-folds cross-validation on the standard neural-network
model.


### kfoldsEnsemble.yaml

`kfoldsEnsemble.yaml` contains configuration for the `kfoldsEnsemble`
task, which performs ensembling to create a meta-learner based on the
submodels produced during k-folds cross-validation.

### permutation.yaml

`permutation.yaml` contains configuration for the
`feature_permutation` task, which shuffles features and measures the
resulting change in model efficacy to judge their importance.


### composition_scan.yaml

`composition_scan.yaml` contains configuration for the
`composition_scan` task, which takes as input alloy spaces such as
CuZr or FeNiBe, and creates graphs of features and predictions across
composition space.


