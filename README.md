# PicEWS
Public repository for the scripts for the PicEWS Project. This started as a my thesis project from the Cambridge MPhil in Computational biology, and was recently published in [_eClinicalMedicine_](https://doi.org/10.1016/j.eclinm.2025.103255), where a more complete set of methods can be found. Data from GOSH Digital Research Environment was used.

## Processing
This folder contains all of the processing pipeline, which involves `PICU_wrangling.py` file, followed by the `Param_correction.R` file, followed by the `Pandas_to_np.py` file. PICU_wrangling brings together the raw data and does the bulk of the processing. Param_correction does the age normalisation steps where `childsds` was used, which required R. Pandas_to_np cuts the data into the relevant length time slices, links the outcomes, and converts to numpy objects for easier loading and storage. It also contains the `Calculate_mean_sd.R` file, which was used to calculate the size of the standard deviations for blood pressure, heart rate, respiratory rate (using data frome Fleming _et al._ and Zaritsky and Haque, which we refer to in the paper, with references in the paper). 

## Neural networks
This folder contains the neural networks used for comparison. I have not included the tuning scripts as the models were outperformed by XGBoost, so would likely not be used in the future, as they are additionally more computationally intensive.

## XGBoost
This folder contains the script for training the XGBoost model, along with plotting and comparison with logistic regression.

# Reproducibility
If you have any questions feel free to email me at dan.stein [at] nhs.net.

A note on versions:

Python version 3.8.1 used

R Version 4.1.1 used for Parameter_Correction.R with childsds version 0.7.6

R Version 4.0.3 used for Calculate_mean_sd.R with rriskDistributions version 2.1.2

Relevant Packages in Python:

Package               | Version
------------          | -------------
Cython                 | 0.29.23
Keras                  | 2.4.3
keras-nightly          | 2.5.0.dev2021032900
Keras-Preprocessing    | 1.1.2
keras-tuner            | 1.0.3
kerasplotlib           | 0.1.6
matplotlib             | 3.4.2
numba                  | 0.53.1
numpy                  | 1.19.5
pandas                 | 1.2.4
pip                    | 20.3.3
progress               | 1.5
scikit-datasets        | 0.1.38
scikit-learn           | 0.24.2
scikit-optimize        | 0.8.1
scipy                  | 1.2.0
seaborn                | 0.11.1
shap                   | 0.39.0
sklearn                | 0.0
tensorflow             | 2.5.0
tensorflow-addons      | 0.13.0
tensorflow-estimator   | 2.5.0
xgboost                | 1.4.2

All Packages (if issue with dependencies):
Package               | Version
------------          | -------------
absl-py                | 0.12.0
astetik                | 1.11.1
astunparse             | 1.6.3
backcall               | 0.2.0
brotlipy               | 0.7.0
cachetools             | 4.2.2
certifi                | 2020.12.5
cffi                   | 1.14.0
chances                | 0.1.9
chardet                | 4.0.0
cloudpickle            | 1.6.0
conda                  | 4.10.1
conda-package-handling | 1.7.3
cryptography           | 3.4.7
cycler                 | 0.10.0
Cython                 | 0.29.23
dcor                   | 0.5.3
decorator              | 5.0.9
fdasrsf                | 2.3.1
findiff                | 0.8.9
flatbuffers            | 1.12
gast                   | 0.4.0
geonamescache          | 1.2.0
google-auth            | 1.30.1
google-auth-oauthlib   | 0.4.4
google-pasta           | 0.2.0
GPy                    | 1.10.0
graphviz               | 0.16
grpcio                 | 1.34.1
h5py                   | 3.1.0
idna                   | 2.10
ipython                | 7.25.0
ipython-genutils       | 0.2.0
jedi                   | 0.18.0
joblib                 | 1.0.1
Keras                  | 2.4.3
keras-nightly          | 2.5.0.dev2021032900
Keras-Preprocessing    | 1.1.2
keras-tcn              | 3.4.0
keras-tuner            | 1.0.3
kerasplotlib           | 0.1.6
kiwisolver             | 1.3.1
kt-legacy              | 1.0.3
llvmlite               | 0.36.0
Markdown               | 3.3.4
matplotlib             | 3.4.2
matplotlib-inline      | 0.1.2
mpldatacursor          | 0.7.1
mpmath                 | 1.2.1
multimethod            | 1.5
numba                  | 0.53.1
numpy                  | 1.19.5
oauthlib               | 3.1.0
opt-einsum             | 3.3.0
packaging              | 21.0
pandas                 | 1.2.4
paramz                 | 0.9.5
parso                  | 0.8.2
pathlib                | 1.0.1
patsy                  | 0.5.1
pexpect                | 4.8.0
pickleshare            | 0.7.5
Pillow                 | 8.2.0
pip                    | 20.3.3
progress               | 1.5
prompt-toolkit         | 3.0.19
protobuf               | 3.17.1
ptyprocess             | 0.7.0
pyaml                  | 20.4.0
pyasn1                 | 0.4.8
pyasn1-modules         | 0.2.8
pycosat                | 0.6.3
pycparser              | 2.20
pydot                  | 1.4.2
pydot-ng               | 2.0.0
Pygments               | 2.9.0
pyOpenSSL              | 20.0.1
pyparsing              | 2.4.7
PySocks                | 1.7.1
python-dateutil        | 2.8.1
pytz                   | 2021.1
PyYAML                 | 5.4.1
rdata                  | 0.5
requests               | 2.25.1
requests-oauthlib      | 1.3.0
rsa                    | 4.7.2
ruamel-yaml-conda      | 0.15.100
scikit-datasets        | 0.1.38
scikit-learn           | 0.24.2
scikit-optimize        | 0.8.1
scipy                  | 1.2.0
seaborn                | 0.11.1
setuptools             | 52.0.0.post20210125
shap                   | 0.39.0
six                    | 1.15.0
sklearn                | 0.0
slicer                 | 0.0.7
statsmodels            | 0.12.2
sympy                  | 1.8
talos                  | 1.0
tensorboard            | 2.5.0
tensorboard-data-server| 0.6.1
tensorboard-plugin-wit | 1.8.0
tensorflow             | 2.5.0
tensorflow-addons      | 0.13.0
tensorflow-estimator   | 2.5.0
termcolor              | 1.1.0
threadpoolctl          | 2.2.0
tqdm                   | 4.59.0
traitlets              | 5.0.5
typeguard              | 2.12.1
typing-extensions      | 3.7.4.3
urllib3                | 1.26.4
wcwidth                | 0.2.5
Werkzeug               | 2.0.1
wheel                  | 0.36.2
wrangle                | 0.6.7
wrapt                  | 1.12.1
xarray                 | 0.18.2
xgboost                | 1.4.2

