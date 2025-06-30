### XGBoost

import numpy as np
import sklearn 
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from itertools import cycle
#from scipy import interp
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import pandas as pd
import json
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, mean_squared_error, mean_absolute_error, auc, confusion_matrix, roc_curve, precision_score, recall_score, f1_score
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import argparse 
import re
from tqdm import tqdm
import pickle 
import shap

#Make it so you can run this from the command line
parser = argparse.ArgumentParser(description="Allow running of XGBoost with different input parameters",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-l", "--length", default=12, type=int, help="Length of predictive target in hours")
parser.add_argument("-z", "--no_z", default=False, type=bool, help="Whether or not to use z-scores")
parser.add_argument("-i", "--input_length", default=6, type=int, help="Length of the input window in hours")
parser.add_argument("-p", "--perc", default=True, type=bool, help="Input scaled to percentile (T/F)")
args = vars(parser.parse_args())

#Set up whether you want to use the no_z values - default value to be false - but will not override if given above
no_z = args['no_z']
input_length = args['input_length']
length = args['length']
perc = args['perc']

#Set the current working directory
os.chdir('/Users/danstein/Documents/Masters/Course materials/Project/PICU project/PICU_project')
    
print(f'Running with {length}h time target window, {input_length}h input window, using z-scores {no_z == False}, percentile: {perc}')

#Pull out the data depending on whether no_z was used
if not no_z:
    #Read in the data
    file_start = '/Users/danstein/Documents/Masters/Course materials/Project/PICU project/new_project_files/np_arrays_pSOFA'
else:
    file_start = '/Users/danstein/Documents/Masters/Course materials/Project/PICU project/new_project_files/np_arrays_no_zscore_pSOFA'
    
if not perc:
    file_start += '_no_perc'
    
if input_length == 3:
    file_name = file_start + '.npz' 
else:
    file_name = file_start + f'_{input_length}h.npz'
    

#Read in data    
data = np.load(file_name)
array3d = data['d3']
array2d = data['d2']
outcomes = data['outcomes']
characteristics = data['chars']
splines = data['splines']
slopes = data['slopes']
r_values = data['r_values']

#Read in the demographics file
demographics = pd.read_csv('~/Documents/Masters/Course materials/Project/PICU project/caboodle_patient_demographics.csv', sep = ',', parse_dates = ['birth_date', 'death_date'])

#Make an integer project_id version, and use this to make a hash-table
demographics['project_int'] = [int(i[-4:]) for i in demographics.project_id]
unique_hospital_no = demographics.hospital_no.drop_duplicates().reset_index()
unique_hospital_no['hospital_no_int'] = unique_hospital_no.index
demographics = demographics.merge(unique_hospital_no.loc[:, ['hospital_no', 'hospital_no_int']], how = 'left', on = 'hospital_no')

#Now make a lookup table for the unique hospital number against project_id for outcomes
outcomes_lookup = pd.DataFrame({'project_int': outcomes[:, 0].astype(int)})
outcomes_lookup = outcomes_lookup.merge(demographics.loc[:, ['project_int', 'hospital_no_int']], 
                                        how = 'left',
                                        on = 'project_int')
hospital_lookup_np = np.transpose(np.array([outcomes_lookup.hospital_no_int]))
outcomes = np.append(outcomes, hospital_lookup_np, axis = 1)

def test_trainsplit(array, split, array3d = array3d, outcome_array = outcomes, seed = 1):
    """
    Function to split up 3d slices into test, train, validate
    split is an np.ndarray
    """

    #Flexibly choose dimension to split along
    shape = array3d.shape
    z_dim = np.max(shape)
    
    #Get the unique identifiers 
    unique_pts = np.unique(outcome_array[:, 14])
    
    #Now randomly shuffle the order of patients
    np.random.seed(seed)
    np.random.shuffle(unique_pts) #Annoyingly does this in-place - not sure how to change it
    
    #Set some initial parameters
    total_length = outcomes.shape[0]
    proportions = split / np.sum(split)
    total_proportion_length = np.round(np.cumsum(total_length * proportions))
    
    # Calculate lengths of each patient
    lengths = np.array([np.sum(outcome_array[:, 14] == pt) for pt in unique_pts])
    
    # Set up some storage
    sampled_indices = np.array([])
    final_splits = list()
    
    # Iterate over split proportions
    for proportion_length in total_proportion_length:
        # Get the cumulative lengths and where they are less than the cumulative proportion
        cum_lengths = np.cumsum(lengths)
        indices = np.where(cum_lengths <= proportion_length)[0]
        
        # Exclude already sampled indices
        indices = np.setdiff1d(indices, sampled_indices)
        chosen_index = indices
        
        # Add chosen index to sampled indices
        final_splits.append(chosen_index)
        sampled_indices = np.concatenate((sampled_indices, chosen_index), axis=None)
    
    #Now add back so they are shuffled.
    random_split = [unique_pts[i] for i in final_splits]
    
    #Work through the array and pick out the outcome indices and add them to the split list
    split_array = list()
    for i in random_split:
        array_locs = pd.Series(outcome_array[:, 14]).isin(i)
        split_array.append(array[array_locs, ])
    
    return split_array


#Split up testing and outcomes
split_characteristics = test_trainsplit(characteristics, np.array([85, 15]))
split_array2d = test_trainsplit(array2d, np.array([85, 15]))
split_array3d = test_trainsplit(array3d, np.array([85, 15]))
split_slopes = test_trainsplit(slopes, np.array([85, 15]))
split_R = test_trainsplit(r_values, np.array([85, 15]))
split_outcomes = test_trainsplit(outcomes, np.array([85, 15]))

#Training sets
train_characteristics = split_characteristics[0]
train_array3d = split_array3d[0]
train_array2d = split_array2d[0]
train_slopes = split_slopes[0]
train_R = split_R[0]
train_outcomes = split_outcomes[0]

#Test sets
test_characteristics = split_characteristics[1]
test_array3d = split_array3d[1]
test_array2d = split_array2d[1]
test_slopes = split_slopes[1]
test_R = split_R[1]
test_outcomes = split_outcomes[1]

#Make binary outcomes
#Make the binary values
binary_deterioration_train_hour_outcomes = np.array(train_outcomes[:, 12] < length, dtype = int)
binary_deterioration_train_outcomes = np.transpose(np.array([1- binary_deterioration_train_hour_outcomes, binary_deterioration_train_hour_outcomes]))
binary_deterioration_test_hour_outcomes = np.array(test_outcomes[:, 12] < length, dtype = int)
binary_deterioration_test_outcomes = np.transpose(np.array([1- binary_deterioration_test_hour_outcomes, binary_deterioration_test_hour_outcomes]))

#Set x and y
X = np.concatenate((train_array2d, train_characteristics, train_slopes, train_R), axis=1)
y = np.argmax(binary_deterioration_train_outcomes, axis = 1)

# grid search
model = XGBClassifier(objective='binary:logistic', eval_metric = 'aucpr')
kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=7)
param_grid = {"learning_rate"    : [0.001, 0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
 				'max_depth' : [5, 10, 15, 20, 25, 30, 35, 40],
 					"min_child_weight" : [ 1, 5, 7, 15],
 					"gamma"            : [ 0.0, 0.2, 0.4 ],
 					"colsample_bytree" : [ 0.3, 0.5 , 0.7, 1], 
					 "subsample":[0.5, 0.75, 0.9, 1], 
					 "scale_pos_weight" : [0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1, 2, 4], 
					 "n_estimators" : [50, 100, 150, 200, 250]}

model_path_name = 'models/'
file_name = 'XGBoost_best_'
file_suffix = f'{input_length}h_{length}h_{"no_zscore_" if no_z else ""}pSOFA'

if not os.path.exists(model_path_name + file_name + file_suffix + '.json'):                         
    clf = BayesSearchCV(model, param_grid, random_state=0, cv = kfold, iid = False)
    search = clf.fit(X, y)
    
    #Save the best hyperparameters
    best_hyperparameters = param_grid = search.best_params_

    a_file = open(model_path_name + file_name + file_suffix + '.json', "w")
    json.dump(best_hyperparameters, a_file)
    a_file.close()

else:
    f = open(model_path_name + file_name + file_suffix + '.json', )
    param_grid = json.load(f)
    
if not perc:
    file_suffix += '_no_perc'

#get colnames
point_cols = data['point_cols']
point_cols = [i for i in point_cols]
series_cols = data['series_cols']

if not no_z:
    #Combine the series cols objects and name them for labelling of axes
    series_cols = ['Ventilation Status', 'High Frequency Oscillatory Ventilation',
                            'Tracheostomy Status', 'End Tidal CO2 (mmHg)', 'FiO2', 'Oxygen Flow Rate (L/min)',
                            'Weight Adjusted Oxygen Flow Rate (L/min/kg)', 'IPAP (cmH2O)', 'Ventilation (ml/min)',
                            'Mean Airway Pressure', 'EPAP (cmH2O)', 'Ventilation Status Frequency of Input',
                            'Oxygen Flow Rate Missingess', 'IPAP Frequency of Input', 'EPAP Frequency of Input',
                            'FiO2 Frequency of Input', 'High Frequency Oscillatory Ventilation Frequency of Input',
                            'Tracheostomy status Frequency of Input',
                            'Ventilation (ml/min) Frequency of Input', 'Mean Airway Pressure Frequency of Input',
                            'End Tidal CO2 Frequency of Input', 'Systolic BP Frequency of Input', 'Diastolic BP Frequency of Input',
                            'Mean Arterial Pressure Frequency of Input', 'Heart Rate Frequency of Input', 'Comfort Score (Alertness)', 
                            'Comfort Score (BP)', 'Comfort Score (Calmness)', 'Comfort Score', 'Comfort Score (HR)',
                            'Comfort Score (Resp)', 'AVPU', 'GCS (Verbal)', 'GCS (Eyes)', 'GCS (Motor)',
                            'GCS', 'GCS Frequency of Input', 'AVPU Frequency of Input', 'Capillary Refill Time (s)', 'Capillary Refill Time Frequency of Input',
                            'Oxygen Saturation (%)', 'Oxygen Saturation Frequency of Input',
                            'Height (m)', 'Extra-Corporeal Membrane Oxygenation Status', 'Extra-Corporeal Membrane Oxygenation Status Frequency of Input',
                            'Vasoactive Inotropic Score (mcg/kg/min)',
                            'Vasopressin Equivalent dose (mcg/kg/min)', 'Dopamine Equivalent Dose (mcg/kg/min)',
                            'Total Noradrenaline and Adrenaline Dose (mcg/kg/min)',
                            'Milrinone Equivalent Dose (mcg/kg/min)', 'Inotropes Frequency of Input', 'Vasopressin Frequency of Input', 'Dopamine Frequency of Input',
                            'Noradrenaline and Adrenaline Frequency of Input', 'Milrinone Frequency of Input', 'SpO2:FiO2 ratio', 'pSOFA Score (Respiratory)', 
                            'pSOFA Score (Cardiac)', 'pSOFA Score (Platelets)', 'pSOFA Score (Bilirubin)', 'pSOFA Score (GCS)',
                            'pSOFA Score (Creatinine)', 'Respiratory Rate Frequency of Input',
                            'Dialysis Status', 'Dialysis Frequency of Input', 'pSOFA Score', 'Temperature (F)', 'Temperature Frequency of Input',
                            'Urine Output (ml)', 'Urine Output (ml/kg)', 'Urine Output Frequency of Input', 'Paediatric Early Warning Score',
                            'Age Normalised Heart Rate', 'Age Normalised Respiratory Rate',
                            'Age Normalised Diastolic BP', 'Age Normalised Systolic BP', 'Age Normalised Mean Arterial Pressure']

    point_cols = ['ALT (IU/L)', 'Albumin (g/L)', 'Alkaline Phosphatase (U/L)', 'AST (IU/L)', 'Aspartate (U/L)',
                            'Amylase (U/L)', 'APTT (s)', 'Anion gap', 'Base excess (mEq/L)', 'Basophils (x10^9/L)',
                            'Bicarbonate (mmol/L)', 'pH', 'Blood culture performed', 'Creatinine (micromol/L)', 'CRP (mg/L)',
                            'Calcium (mmol/L)', 'Chloride (mmol/L)', 'Eosinophils (x10^9/L)', 'Fraction of Deoxyhaemoglobin',
                            'Methemoglobin (%)', 'Fraction of Oxygenated Haemoglobin', 'Glucose (mmol/L)', 'HCT (%)', 'Haematocrit (L/L)', 'INR',
                            'Lactate (mmol/L)', 'Lymphocytes (x10^9/L)', 'Magnesium (mmol/L)', 'Monocytes (x10^9/L)', 'Neutrophils (x10^9/L)', 'P50 (mmHg)',
                            'PaCO2 (kPa)', 'PcCO2 (kPa)', 'PmCO2 (kPa)', 'PaO2 (kPa)', 'PcO2 (kPa)', 'PmO2 (kPa)', 'PO2 (kPa)', 'PvCO2 (kPa)',
                            'Phosphate (mmol/L)', 'Platelets (x10^9/L)', 'Potassium (mmol/L)', 'PT (s)', 'Reticulocytes (%)', 'Sodium (mmol/L)',
                            'TT (s)', 'Bilirubin (micromol/L)', 'White Cell Count (x10^9/L)', 'Strong Ion Gap (mEq/L)', 'Age (years)', 'Sex',
                            'Weight (kg)', 'Height (m)', 'Age Normalised Weight', 'Age Normalised Height']
else:
    point_cols = ['ALT (IU/L)', 'Albumin (g/L)', 'Alkaline Phosphatase (U/L)', 'AST (IU/L)', 'Aspartate (U/L)',
                  'Amylase (U/L)', 'APTT (s)', 'Anion gap', 'Base excess (mEq/L)', 'Basophils (x10^9/L)', 'Bicarbonate (mmol/L)', 'pH', 'Blood culture performed',
                  'Creatinine (micromol/L)', 'CRP (mg/L)', 'Calcium (mmol/L)', 'Chloride (mmol/L)', 'Eosinophils (x10^9/L)', 'Fraction of Deoxyhaemoglobin',
                  'Methemoglobin (%)', 'Fraction of Oxygenated Haemoglobin', 'Glucose (mmol/L)', 'HCT (%)', 'Haematocrit (L/L)', 'INR',
                  'Lactate (mmol/L)', 'Lymphocytes (x10^9/L)', 'Magnesium (mmol/L)', 'Monocytes (x10^9/L)', 'Neutrophils (x10^9/L)', 'P50 (mmHg)',
                  'PaCO2 (kPa)', 'PcCO2 (kPa)', 'PmCO2 (kPa)', 'PaO2 (kPa)', 'PcO2 (kPa)', 'PmO2 (kPa)', 'PO2 (kPa)', 'PvCO2 (kPa)',
                  'Phosphate (mmol/L)', 'Platelets (x10^9/L)', 'Potassium (mmol/L)', 'PT (s)', 'Reticulocytes (%)', 'Sodium (mmol/L)', 
                  'TT (s)', 'Bilirubin (micromol/L)', 'White Cell Count (x10^9/L)', 'Strong Ion Gap (mEq/L)', 'Age (years)', 'Sex', 'Weight (kg)', 'Height (m)']
    series_cols = ['Ventilation Status', 'High Frequency Oscillatory Ventilation',
                   'Tracheostomy Status', 'End Tidal CO2 (mmHg)', 'FiO2', 'Oxygen Flow Rate (L/min)',
                   'IPAP (cmH2O)', 'Ventilation (ml/min)',
                   'Mean Airway Pressure', 'EPAP (cmH2O)', 'Ventilation Status Frequency of Input',
                   'Oxygen Flow Rate Missingess', 'IPAP Frequency of Input', 'EPAP Frequency of Input',
                   'FiO2 Frequency of Input', 'High Frequency Oscillatory Ventilation Frequency of Input',
                   'Tracheostomy status Frequency of Input', 'Ventilation (ml/min) Frequency of Input',
                   'Mean Airway Pressure Frequency of Input', 'End Tidal CO2 Frequency of Input', 'Systolic BP',
                   'Diastolic BP', 'Mean Arterial Pressure', 'Systolic BP Frequency of Input', 'Diastolic BP Frequency of Input',
                   'Mean Arterial Pressure Frequency of Input', 'Heart Rate', 'Heart Rate Frequency of Input', 'Comfort Score (Alertness)', 
                   'Comfort Score (BP)', 'Comfort Score (Calmness)', 'Comfort Score', 'Comfort Score (HR)',
                   'Comfort Score (Resp)', 'AVPU', 'GCS (Verbal)', 'GCS (Eyes)', 'GCS (Motor)',
                   'GCS', 'GCS Frequency of Input', 'AVPU Frequency of Input', 'Capillary Refill Time (s)', 'Capillary Refill Time Frequency of Input',
                   'Oxygen Saturation (%)', 'Oxygen Saturation Frequency of Input', 'Extra-Corporeal Membrane Oxygenation Status',
                   'Extra-Corporeal Membrane Oxygenation Status Frequency of Input', 'Vasoactive Inotropic Score (mcg/kg/min)',
                   'Vasopressin Equivalent dose (mcg/kg/min)', 'Dopamine Equivalent Dose (mcg/kg/min)',
                   'Total Noradrenaline and Adrenaline Dose (mcg/kg/min)', 'Milrinone Equivalent Dose (mcg/kg/min)',
                   'Inotropes Frequency of Input', 'Vasopressin Frequency of Input', 'Dopamine Frequency of Input', 'Noradrenaline and Adrenaline Frequency of Input',
                   'Milrinone Frequency of Input', 'SpO2:FiO2 ratio', 'pSOFA Score (Respiratory)', 'pSOFA Score (Cardiac)',
                   'pSOFA Score (Platelets)', 'pSOFA Score (Bilirubin)', 'pSOFA Score (GCS)',
                   'pSOFA Score (Creatinine)', 'Respiratory Rate', 'Respiratory Rate Frequency of Input',
                   'Dialysis Status', 'Dialysis Frequency of Input', 'pSOFA Score', 'Temperature (F)', 'Temperature Frequency of Input',
                   'Urine Output (ml)', 'Urine Output Frequency of Input', 'Paediatric Early Warning Score']

#Combine the series cols objects and name them for labelling of axes
series_cols_mean = ['Mean: ' + i for i in series_cols]
series_cols_std = ['Variability: ' + i for i in series_cols]
series_cols_slopes = ['Trend: ' + i for i in series_cols]
series_cols_R = ['Strength of Trend: ' + i for i in series_cols]
all_cols = series_cols_mean
all_cols.extend(series_cols_std)
all_cols.extend(series_cols_slopes)
all_cols.extend(series_cols_R)

#Now run and get parameters
model1 = XGBClassifier(objective='binary:logistic', eval_metric = 'aucpr', use_label_encoder=False, n_jobs = 32, **param_grid)
X_test = np.concatenate((test_array2d, test_characteristics, test_slopes, test_R), axis=1)
y_test = np.argmax(binary_deterioration_test_outcomes, axis = 1)

#Don't need to rerun 10 times 
accuracy = list()
MSE = list()
AUROC = list()
MAE = list()
Precision = list()
Recall = list()
F1 = list()
AUPRC = list()

#Need to one hot encode
onehot_encoder = sklearn.preprocessing.OneHotEncoder()
integer_encoded_test = y_test.reshape(len(y_test), 1)
onehot_encoded_test = onehot_encoder.fit_transform(integer_encoded_test)

#Run the model
clf1 = model1.fit(X, y)
y_pred = clf1.predict(X_test)
y_pred_proba = clf1.predict_proba(X_test)

integer_encoded_pred = y_pred.reshape(len(y_pred), 1)
onehot_encoded_pred = onehot_encoder.fit_transform(integer_encoded_pred)

#Save the outcomes
accuracy.append(accuracy_score(y_test, y_pred))
MSE.append(mean_squared_error(y_test, y_pred))
MAE.append(mean_absolute_error(y_test, y_pred))
AUROC.append(roc_auc_score(y_test, y_pred_proba[:, 1]))
Recall.append(recall_score(y_test, y_pred))
Precision.append(precision_score(y_test, y_pred))
F1.append(f1_score(y_test, y_pred))
AUPRC.append(average_precision_score(y_test, y_pred_proba[:,1]))

#Calculate precision at recall 0.9
prec, recall, thresholds_prc = precision_recall_curve(y_test, y_pred_proba[:,1], pos_label=clf1.classes_[1])
pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()
thresholds_prc = np.append([0], thresholds_prc) #Thresholds is n-1 length of precision and recall
threshold_90 = thresholds_prc[recall > 0.9][-1]

#Now  calculate F1 at that level - should get precision at level of recall
y_pred_tuned = (y_pred_proba[:,1] > threshold_90).astype(int)
recall_90 = recall_score(y_test, y_pred_tuned)
i = 1
while recall_90 < 0.9:
    #Make sure that recall definitely above 0.9
    y_pred_tuned = (y_pred_proba[:,1] > threshold_90).astype(int)
    recall_90 = recall_score(y_test, y_pred_tuned)
    i += 1

f1_at_90 = f1_score(y_test, y_pred_tuned)
precision_at_90 = precision_score(y_test, y_pred_tuned)


#Save to a json file
results = {'acc_PEWS' : np.mean(accuracy),
            'AUC_PEWS' : np.mean(AUROC),
            'MSE_PEWS' : np.mean(MSE),
            'MAE_PEWS' : np.mean(MAE), 
            'precision_PEWS' : np.mean(Precision), 
            'recall_PEWS' : np.mean(Recall), 
            'F1_PEWS' : np.mean(F1), 
            'AUPRC_PEWS' : np.mean(AUPRC), 
            'precision_at_recall_90': precision_at_90,
            'f1_at_recall_90': f1_at_90, 
            'recall_score_closest_90': recall_90}

res_path_name = 'files/new_res/'

results_df = pd.DataFrame({'metrics': results.keys(), 'results': results.values()})
results_df.to_csv(res_path_name + 'XGBoost_' + file_suffix + '.csv')

def return_results_df(X, y, X_test, y_test, model, reps = 1):
    """Making this into a function as a seem to use it a lot"""
    
    #Now need to rerun reps times 
    accuracy = list()
    MSE = list()
    AUROC = list()
    MAE = list()
    Precision = list()
    Recall = list()
    F1 = list()
    AUPRC = list()
    precision_at_90 = list()
    f1_at_90 = list()
    recall_at_90 = list()
        
    for i in range(reps):    
        #Run the model
        clf1 = model.fit(X, y)
        y_pred = clf1.predict(X_test)
        y_pred_proba = clf1.predict_proba(X_test)

        #Save the outcomes
        accuracy.append(accuracy_score(y_test, y_pred))
        MSE.append(mean_squared_error(y_test, y_pred))
        MAE.append(mean_absolute_error(y_test, y_pred))
        AUROC.append(roc_auc_score(y_test, y_pred_proba[:, 1]))
        Recall.append(recall_score(y_test, y_pred))
        Precision.append(precision_score(y_test, y_pred))
        F1.append(f1_score(y_test, y_pred))
        AUPRC.append(average_precision_score(y_test, y_pred_proba[:,1]))

        #Calculate precision at recall 0.9
        _, recall, thresholds_prc = precision_recall_curve(y_test, y_pred_proba[:,1], pos_label=clf1.classes_[1])
        thresholds_prc = np.append([0], thresholds_prc) #Thresholds is n-1 length of precision and recall
        threshold_90 = thresholds_prc[recall > 0.9][-1]

        #Now  calculate F1 at that level - should get precision at level of recall
        y_pred_tuned = (y_pred_proba[:,1] > threshold_90).astype(int)
        recall_90 = recall_score(y_test, y_pred_tuned)
        i = 1
        while recall_90 < 0.9:
            #Make sure that recall definitely above 0.9
            y_pred_tuned = (y_pred_proba[:,1] > threshold_90).astype(int)
            recall_90 = recall_score(y_test, y_pred_tuned)
            i += 1
        
        recall_at_90.append(recall_90)
        f1_at_90.append(f1_score(y_test, y_pred_tuned))
        precision_at_90.append(precision_score(y_test, y_pred_tuned))


    #Save to a json file
    results = {'acc_PEWS' : [np.mean(accuracy), np.std(accuracy)],
            'AUC_PEWS' : [np.mean(AUROC), np.std(AUROC)],
            'MSE_PEWS' : [np.mean(MSE), np.std(MSE)],
            'MAE_PEWS' : [np.mean(MAE), np.std(MAE)], 
            'precision_PEWS' : [np.mean(Precision), np.std(Precision)], 
            'recall_PEWS' : [np.mean(Recall), np.std(Recall)], 
            'F1_PEWS' : [np.mean(F1), np.std(F1)], 
            'AUPRC_PEWS' : [np.mean(AUPRC), np.std(AUPRC)], 
            'precision_at_recall_90': [np.mean(precision_at_90), np.std(precision_at_90)],
            'f1_at_recall_90': [np.mean(f1_at_90), np.std(f1_at_90)], 
            'recall_score_closest_90': [np.mean(recall_at_90), np.std(recall_at_90)]}

    results_df = pd.DataFrame(results).transpose()
    results_df.columns = ['Mean', 'SD']
    return results_df

#Now need to rerun 10 times 
accuracy = list()
MSE = list()
AUROC = list()
MAE = list()
Precision = list()
Recall = list()
F1 = list()
AUPRC = list()
precision_at_90 = list()
f1_at_90 = list()
recall_at_90 = list()
model2 = LogisticRegression(random_state=0, max_iter= 1000, multi_class='multinomial', solver='lbfgs')

for i in range(10):
    #Run the model
    clf2 = model2.fit(X, y)
    y_pred2 = clf2.predict(X_test)
    y_pred_proba2 = clf2.predict_proba(X_test)
    integer_encoded_pred2 = y_pred2.reshape(len(y_pred2), 1)
    onehot_encoded_pred2 = onehot_encoder.fit_transform(integer_encoded_pred2)

    #Save the outcomes
    accuracy.append(accuracy_score(y_test, y_pred2))
    MSE.append(mean_squared_error(y_test, y_pred2))
    MAE.append(mean_absolute_error(y_test, y_pred2))
    AUROC.append(roc_auc_score(y_test, y_pred_proba2[:, 1]))
    Recall.append(recall_score(y_test, y_pred2))
    Precision.append(precision_score(y_test, y_pred2))
    F1.append(f1_score(y_test, y_pred2))
    AUPRC.append(average_precision_score(y_test, y_pred_proba2[:,1]))

    #Calculate precision at recall 0.9
    prec2, recall2, thresholds_prc2 = precision_recall_curve(y_test, y_pred_proba2[:,1], pos_label=clf2.classes_[1])
    pr_display2 = PrecisionRecallDisplay(precision=prec2, recall=recall2).plot()
    thresholds_prc2 = np.append([0], thresholds_prc2) #Thresholds is n-1 length of precision and recall
    threshold_902 = thresholds_prc2[recall2 > 0.9][-1]

    #Now  calculate F1 at that level - should get precision at level of recall
    y_pred_tuned2 = (y_pred_proba2[:,1] > threshold_902).astype(int)
    recall_902 = recall_score(y_test, y_pred_tuned2)
    i = 1
    while recall_902 < 0.9:
        #Make sure that recall definitely above 0.9
        y_pred_tuned2 = (y_pred_proba2[:,1] > threshold_902).astype(int)
        recall_902 = recall_score(y_test, y_pred_tuned2)
        i += 1
        
    recall_at_90.append(recall_902)

    f1_at_90.append(f1_score(y_test, y_pred_tuned2))
    precision_at_90.append(precision_score(y_test, y_pred_tuned2))


#Save to a json file
results = {'acc_PEWS' : [np.mean(accuracy), np.std(accuracy)],
            'AUC_PEWS' : [np.mean(AUROC), np.std(AUROC)],
            'MSE_PEWS' : [np.mean(MSE), np.std(MSE)],
            'MAE_PEWS' : [np.mean(MAE), np.std(MAE)], 
            'precision_PEWS' : [np.mean(Precision), np.std(Precision)], 
            'recall_PEWS' : [np.mean(Recall), np.std(Recall)], 
            'F1_PEWS' : [np.mean(F1), np.std(F1)], 
            'AUPRC_PEWS' : [np.mean(AUPRC), np.std(AUPRC)], 
            'precision_at_recall_90': [np.mean(precision_at_90), np.std(precision_at_90)],
            'f1_at_recall_90': [np.mean(f1_at_90), np.std(f1_at_90)], 
            'recall_score_closest_90': [np.mean(recall_at_90), np.std(recall_at_90)]}

res_path_name = 'files/new_res/'

results_df = pd.DataFrame({'metrics': results.keys(), 'results': results.values()})
results_df.to_csv(res_path_name + 'Logistic_regression_' + file_suffix + '.csv')



### Save some relevant plots 
#Importance plot
feature_importance1 = clf1.feature_importances_
#Get top 20 most important features
feature_order1 = np.argsort(-1*feature_importance1)
feature_names1 =  point_cols + all_cols
top_features_names1 = [feature_names1[i] for i in feature_order1[0:20]]
top_feature_values1 = [feature_importance1[i] for i in feature_order1[0:20]]

#Save order of features for the best model:
pd.DataFrame([feature_names1[i] for i in feature_order1]).to_csv('files/new_feature_order_XGB' + file_suffix + '.csv')

#Horizontal bar plot
fig, ax = plt.subplots(1, 1, figsize = (10, 7))
ax.barh(top_features_names1, top_feature_values1, color = 'red')
ax.set_yticks(np.arange(len(top_features_names1)))
ax.set_yticklabels(top_features_names1)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Gain (feature importance)')
ax.set_title(f'Feature importance for XGBoost - {length}h model')
plt.subplots_adjust(left=0.3)

figs_path_name = 'figs/PICU/XGBoost/new/'
fig.savefig(figs_path_name + 'XGBoost_importance_' + file_suffix + '.pdf', format="pdf", bbox_inches = 'tight')


#Work through the number of features and rerun the model with features in order of importance 
# (using gain so as not to post-hoc bias as SHAP uses the test set)
test_results_length = False
if test_results_length:
    
    #Sort the features
    for i in tqdm(range(len(feature_names1))):
        X_len = X[:, feature_order1[:(i + 1)]]
        X_test_len = X_test[:, feature_order1[:(i + 1)]]
        short_results_df = return_results_df(X_len, y, X_test_len, y_test, model1)
        try:
            all_short_results_df = pd.concat([all_short_results_df, short_results_df.Mean], axis = 1)
        except:
            all_short_results_df = short_results_df.Mean
            
    all_short_results_df.columns = [f'input_{i+1}' for i in range(363)]
    all_short_results_df.to_csv(res_path_name + 'XBG_input_number_' + file_suffix + '.csv')
    
    fig, ax = plt.subplots(1,1)
    ax.plot([i + 1 for i in range(all_short_results_df.shape[1])], 
            np.array(all_short_results_df.transpose().AUPRC_PEWS), 
            color = 'blue', label='AUPRC')
    ax.plot([i + 1 for i in range(all_short_results_df.shape[1])], 
            np.array(all_short_results_df.transpose().AUC_PEWS), 
            color = 'red', label = 'AUROC')
    ax.plot([i + 1 for i in range(all_short_results_df.shape[1])], 
            np.array(all_short_results_df.transpose().precision_at_recall_90), 
            color = 'green', label = 'Precision at Recall 0.9')
    ax.set_xlabel('Number of Input Features')
    ax.set_ylabel('AUPRC')
    ax.legend()
    ax.set_title(f'Model Performance against number of Input features\n{length}h model')
    fig.savefig(figs_path_name + 'Input_features_vs_performance' + file_suffix + '.pdf', format="pdf")
    
        

### Save some relevant plots 
#Importance plot
feature_importance2 = clf2.coef_
#Get top 20 most important features
feature_order2 = np.argsort(-1*feature_importance2)
top_features_names2 = [feature_names1[i] for i in feature_order2[0][0:20]]
top_feature_values2 = [feature_importance2[0][i] for i in feature_order2[0][0:20]]

#Save order of features for the best model:
pd.DataFrame([feature_names1[i] for i in feature_order2[0]]).to_csv('files/new_feature_order_LR' + file_suffix + '.csv')

#Horizontal bar plot
fig, ax = plt.subplots(1, 1, figsize = (10, 7))
ax.barh(top_features_names2, top_feature_values2)
ax.set_yticks(np.arange(len(top_features_names2)))
ax.set_yticklabels(top_features_names2)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Gain (feature importance)')
ax.set_title(f'Feature importance for Logistic Regression \n{length}h model')
plt.subplots_adjust(left=0.3)

LR_figs_path_name = 'figs/PICU/Logistic_regression/new_figs/'
fig.savefig(LR_figs_path_name + 'Logistic_regression_' + file_suffix + '.pdf', format="pdf")


#### Do some SHAP plotting
# # make sure the SHAP values add up to marginal predictions
#Summary plot
explainer = shap.TreeExplainer(clf1)
explainer.feature_names = feature_names1
shap_values = explainer.shap_values(X_test, check_additivity=False)

#Pure features
shap_explainer = explainer(X_test, check_additivity=False)
shap_explainer.feature_names = feature_names1

#Save the shap values as a pickle file so we can recreate these 
shap_directory = '/Users/danstein/Documents/Masters/Course materials/Project/PICU project/new_project_files/shap_pickle/new_shaps/'
try: 
    with open(shap_directory + file_suffix + '.pkl', 'rb') as file:
        shap_values = pickle.load(file)
except:
    with open(shap_directory + file_suffix + '.pkl', 'wb') as file:
        pickle.dump(shap_values, file)
    
#Save the shap values as a pickle file so we can recreate these 
try: 
    with open(shap_directory + 'Explainer_' + file_suffix + '.pkl', 'rb') as file:
        explainer = pickle.load(file)
except:
    with open(shap_directory + 'Explainer_' + file_suffix + '.pkl', 'wb') as file:
        pickle.dump(explainer, file)

try: 
    with open(shap_directory + 'Shap_explainer_' + file_suffix + '.pkl', 'rb') as file:
        shap_explainer = pickle.load(file)
except:
    with open(shap_directory + 'Shap_explainer_' + file_suffix + '.pkl', 'wb') as file:
        pickle.dump(shap_explainer, file)    

#Invert missingness (mean)
mean_missingness_features = {j: i for i, j in enumerate(feature_names1) if re.search('Frequency of Input', j) and re.search('Mean:', j)}
foo_values = 1 - shap_explainer.data[:, [i for i in mean_missingness_features.values()]]
shap_explainer.data[:, [i for i in mean_missingness_features.values()]] = foo_values

fig, ax = plt.subplots(1, 1, figsize = (10, 7))
shap_plot1 = shap.summary_plot(shap_values, shap_explainer.data, feature_names = feature_names1, title = f'SHAP values for model predicting deterioration within {length}h', show=False, 
                               plot_size = (10, 8)
)
plt.savefig(figs_path_name + 'SHAP_xgboost_inverted' + file_suffix + '.pdf', format="pdf", bbox_inches = 'tight')


fig, ax = plt.subplots(1, 1, figsize = (15, 15))
shap_plot2 = shap.plots.bar(shap_explainer, max_display=30, show = False)
plt.savefig(figs_path_name + 'SHAP_importances_xgboost_' + file_suffix + '.pdf', bbox_inches = 'tight', format="pdf")

plotting_shap = False
if plotting_shap:
    #Find highest features
    average_shap = np.mean(shap_explainer.values, axis = 0)
    sorted_shap_features = [feature_names1[i] for i in np.argsort(-np.abs(average_shap))]

    i = 0
    shap.plots.scatter(shap_explainer[:, 'Na.'], color = shap_explainer[:, sorted_shap_features[i]], 
                    xmin = 120, xmax = 165); i+=1

    i = 0
    shap.plots.scatter(shap_explainer[:, 'Cr'], color = shap_explainer[:, 'Mean pSOFA_Cr']); i+=1

    #Plot MAP against SHAP values (for pSOFA)
    fig, ax = plt.subplots(tight_layout=True, figsize=(7, 5))
    map_loc = [i for i, j in enumerate(feature_names1) if j == 'Mean: Mean Arterial Pressure']
    psofa_loc = [i for i, j in enumerate(feature_names1) if j == 'Mean: pSOFA Score']
    shap.plots.scatter(shap_explainer[:, map_loc], color = shap_explainer[:, psofa_loc], ax = ax, xmax = 130, show = False)
    
    #Set x-axis limit
    ax.set_xlim(0, 130)  # or whatever limits you prefer

    plt.show()

    #Needs running with raw values
    fig, ax = plt.subplots(tight_layout=True, figsize=(7, 5))
    shap.plots.scatter(shap_explainer[:, 'Mean MAP'], color = shap_explainer[:, 'Mean pSOFA_cardio'], xmax = 130, ax = ax, show = False)
    ax.set_xlabel('Mean MAP (Mean Arterial Pressure, mmHg)')
    ax.set_ylabel('SHAP Value for \nMean MAP')
    cbar = ax.collections[0].colorbar
    #cbar = plt.gcf().axes[-1]
    cbar.set_label('Mean pSOFA (Cardiac)')#, rotation=90, labelpad=15)
    plt.show()

    #Plot platelets against pSOFA (plt)
    fig, ax = plt.subplots(tight_layout=True, figsize=(7, 5))
    shap.plots.scatter(shap_explainer[:, 'Plts'], color = shap_explainer[:, 'Mean pSOFA_plt'], ax = ax, show = False)
    ax.set_xlabel('Platelet count (10^9/L)')
    ax.set_ylabel('SHAP Value for \nPlatelet count')

    # Create a colorbar object
    cbar = ax.collections[0].colorbar

    # Set the colorbar label and tick labels
    cbar.set_label('Mean pSOFA (Platelets)', rotation=90, labelpad=15)
    #cbar.set_ticks([0, 1, 2, 3, 4])
    #cbar.set_ticklabels(['0', '1', '2', '3', '4'])

    # Display the plot
    plt.show()

    #Plot creatinine against pSOFA (Cr)
    fig, ax = plt.subplots(tight_layout=True, figsize=(7, 5))
    shap.plots.scatter(shap_explainer[:, 'Cr'], color = shap_explainer[:, 'Mean pSOFA_Cr'], ax = ax, show = False, xmax = 300)
    ax.set_xlabel('Serum Creatinine (micromol/L)')
    ax.set_ylabel('SHAP Value for \nSerum Creatinine')

    # Create a colorbar object
    cbar = ax.collections[0].colorbar

    # Set the colorbar label and tick labels
    cbar.set_label('Mean pSOFA (Creatinine)', rotation=90, labelpad=15)
    #cbar.set_ticks([0, 1, 2, 3, 4])
    #cbar.set_ticklabels(['0', '1', '2', '3', '4'])

    # Display the plot
    plt.show()

    #Plot bili against pSOFA (bili)
    fig, ax = plt.subplots(tight_layout=True, figsize=(7, 5))
    shap.plots.scatter(shap_explainer[:, 'Bili'], color = shap_explainer[:, 'Mean pSOFA_bili'], ax = ax, show = False)
    ax.set_xlabel('Serum Bilirubin (micromol/L)')
    ax.set_ylabel('SHAP Value for \nSerum Bilirubin')

    # Create a colorbar object
    cbar = ax.collections[0].colorbar

    # Set the colorbar label and tick labels
    cbar.set_label('Mean pSOFA (Bilirubin)', rotation=90, labelpad=15)
    #cbar.set_ticks([0, 1, 2, 3, 4])
    #cbar.set_ticklabels(['0', '1', '2', '3', '4'])

    # Display the plot
    plt.show()

    #Plot Na against pSOFA
    fig, ax = plt.subplots(tight_layout=True, figsize=(7, 5))
    shap.plots.scatter(shap_explainer[:, 'Na.'], color = shap_explainer[:, 'Mean pSOFA'], ax = ax, show = False)
    ax.set_xlabel('Serum Sodium (mmol/L)')
    ax.set_ylabel('SHAP Value for \nSerum Sodium')
    plt.show()

    #Plot cl against pSOFA
    fig, ax = plt.subplots(tight_layout=True, figsize=(7, 5))
    shap.plots.scatter(shap_explainer[:, 'Cl'], color = shap_explainer[:, 'Mean pSOFA'], ax = ax, show = False)
    ax.set_xlabel('Serum Chloride (mmol/L)')
    ax.set_ylabel('SHAP Value for \nSerum Chloride')
    plt.show()

    #Plot SIG against pSOFA
    fig, ax = plt.subplots(tight_layout=True, figsize=(7, 5))
    shap.plots.scatter(shap_explainer[:, 'Strong_ion_gap'], color = shap_explainer[:, 'Mean pSOFA'], ax = ax, show = False)
    ax.set_xlabel('Serum Strong Ion Gap (mEq/L)')
    ax.set_ylabel('SHAP Value for \nSerum SIG')
    plt.show()

    #Plot Comfort.Alertness against pSOFA
    fig, ax = plt.subplots(tight_layout=True, figsize=(7, 5))
    shap.plots.scatter(shap_explainer[:, 'Mean Comfort.Alertness'], color = shap_explainer[:, 'Mean pSOFA'], ax = ax, show = False)
    ax.set_xlabel('Mean Comfort Score (Alertness)')
    ax.set_ylabel('SHAP Value for \nMean Comfort (Alertness)')
    plt.show()

    #Plot psofa (cardio) against pSOFA
    fig, ax = plt.subplots(tight_layout=True, figsize=(7, 5))
    shap.plots.scatter(shap_explainer[:, 'Mean pSOFA_cardio'], color = shap_explainer[:, 'Mean pSOFA'], ax = ax, show = False)
    ax.set_xlabel('Mean pSOFA (Cardiac)')
    ax.set_ylabel('SHAP Value for \nMean pSOFA (Cardiac)')
    plt.show()

    #STD SBP zscore, dbp missing (1), cr, comfort alertness, mean map zscore, mean sbp zscore, cl, alkphos, bili, mean comfort hr, na, mean SF ratio
    shap.plots.scatter(shap_explainer[:, 'Mean Comfort.Alertness'], color = shap_explainer[:, 'Mean Comfort']); i+=1
    #Chloride has a big split, std rr zscore

    #Shap waterfall and force plots
    fig, ax = plt.subplots(1, 1, figsize = (8, 6))
    ax = plt.scatter(x = X_test[:, [i for i, j in enumerate(feature_names1) if j == 'Age_yrs']], 
                    y = X_test[:, [i for i, j in enumerate(feature_names1) if j == 'Mean MAP']],
                    c = shap_values[:, [i for i, j in enumerate(feature_names1) if j == 'Mean MAP']], 
                    cmap='cool', alpha = 0.3, s = 50)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('SHAP value for MAP', rotation=90)
    plt.gca().set_ylim(20, 130)
    plt.xlabel('Age (yrs)')
    plt.ylabel('Mean Arterial Pressure (mmHg)')
    fig.show()


    shap.plots.scatter(shap_explainer[:, 'STD DiaBP_missing'],
                    color = shap_explainer[:, 'Mean DiaBP_missing'], x_jitter = 2)
    shap.plots.scatter(shap_explainer[:, 'Mean MAP'],
                    color = shap_explainer[:, 'Age_yrs'], x_jitter = 2, #dot_size=2, 
                    xmin=30, xmax=125, cmap = 'cool'
                    )
    shap.plots.scatter(shap_explainer[:, 'Mean DiaBP'],
                    color = shap_explainer[:, 'Age_yrs'], x_jitter = 2, #dot_size=2, 
                    #xmin=30, xmax=125
                    )
    shap.plots.scatter(shap_explainer[:, 'STD pSOFA_cardio'],
                    color = shap_explainer[:, 'Mean pSOFA_cardio'], x_jitter = 2)
    shap.plots.scatter(shap_explainer[:, 'Mean AVPU'],
                    color = shap_explainer[:, 'Mean Comfort.Alertness'], x_jitter = 2)
    shap.plots.scatter(shap_explainer[:, 'Mean pSOFA_cardio'],
                    color = shap_explainer[:, 'STD pSOFA_cardio'], x_jitter = 2)
    shap.plots.scatter(shap_explainer[:, 'Mean Comfort.Alertness'],
                    color = shap_explainer[:, 'STD Comfort.Alertness'], x_jitter = 2)
    shap.plots.scatter(shap_explainer[:, 'Mean Comfort'],
                    color = shap_explainer[:, 'Mean Comfort.Alertness'], x_jitter = 2)
    shap.plots.scatter(shap_explainer[:, 'Mean FiO2'],
                    color = shap_explainer[:, 'Mean SpO2'], x_jitter = 2, xmin = 20)
    shap.plots.scatter(shap_explainer[:, 'Age_yrs'],
                    color = shap_explainer[:, 'interpolated_ht_m'], x_jitter = 2)
    i = 0
    for i in range(26, len(feature_names1)):
        shap.plots.scatter(shap_explainer[:, 'Strong_ion_gap'], color = shap_explainer[:, feature_names1[i]], x_jitter = 2)
    shap.plots.scatter(shap_explainer[:, 'Bili'])
    shap.plots.scatter(shap_explainer[:, 'Bili'], color = shap_explainer[:, 'pH'])
    shap.plots.scatter(shap_explainer[:, 'pH'], color = shap_explainer[:, 'pH'])
    shap.plots.scatter(shap_explainer[:, 'Cr'], color = shap_values[:, [i for i, j in enumerate(feature_names1) if j == 'Cr'][0]])
    shap.plots.scatter(shap_explainer[:, 'Cr'], color = shap_explainer[:, 'pH'],
                    xmax = 150)


    #Some waterfall plots
    #True deterioration
    np.intersect1d(np.where(y_pred == 1)[0], 
                np.where(y_test == 1)[0])
    shap.plots.waterfall(shap_explainer[710])

#Plot precision recall curve
fig, ax = plt.subplots(1, 1, figsize = (15, 11))
pr_display1 = PrecisionRecallDisplay(precision=prec, recall=recall)
pr_display1.plot().figure_.savefig(figs_path_name + 'PRC_xgboost_' + file_suffix + '.pdf', bbox_inches = 'tight', format="pdf")

#Matched precision_recall curves:
clf_pSOFA = LogisticRegression(random_state=0, max_iter= 1000, multi_class='multinomial', solver='lbfgs', penalty = 'l2')
X_psofa = np.max(train_array3d[:, np.array(series_cols) == 'pSOFA Score', :], axis = 2)
X_test_psofa = np.max(test_array3d[:, np.array(series_cols) == 'pSOFA Score', :], axis = 2)
clf_pSOFA.fit(X_psofa, y)
y_predict_pSOFA = clf_pSOFA.predict_proba(X_test_psofa)
prec_pSOFA, recall_pSOFA, thresholds_prc = precision_recall_curve(binary_deterioration_test_outcomes[:,1], y_predict_pSOFA[:,1], pos_label=clf1.classes_[1])
pr_display_pSOFA = PrecisionRecallDisplay(precision=prec_pSOFA, recall=recall_pSOFA)
pr_display1 = PrecisionRecallDisplay(precision=prec, recall=recall)

pSOFA_results_XGB = return_results_df(X_psofa, y, X_test_psofa, y_test, XGBClassifier())
pSOFA_results_XGB.to_csv(res_path_name + 'pSOFA_results_XGB_' + file_suffix + '.csv')
pSOFA_results_LR = return_results_df(X_psofa, y, X_test_psofa, y_test, clf_pSOFA, reps = 10)
pSOFA_results_LR.to_csv(res_path_name + 'pSOFA_results_LR_' + file_suffix + '.csv')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
#ax1.set_ylim((0.7, 1))
#ax2.set_ylim((0.7, 1))
ax1.title.set_text('PRC for XGBoost Model')
ax2.title.set_text('PRC for pSOFA Model')
pr_display1.plot(ax=ax1)
pr_display_pSOFA.plot(ax=ax2)

plt.savefig(figs_path_name + 'Paired_PRC_' + file_suffix + '.pdf', bbox_inches = 'tight', format="pdf")

### Run the XGBoost model with components of pSOFA and variability etc
pSOFA_names_max = [re.search('pSOFA Score ', i)!= None for i in series_cols]
X_psofa_ind = np.max(train_array3d[:, pSOFA_names_max, :], axis = 2)
X_test_psofa_ind = np.max(test_array3d[:, pSOFA_names_max, :], axis = 2)
pSOFA_results_ind_XGB = return_results_df(X_psofa_ind, y, X_test_psofa_ind, y_test, XGBClassifier())
pSOFA_results_ind_XGB.to_csv(res_path_name + 'pSOFA_results_XGB_individual_inputs_' + file_suffix + '.csv')
pSOFA_results_ind_LR = return_results_df(X_psofa_ind, y, X_test_psofa_ind, y_test, clf_pSOFA, reps = 10)
pSOFA_results_ind_LR.to_csv(res_path_name + 'pSOFA_results_LR_individual_inputs_' + file_suffix + '.csv')

#other way of calculating worst pSOFA?
X_psofa_alt = np.sum(X_psofa_ind, axis =1).reshape(X_psofa_ind.shape[0], 1)
X_test_psofa_alt = np.sum(X_test_psofa_ind, axis =1).reshape(X_test_psofa_ind.shape[0], 1)
pSOFA_results_alt_XGB = return_results_df(X_psofa_alt, y, X_test_psofa_alt, y_test, XGBClassifier())
pSOFA_results_alt_XGB.to_csv(res_path_name + 'pSOFA_alt_results_XGB_' + file_suffix + '.csv')
pSOFA_results_alt_LR = return_results_df(X_psofa_alt, y, X_test_psofa_alt, y_test, clf_pSOFA, reps = 10)
pSOFA_results_alt_LR.to_csv(res_path_name + 'pSOFA_alt_results_LR_' + file_suffix + '.csv')

pSOFA_names = [re.search('pSOFA.*', i)!= None for i in feature_names1]
X_psofa_ind_vars = X[:, pSOFA_names]
X_test_psofa_ind_vars = X_test[:, pSOFA_names]
pSOFA_results_ind_vars_XGB = return_results_df(X_psofa_ind_vars, y, X_test_psofa_ind_vars, y_test, XGBClassifier())
pSOFA_results_ind_vars_XGB.to_csv(res_path_name + 'pSOFA_results_XGB_individual_inputs_vars_' + file_suffix + '.csv')
pSOFA_results_ind_vars_LR = return_results_df(X_psofa_ind_vars, y, X_test_psofa_ind_vars, y_test, clf_pSOFA, reps = 10)
pSOFA_results_ind_vars_LR.to_csv(res_path_name + 'pSOFA_results_LR_individual_inputs_vars_' + file_suffix + '.csv')

#Results without missingness etc
no_missing_names = [re.search('Frequency of Input', i) == None for i in feature_names1]
X_no_missingness = X[:, no_missing_names]
X_test_no_missingness = X_test[:, no_missing_names]
no_missingness_results_XGB = return_results_df(X_no_missingness, y, X_test_no_missingness, y_test, XGBClassifier())
no_missingness_results_XGB.to_csv(res_path_name + 'pSOFA_results_XGB_no_missingness_' + file_suffix + '.csv')
no_missingness_results_LR = return_results_df(X_no_missingness, y, X_test_no_missingness, y_test, clf_pSOFA)
no_missingness_results_LR.to_csv(res_path_name + 'pSOFA_results_LR_no_missingness_' + file_suffix + '.csv')

#Results with only mean and variability
X_mean_var = np.concatenate((train_array2d, train_characteristics), axis=1)
X_test_mean_var = np.concatenate((test_array2d, test_characteristics), axis=1)
mean_var_XGB = return_results_df(X_mean_var, y, X_test_mean_var, y_test, XGBClassifier())
mean_var_XGB.to_csv(res_path_name + 'pSOFA_results_XGB_mean_variability_' + file_suffix + '.csv')
mean_var_LR = return_results_df(X_mean_var, y, X_test_mean_var, y_test, clf_pSOFA)
mean_var_LR.to_csv(res_path_name + 'pSOFA_results_LR_mean_variability_' + file_suffix + '.csv')

#Latest value only
X_latest = np.concatenate((train_array2d, train_array3d[:, :, 0]), axis=1)
X_test_latest = np.concatenate((test_array2d, test_array3d[:, :, 0]), axis=1)
latest_XGB = return_results_df(X_latest, y, X_test_latest, y_test, XGBClassifier())
latest_XGB.to_csv(res_path_name + 'pSOFA_results_XGB_latest_only_' + file_suffix + '.csv')
latest_LR = return_results_df(X_latest, y, X_test_latest, y_test, clf_pSOFA)
latest_LR.to_csv(res_path_name + 'pSOFA_results_LR_latest_only_' + file_suffix + '.csv')

#Some experiments with number of features (20, 60 seem to be good ones)
X_top20 = X[:, feature_order1[0:20]]
X_test_top20 = X_test[:, feature_order1[0:20]]
top20_XGB = return_results_df(X_top20, y, X_test_top20, y_test, XGBClassifier())
top20_XGB.to_csv(res_path_name + 'pSOFA_results_XGB_top20_' + file_suffix + '.csv')

#Top 60
X_top60 = X[:, feature_order1[0:60]]
X_test_top60 = X_test[:, feature_order1[0:60]]
top60_XGB = return_results_df(X_top60, y, X_test_top60, y_test, XGBClassifier())
top60_XGB.to_csv(res_path_name + 'pSOFA_results_XGB_top60_' + file_suffix + '.csv')

#Top 70
X_top70 = X[:, feature_order1[0:70]]
X_test_top70 = X_test[:, feature_order1[0:70]]
top70_XGB = return_results_df(X_top70, y, X_test_top70, y_test, XGBClassifier())
top70_XGB.to_csv(res_path_name + 'pSOFA_results_XGB_top70_' + file_suffix + '.csv')

#Reduce to 15 minute max Frequency of Inputs, then calculate mean and SD then 
train_array3d_15min = train_array3d[:, :, [i for i in range(0, train_array3d.shape[2], 15)]]
test_array3d_15min = test_array3d[:, :, [i for i in range(0, test_array3d.shape[2], 15)]]
train_means15min = np.mean(train_array3d_15min, axis = 2)
test_means15min = np.mean(test_array3d_15min, axis = 2)
train_std15min = np.std(train_array3d_15min, axis = 2)
test_std15min = np.std(test_array3d_15min, axis = 2)
X_15min = np.concatenate((train_array2d, train_means15min, train_std15min), axis=1)
X_test_15min = np.concatenate((test_array2d, test_means15min, test_std15min), axis=1)
XGB_15min = return_results_df(X_15min, y, X_test_15min, y_test, XGBClassifier())
XGB_15min.to_csv(res_path_name + 'pSOFA_results_15min_sample_' + file_suffix + '.csv')

#Reduce to 60 minute max Frequency of Inputs, then calculate mean and SD then 
train_array3d_60min = train_array3d[:, :, [i for i in range(0, train_array3d.shape[2], 60)]]
test_array3d_60min = test_array3d[:, :, [i for i in range(0, test_array3d.shape[2], 60)]]
train_means60min = np.mean(train_array3d_60min, axis = 2)
test_means60min = np.mean(test_array3d_60min, axis = 2)
train_std60min = np.std(train_array3d_60min, axis = 2)
test_std60min = np.std(test_array3d_60min, axis = 2)
X_60min = np.concatenate((train_array2d, train_means60min, train_std60min), axis=1)
X_test_60min = np.concatenate((test_array2d, test_means60min, test_std60min), axis=1)
XGB_60min = return_results_df(X_60min, y, X_test_60min, y_test, XGBClassifier())
XGB_60min.to_csv(res_path_name + 'pSOFA_results_60min_sample_' + file_suffix + '.csv')

#Reduce to 30 minute max Frequency of Inputs, then calculate mean and SD then 
train_array3d_30min = train_array3d[:, :, [i for i in range(0, train_array3d.shape[2], 30)]]
test_array3d_30min = test_array3d[:, :, [i for i in range(0, test_array3d.shape[2], 30)]]
train_means30min = np.mean(train_array3d_30min, axis = 2)
test_means30min = np.mean(test_array3d_30min, axis = 2)
train_std30min = np.std(train_array3d_30min, axis = 2)
test_std30min = np.std(test_array3d_30min, axis = 2)
X_30min = np.concatenate((train_array2d, train_means30min, train_std30min), axis=1)
X_test_30min = np.concatenate((test_array2d, test_means30min, test_std30min), axis=1)
XGB_30min = return_results_df(X_30min, y, X_test_30min, y_test, XGBClassifier())
XGB_30min.to_csv(res_path_name + 'pSOFA_results_30min_sample_' + file_suffix + '.csv')

#Reduce to 5 minute max Frequency of Inputs, then calculate mean and SD then 
train_array3d_5min = train_array3d[:, :, [i for i in range(0, train_array3d.shape[2], 5)]]
test_array3d_5min = test_array3d[:, :, [i for i in range(0, test_array3d.shape[2], 5)]]
train_means5min = np.mean(train_array3d_5min, axis = 2)
test_means5min = np.mean(test_array3d_5min, axis = 2)
train_std5min = np.std(train_array3d_5min, axis = 2)
test_std5min = np.std(test_array3d_5min, axis = 2)
X_5min = np.concatenate((train_array2d, train_means5min, train_std5min), axis=1)
X_test_5min = np.concatenate((test_array2d, test_means5min, test_std5min), axis=1)
XGB_5min = return_results_df(X_5min, y, X_test_5min, y_test, XGBClassifier())
XGB_5min.to_csv(res_path_name + 'pSOFA_results_5min_sample_' + file_suffix + '.csv')

### Plot model performance over different amounts of input
#Set up some storage
#Don't need to rerun 10 times 
if not no_z and perc and input_length == 6 and length == 12:
    unique_patients = list()
    total_samples = list()
    accuracy = list()
    MSE = list()
    AUROC = list()
    MAE = list()
    Precision = list()
    Recall = list()
    F1 = list()
    AUPRC = list()
    recall_90 = list()
    f1_at_90 = list()
    precision_at_90 = list()

    ## Need to re-run the splits
    for i in tqdm(range(5, 86, 5)):
        #Split up testing and outcomes
        split_proportions = np.array([i, 85-i, 15])
        split_characteristics = test_trainsplit(characteristics, split_proportions)
        split_array2d = test_trainsplit(array2d, split_proportions)
        split_array3d = test_trainsplit(array3d, split_proportions)
        split_slopes = test_trainsplit(slopes, split_proportions)
        split_R = test_trainsplit(r_values, split_proportions)
        split_outcomes = test_trainsplit(outcomes, split_proportions)

        #Training sets
        train_characteristics = split_characteristics[0]
        train_array3d = split_array3d[0]
        train_array2d = split_array2d[0]
        train_slopes = split_slopes[0]
        train_R = split_R[0]
        train_outcomes = split_outcomes[0]
        
        total_samples.append(train_outcomes.shape[0])
        unique_patients.append(np.unique(train_outcomes[:, 14]).shape[0])

        #Make binary outcomes
        #Make the binary values
        binary_deterioration_train_hour_outcomes = np.array(train_outcomes[:, 12] < length, dtype = int)
        binary_deterioration_train_outcomes = np.transpose(np.array([1- binary_deterioration_train_hour_outcomes, binary_deterioration_train_hour_outcomes]))

        #Set x and y
        X = np.concatenate((train_array2d, train_characteristics, train_slopes, train_R), axis=1)
        y = np.argmax(binary_deterioration_train_outcomes, axis = 1)

        #Need to one hot encode
        onehot_encoder = sklearn.preprocessing.OneHotEncoder()
        integer_encoded_test = y_test.reshape(len(y_test), 1)
        onehot_encoded_test = onehot_encoder.fit_transform(integer_encoded_test)

        #Run the model
        clf1 = model1.fit(X, y)
        y_pred = clf1.predict(X_test)
        y_pred_proba = clf1.predict_proba(X_test)
        integer_encoded_pred = y_pred.reshape(len(y_pred), 1)
        onehot_encoded_pred = onehot_encoder.fit_transform(integer_encoded_pred)

        #Save the outcomes
        accuracy.append(accuracy_score(y_test, y_pred))
        MSE.append(mean_squared_error(y_test, y_pred))
        MAE.append(mean_absolute_error(y_test, y_pred))
        AUROC.append(roc_auc_score(y_test, y_pred_proba[:, 1]))
        Recall.append(recall_score(y_test, y_pred))
        Precision.append(precision_score(y_test, y_pred))
        F1.append(f1_score(y_test, y_pred))
        AUPRC.append(average_precision_score(y_test, y_pred_proba[:,1]))

        #Calculate precision at recall 0.9
        prec, recall, thresholds_prc = precision_recall_curve(y_test, y_pred_proba[:,1], pos_label=clf1.classes_[1])
        thresholds_prc = np.append([0], thresholds_prc) #Thresholds is n-1 length of precision and recall
        threshold_90 = thresholds_prc[recall > 0.9][-1]

        #Now  calculate F1 at that level - should get precision at level of recall
        y_pred_tuned = (y_pred_proba[:,1] > threshold_90).astype(int)
        recall_90_temp = recall_score(y_test, y_pred_tuned)
        recall_90.append(recall_90_temp)
        i = 1
        while recall_90_temp < 0.9:
            #Make sure that recall definitely above 0.9
            y_pred_tuned = (y_pred_proba[:,1] > threshold_90).astype(int)
            recall_90_temp = recall_score(y_test, y_pred_tuned)
            i += 1

        f1_at_90.append(f1_score(y_test, y_pred_tuned))
        precision_at_90.append(precision_score(y_test, y_pred_tuned))

    results_over_time = pd.DataFrame({'total_samples': total_samples,
                                    'unique_patients': unique_patients,
                                    'accuracy': accuracy,
                                    'AUROC': AUROC,
                                    'Precision': Precision, 
                                    'Recall': Recall,
                                    'F1': F1,
                                    'AUPRC': AUPRC,
                                    'recall_90': recall_90,
                                    'f1_at_90': f1_at_90,
                                    'precision_at_90': precision_at_90})

    results_over_time.to_csv(res_path_name + 'XGBoost_results_time_' + file_suffix + '.csv')

#Rerun using an effective k-fold x-val and save the results + samples
if not no_z and perc and input_length == 6 and length == 12:
    k_fold = True
else:
    k_fold = False

folds = 10
if k_fold:
    xgb_prec = list()
    xgb_recall = list()
    lr_prec = list()
    lr_recall = list()
    
    model = XGBClassifier(objective='binary:logistic', eval_metric = 'aucpr', use_label_encoder=False, n_jobs = 32)
    param_grid = {"learning_rate"    : [0.001, 0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
                    'max_depth' : [5, 10, 15, 20, 25, 30, 35, 40],
                        "min_child_weight" : [ 1, 5, 7, 15],
                        "gamma"            : [ 0.0, 0.2, 0.4 ],
                        "colsample_bytree" : [ 0.3, 0.5 , 0.7, 1], 
                        "subsample":[0.5, 0.75, 0.9, 1], 
                        "scale_pos_weight" : [0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1, 2, 4], 
                        "n_estimators" : [50, 100, 150, 200, 250]}

    model_path_name = 'models/'
    file_name = 'XGBoost_best_'
    file_suffix = f'{input_length}h_{length}h_{"no_zscore_" if no_z else ""}pSOFA'
    
    #Work through the different folds
    for fold in tqdm(range(folds)):
        
        #Move through the different folds so the middle split is always the test one, but shifts along
        split_fold = np.array([fold*10, 10, 90 - fold*10])
        
        #Split up testing and outcomes
        split_characteristics = test_trainsplit(characteristics, split_fold)
        split_array2d = test_trainsplit(array2d, split_fold)
        split_array3d = test_trainsplit(array3d, split_fold)
        split_slopes = test_trainsplit(slopes, split_fold)
        split_R = test_trainsplit(r_values, split_fold)
        split_outcomes = test_trainsplit(outcomes, split_fold)

        #Training sets
        train_characteristics = np.concatenate((split_characteristics[0], split_characteristics[2]))
        train_array3d = np.concatenate((split_array3d[0], split_array3d[2]))
        train_array2d = np.concatenate((split_array2d[0], split_array2d[2]))
        train_slopes = np.concatenate((split_slopes[0], split_slopes[2]))
        train_R = np.concatenate((split_R[0], split_R[2]))
        train_outcomes = np.concatenate((split_outcomes[0], split_outcomes[2]))

        #Test sets
        test_characteristics = split_characteristics[1]
        test_array3d = split_array3d[1]
        test_array2d = split_array2d[1]
        test_slopes = split_slopes[1]
        test_R = split_R[1]
        test_outcomes = split_outcomes[1]

        #Make binary outcomes
        #Make the binary values
        binary_deterioration_train_hour_outcomes = np.array(train_outcomes[:, 12] < length, dtype = int)
        binary_deterioration_train_outcomes = np.transpose(np.array([1- binary_deterioration_train_hour_outcomes, binary_deterioration_train_hour_outcomes]))
        binary_deterioration_test_hour_outcomes = np.array(test_outcomes[:, 12] < length, dtype = int)
        binary_deterioration_test_outcomes = np.transpose(np.array([1- binary_deterioration_test_hour_outcomes, binary_deterioration_test_hour_outcomes]))

        #Set x and y
        X = np.concatenate((train_array2d, train_characteristics, train_slopes, train_R), axis=1)
        y = np.argmax(binary_deterioration_train_outcomes, axis = 1)
        
        param_grid = {"learning_rate"    : [0.001, 0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
                    'max_depth' : [5, 10, 15, 20, 25, 30, 35, 40],
                        "min_child_weight" : [ 1, 5, 7, 15],
                        "gamma"            : [ 0.0, 0.2, 0.4 ],
                        "colsample_bytree" : [ 0.3, 0.5 , 0.7, 1], 
                        "subsample":[0.5, 0.75, 0.9, 1], 
                        "scale_pos_weight" : [0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1, 2, 4], 
                        "n_estimators" : [50, 100, 150, 200, 250]}

        if not os.path.exists(model_path_name + file_name + file_suffix + f'_fold_{fold}' + '.json'):                         
            clf = BayesSearchCV(model, param_grid, random_state=0, cv = kfold, iid = False)
            search = clf.fit(X, y)
            
            #Save the best hyperparameters
            best_hyperparameters = param_grid = search.best_params_

            a_file = open(model_path_name + file_name + file_suffix + f'_fold_{fold}' + '.json', 'w')
            json.dump(best_hyperparameters, a_file)
            a_file.close()

        else:
            f = open(model_path_name + file_name + file_suffix + f'_fold_{fold}' + '.json')
            param_grid = json.load(f)
        
        #Now run and get parameters
        model1 = XGBClassifier(objective='binary:logistic', eval_metric = 'aucpr', use_label_encoder=False, n_jobs = 32, **param_grid)
        X_test = np.concatenate((test_array2d, test_characteristics, test_slopes, test_R), axis=1)
        y_test = np.argmax(binary_deterioration_test_outcomes, axis = 1)    
        
        #Run the model
        clf1 = model1.fit(X, y)
        y_pred = clf1.predict(X_test)
        y_pred_proba = clf1.predict_proba(X_test)
        
        #Make the precision recall curve for xgb
        prec, recall, thresholds_prc = precision_recall_curve(y_test, y_pred_proba[:,1], pos_label=clf1.classes_[1])
        xgb_prec.append(prec)
        xgb_recall.append(recall)
    
        #Make the precision recall curve for logistic regression
        clf_pSOFA = LogisticRegression(random_state=0, max_iter= 100, multi_class='multinomial', solver='lbfgs', penalty = 'l2')
        clf_pSOFA.fit(np.max(train_array3d[:, np.array(series_cols) == 'pSOFA Score', :], axis = 2), np.argmax(binary_deterioration_train_outcomes, axis = 1))
        y_predict_pSOFA = clf_pSOFA.predict_proba(np.max(test_array3d[:, np.array(series_cols) == 'pSOFA Score', :], axis = 2))
        prec_pSOFA, recall_pSOFA, thresholds_prc = precision_recall_curve(binary_deterioration_test_outcomes[:,1], y_predict_pSOFA[:,1], pos_label=clf1.classes_[1])
        
        lr_prec.append(prec_pSOFA)
        lr_recall.append(recall_pSOFA)

    #Now plot these
    #Make some color ranges for the lines
    colors_xgb = [cm.Purples(x) for x in range(0, cm.Purples.N, round(cm.Purples.N/folds))]
    colors_lr = [cm.Oranges(x) for x in range(0, cm.Oranges.N, round(cm.Oranges.N/folds))]

    #Work through the prcs and plot them
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    for i, (color_xgb, color_lr) in enumerate(zip(colors_xgb, colors_lr)):
        ax[0].plot(xgb_recall[i], xgb_prec[i], color = color_xgb)
        ax[1].plot(lr_recall[i], lr_prec[i], color = color_lr)
        
    #Some settings
    ax[1].set_xlabel('Recall')
    ax[0].set_ylabel('Precision')
    ax[1].set_ylabel('Precision')
    ax[0].set_title('XGBoost')
    ax[1].set_title('pSOFA')
    fig.savefig(figs_path_name + 'Paired_PRC_cross_val_' + file_suffix + '.pdf', bbox_inches = 'tight', format="pdf")

plotting_individual_deterioration = False
if plotting_individual_deterioration:

    #Make some figures tracking predictions against input values
    file_name = '/Users/danstein/Documents/Masters/Course materials/Project/PICU project/new_project_files/np_arrays_no_zscore_pSOFA_no_perc_6h.npz'

    #Read in data    
    data2 = np.load(file_name)
    array3d2 = data2['d3']
    array2d2 = data2['d2']
    characteristics2 = data2['chars']
    slopes2 = data2['slopes']
    r_values2 = data2['r_values']

    #Split up testing and outcomes
    split_array2d2 = test_trainsplit(array2d2, np.array([85, 15]))
    split_array3d2 = test_trainsplit(array3d2, np.array([85, 15]))
    split_characteristics2 = test_trainsplit(characteristics2, np.array([85, 15]))
    split_slopes2 = test_trainsplit(slopes2, np.array([85, 15]))
    split_R2 = test_trainsplit(r_values2, np.array([85, 15]))

    #Test sets
    test_array3d2 = split_array3d2[1]
    test_array2d2 = split_array2d2[1]
    test_characteristics2 = split_characteristics2[1]
    test_slopes2 = split_slopes2[1]
    test_R2 = split_R2[1]

    X_test2 = np.concatenate((test_array2d2, test_characteristics2, test_slopes2, test_R2), axis=1)

    #Other series names
    series_cols2 = data2['series_cols']
        
    #Some waterfall plots
    #True deterioration
    true_deteriorations = np.intersect1d(np.where(y_pred == 1)[0], 
                np.where(y_test == 1)[0])

    #Find who deteriorated
    deteriorated_pr_id = np.unique(test_outcomes[true_deteriorations, 0], return_counts = True)
    sorted_deteriorated_pr_id = deteriorated_pr_id[0][np.argsort(-deteriorated_pr_id[1])]

    #Stitch together the 3d arrays for the patients who deteriorated
    pt = 5
    deteriorated_locs1 = np.where(test_outcomes[:, 0] == sorted_deteriorated_pr_id[pt])
    ar3d_deteriorated1 = test_array3d2[deteriorated_locs1, :][0]

    #We need the old dimensions to get the new ones
    (dim_z, dim_y, dim_x) = ar3d_deteriorated1.shape
    stacked_ar3d_deteriorated1 = ar3d_deteriorated1.transpose(1, 0, 2).reshape(1, dim_y, dim_z*dim_x)[0]
    deteriorated_predictions1 = y_pred_proba[deteriorated_locs1, 1][0]

    #Now plot them
    interesting_columns = ['Ventilation', 'FiO2', 'MeanAirwayPressure', 
        'MAP', 'HR', 'Comfort.Alertness', 'Comfort', 'GCS', 'SpO2',
        'Inotropes_mcg_kg_min', 'SF_ratio', 'pSOFA_cardio', 'RR',
        'pSOFA', 'Temp', 'Urine_output']

    interesting_column_names = ['Ventilation', 'FiO2', 'Mean Airway Pressure', 
        'MAP', 'HR', 'Comfort Score (Alertness)', 'Comfort Score', 'GCS', 'SpO2',
        'VIS', 'S:F ratio', 'pSOFA (Cardiac)', 'RR',
        'pSOFA', 'Temp', 'Urine output (ml)']

    #Interesting columns for pt 5
    interesting_columns = ['MAP', 'Comfort', 'SF_ratio']
    interesting_column_names = ['MAP (mmHg)', 'Comfort Score', 'S:F Ratio']

    #square_root_length = int(np.ceil(np.sqrt(len(interesting_columns)))) #So we can make the right sized plot

    #fig, axs = plt.subplots(square_root_length, square_root_length, tight_layout = True)
    fig, axs = plt.subplots(3, 1, tight_layout = True)
    for axis_loc, colname in enumerate(interesting_columns): 
        
        #Instantiate the axis and twin axis so that we can plot predictions and variable on same x axis
        #ax = axs[axis_loc % square_root_length, int(np.floor(axis_loc/square_root_length))]
        ax = axs[axis_loc]
        ax_twin = ax.twinx()
        
        if pt == 4:
            xmin = 4000/60
            xmax = 7500/60
        elif pt == 5:
            xmin = 0
            xmax = 48
        
        #Add grey box
        ax.add_patch(plt.Rectangle((24, 0), 12, 1200, facecolor="silver"))
        
        #Plot the first value
        ax.plot([i/60 for i in range(xmax*60 -1)], 
                stacked_ar3d_deteriorated1[series_cols2 == colname, :xmax*60 - 1 ][0], 
                'b-', label='Original')
        
        #Plot the prediction
        ax_twin.plot([(i+1)*6 for i in range(int(xmax/6) - 1)], 
                    deteriorated_predictions1[:int(xmax/6) - 1], 
                    'r-', label='Twin')
        
        ax.axvline(x=24, color='r', linestyle='dashed')
        
        #Set the xlim so that are start to finish and match all the way along - will probably want to narrow this down to look at a place where there was a big change
        if pt == 4:
            xmin = 4000/60
            xmax = 7500/60
            ax.set_xlim(xmin, xmax)
            ax_twin.set_xlim(xmin, xmax)
        elif pt == 5:
            xmin = 0
            xmax = 47.5
            ax.set_xlim(xmin, xmax)
            ax_twin.set_xlim(xmin, xmax)
        
        if colname == 'SpO2':
            ax.set_ylim(80, 100)
        
        if colname == 'MAP':
            ax.set_ylim(40, 110)
        
        if colname == 'Comfort':
            ax.set_ylim(8, 30)
            
        if colname == 'SF_ratio':
            ax.set_ylim(100, 480)
        
        if axis_loc == 2:
            ax.set_xlabel('Hour of admission')
        
        #Set x and y label. Will probably want to add back title
        ax.set_ylabel(interesting_column_names[axis_loc])
        ax_twin.set_ylabel('Predicted probability \n of deterioration')

    plt.show()

    true_deteriorations = np.intersect1d(np.where(y_pred == 1)[0], 
                np.where(y_test == 1)[0])


    #Open the shap values so we use the ones we previously made
    with open('/store/DAMTP/dfs28/PICU_data/shap_pickle/' + file_suffix + '.pkl', 'rb') as file:
        shap_values = pickle.load(file)

    with open('/store/DAMTP/dfs28/PICU_data/shap_pickle/Explainer_' + file_suffix + '.pkl', 'rb') as file:
        explainer = pickle.load(file)
        
    with open('/store/DAMTP/dfs28/PICU_data/shap_pickle/Shap_explainer_' + file_suffix + '.pkl', 'rb') as file:
        shap_explainer = pickle.load(file)

    feature_names = shap_explainer.feature_names
    better_names_dict = {'STD SysBP_missing' : 'Variability: Missingness of Systolic BP', 
                        'STD pSOFA_cardio': 'Variability: pSOFA (Cardiac)',
                        'Mean pSOFA_cardio': 'Mean: pSOFA (Cardiac)',
                        'Mean Comfort.Alertness': 'Mean: Comfort (Alertness)', 
                        'STD DiaBP_missing': 'Variability: Missingness of Diastolic BP', 
                        'STD DBP_zscore': 'Variability: Age normalised Diastolic BP', 
                        'interpolated_ht_m': 'Height (m)', 
                        'Mean MAP_zscore': 'Mean: Age normalised MAP', 
                        'Mean CRT_missing': 'Mean: Missingness of Capillary Refill Time'}

    #Change some of the feature values for the plot to give better context:
    real_values_dict = {'STD SysBP_missing' : 'STD SysBP_missing', 
                        'STD pSOFA_cardio': 'STD pSOFA_cardio',
                        'Mean pSOFA_cardio': 'Mean pSOFA_cardio',
                        'Mean Comfort.Alertness': 'Mean Comfort.Alertness', 
                        'STD DiaBP_missing': 'STD DiaBP_missing', 
                        'STD DBP_zscore': 'STD DiaBP', 
                        'interpolated_ht_m': 'interpolated_ht_m', 
                        'Mean MAP_zscore': 'Mean MAP', 
                        'Mean CRT_missing': 'Mean CRT_missing'}

    #get colnames
    point_cols2 = data2['point_cols']
    point_cols2 = [i for i in point_cols2]
    series_cols = data2['series_cols']

    #Combine the series cols objects and name them for labelling of axes
    series_cols_mean2 = ['Mean ' + i for i in series_cols2]
    series_cols_std2 = ['STD ' + i for i in series_cols2]
    series_cols_slopes2 = ['Slopes ' + i for i in series_cols2]
    series_cols_R2 = ['R ' + i for i in series_cols2]
    all_cols2 = series_cols_mean2
    all_cols2.extend(series_cols_std2)
    all_cols2.extend(series_cols_slopes2)
    all_cols2.extend(series_cols_R2)

    feature_names2 = point_cols2 + all_cols2

    for key in real_values_dict:
        value_loc = [i for i, j in enumerate(shap_explainer.feature_names) if j == key]
        new_value_loc = [i for i, j in enumerate(feature_names2) if j == real_values_dict[key]]
        shap_explainer.data[deteriorated_locs1[0][3], value_loc] = X_test2[deteriorated_locs1[0][3], new_value_loc]
        
    for key in better_names_dict:
        name_loc = [i for i, j in enumerate(shap_explainer.feature_names) if j == key]
        shap_explainer.feature_names[name_loc[0]] = better_names_dict[key]

    fig, ax = plt.subplots(1,1, tight_layout = True)
    shap.plots.waterfall(shap_explainer[deteriorated_locs1[0][3]], show = False)
    plt.show()
