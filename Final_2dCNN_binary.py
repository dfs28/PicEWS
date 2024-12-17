#!/usr/bin/env python3
### Script for final testing of 2d convnet

#Setup
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers
import numpy as np
import keras_tuner as kt
import json
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, mean_absolute_error, auc, confusion_matrix, roc_curve, precision_score, recall_score, average_precision_score, f1_score, PrecisionRecallDisplay, precision_recall_curve, roc_auc_score
import re
import pandas as pd
import argparse 
import random

#Make it so you can run this from the command line
parser = argparse.ArgumentParser(description="Allow running of XGBoost with different input parameters",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-l", "--length", default=6, type=int, help="Length of predictive target in hours")
parser.add_argument("-z", "--no_z", default=False, type=bool, help="Whether or not to use z-scores")
parser.add_argument("-i", "--input_length", default=3, type=int, help="Length of the input window in hours")
parser.add_argument('-r', '--repeats', default = 5, type = int, help = 'Number of times to repeat networks')
args = vars(parser.parse_args())

#Set up whether you want to use the no_z values - default value to be false - but will not override if given above
no_z = args['no_z']
input_length = args['input_length']
length = args['length']
repeats = args['repeats']
    
print(f'2D: Running with {length}h time target window, {input_length}h input window, using z-scores {no_z == False}')

#Pull out the data depending on whether no_z was used
if not no_z:
    #Read in the data
    file_start = '/store/DAMTP/dfs28/PICU_data/np_arrays_pSOFA'
else:
    file_start = '/store/DAMTP/dfs28/PICU_data/np_arrays_no_zscore_pSOFA'
    
if input_length == 3:
    file_name = file_start + '.npz' 
else:
    file_name = file_start + f'_{input_length}h.npz'

#Read in the data
data = np.load(file_name)
array3d = data['d3']
array2d = data['d2']
outcomes = data['outcomes']

#Read in the demographics file
demographics = pd.read_csv('/mhome/damtp/q/dfs28/Project/Project_data/files/caboodle_patient_demographics.csv', sep = ',', parse_dates = ['birth_date', 'death_date'])

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


### Split up test and train
#Note that this has to be a 4d image with a single colour channel
array3d = np.transpose(array3d, (0, 2, 1))
array_image = np.reshape(array3d, (array3d.shape[0], array3d.shape[1], array3d.shape[2], 1))
input_time_image = keras.Input(shape = array_image.shape[1:])
input_flat = keras.Input(shape = array2d.shape[1:])
split_image = test_trainsplit(array_image, np.array([70, 15, 15]))
split_array2d = test_trainsplit(array2d, np.array([70, 15, 15]))
split_outcomes = test_trainsplit(outcomes, np.array([70, 15, 15]))
split_array3d2 = test_trainsplit(array_image, np.array([85, 15]))
split_array2d2 = test_trainsplit(array2d, np.array([85, 15]))
split_outcomes2 = test_trainsplit(outcomes, np.array([85, 15]))

train_image = split_image[0]
train_array2d = split_array2d[0]
train_outcomes = split_outcomes[0]
test_image = split_image[1]
test_array2d = split_array2d[1]
test_outcomes = split_outcomes[1]
all_test_array3d = validate_image = split_image[2]
all_test_array2d = validate_array2d = split_array2d[2]
all_test_outcomes = validate_outcomes = split_outcomes[2]

all_train_array3d = split_array3d2[0]
all_train_array2d = split_array2d2[0]
all_train_outcomes = split_outcomes2[0]

#Make the binary values
binary_deterioration_train_hour_outcomes = np.array(all_train_outcomes[:, 12] < length, dtype = int)
binary_deterioration_train_outcomes = np.transpose(np.array([1- binary_deterioration_train_hour_outcomes, binary_deterioration_train_hour_outcomes]))
binary_deterioration_test_hour_outcomes = np.array(all_test_outcomes[:, 12] < length, dtype = int)
binary_deterioration_test_outcomes = np.transpose(np.array([1- binary_deterioration_test_hour_outcomes, binary_deterioration_test_hour_outcomes]))

#binary_deterioration_train_outcomes = np.transpose(np.array([np.sum(all_train_outcomes[:, 8:9], axis = 1), np.sum(all_train_outcomes[:,9:11], axis = 1)]))
#binary_deterioration_test_outcomes = np.transpose(np.array([np.sum(all_test_outcomes[:, 8:9], axis = 1), np.sum(all_test_outcomes[:,9:11], axis = 1)]))

binary_death_train_outcomes = np.transpose(np.array([np.sum(all_train_outcomes[:, 2:3], axis = 1), np.sum(all_train_outcomes[:,3:5], axis = 1)]))
binary_death_test_outcomes = np.transpose(np.array([np.sum(all_test_outcomes[:, 2:3], axis = 1), np.sum(all_test_outcomes[:,3:5], axis = 1)]))

binary_LOS_train_outcomes = np.transpose(np.array([np.sum(all_train_outcomes[:, 5:6], axis = 1), np.sum(all_train_outcomes[:,6:8], axis = 1)]))
binary_LOS_test_outcomes = np.transpose(np.array([np.sum(all_test_outcomes[:, 5:6], axis = 1), np.sum(all_test_outcomes[:,5:6], axis = 1)]))

### Now make the tunable model

def make_2DNET(model_type):
    #Set regularisers
    init = initializer = tf.keras.initializers.GlorotUniform()

    # Convolutional input
    x = layers.Conv2D(40, (3, 3), activation='relu', padding='same', kernel_initializer = init)(input_time_image)

    x = layers.MaxPooling2D((2,2 ), padding='same')(x)
        
    #Choose how many convolutional layers to have
    filter_num2 = 48
    filter_size2 = 4
    pool_size2 = 3

    for i in range(1):
        x = layers.Conv2D(filter_num2, (filter_size2, filter_size2), 
                            activation='relu', padding='same', 
                            kernel_initializer = init)(x)
        x = layers.MaxPooling2D((pool_size2, pool_size2), padding='same')(x)


    ##Now make the other head with input
    y = layers.Dense(20, activation = 'relu', kernel_initializer = init)(input_flat)

    #Now make the other head
    flattened = layers.Flatten()(x)
    concatted = layers.Concatenate()([y, flattened])
        
    concatted = layers.Dropout(0.5)(concatted)

    dense1 = layers.Dense(45, activation = 'relu', use_bias = True, kernel_initializer = init)(concatted)
    dense2 = layers.Dense(35, activation = 'relu', use_bias = True, kernel_initializer = init)(dense1)

    #Make this a multihead output
    death_head = layers.Dense(2, activation = 'softmax', use_bias = True, kernel_initializer = init)(dense2)
    time_head = layers.Dense(2, activation = 'softmax', use_bias = True, kernel_initializer = init)(dense2)
    PEWS_head = layers.Dense(2, activation = 'softmax', use_bias = True, kernel_initializer = init)(dense2)

    #This is the full model with death and LOS as the outcome
    full_model_2d = keras.Model([input_time_image, input_flat], [death_head, time_head, PEWS_head])

    #This is the full model with death and LOS as the outcome
    full_model = keras.Model([input_time_image, input_flat], [death_head, time_head, PEWS_head])
    death_model = keras.Model([input_time_image, input_flat], death_head)
    discharge_model = keras.Model([input_time_image, input_flat], [time_head])
    PEWS_model = keras.Model([input_time_image, input_flat], [PEWS_head])
    
    #Allow this to return one of 3 different model structures
    if model_type == 'full':
        return full_model
    elif model_type == 'death':
        return death_model
    elif model_type == 'discharge':
        return discharge_model
    elif model_type == 'PEWS':
        return PEWS_model




#Set up some storage for the different metrics
AUC_death_full = list()
AUC_LOS_full = list()
AUC_PEWS_full = list()
acc_death_full = list()
acc_LOS_full = list()
acc_PEWS_full= list()
MSE_death_full = list()
MSE_LOS_full = list()
MSE_PEWS_full = list()
MAE_death_full = list()
MAE_LOS_full = list()
MAE_PEWS_full = list()
recall_death_full = list()
recall_LOS_full = list()
recall_PEWS_full = list()
precision_death_full = list()
precision_LOS_full = list()
precision_PEWS_full = list()
F1_death_full = list()
F1_LOS_full = list()
F1_PEWS_full = list()
AUPRC_death_full = list()
AUPRC_LOS_full = list()
AUPRC_PEWS_full = list()
prec_at_recall_death_full = list()
prec_at_recall_LOS_full = list()
prec_at_recall_PEWS_full = list()

def pull_keys(input_dict, head: int, metric: str):
    """Function to pull relevant metric from keras history"""
     
    #Find validation metrics:
    validation_keys = [i for i in input_dict if re.search('val_.*', i)]
    
    #Rename AUPRC auc_1
    AUPRC_synonyms = set(['AUPRC', 'auprc', 'PRC', 'prc', 'average_precision', 'AUCPR', 'aucpr'])
    if AUPRC_synonyms.intersection([metric]) != set():
        metric = 'auc_1'
        
    #Set the initial metric name
    if head == False:
        metric_name = 'val'
    else:
        head -= 1
        #Find head numbers
        head_numbers = [re.sub('(val_dense_)(\d+)_(\w+)', '\\2', i) for i in validation_keys]
        head_numbers = np.sort(np.unique(head_numbers))
        metric_name = f'val_dense_{head_numbers[head]}'
        
    if metric.startswith('auc'):
        
        #Find AUC numbers
        auc_metrics = [i for i in validation_keys if re.search('.*_auc_.*', i)]
        auc_numbers = np.array([re.sub('.*_(\d+)', '\\1', i) for i in auc_metrics], dtype = int)
        auc_numbers = np.sort(np.unique(auc_numbers))
        
        #If there is only auc and auc_1
        if len(auc_numbers) == 1:
        
            #Now find and return the relevant name of the metric
            metric_name = metric_name + f'_{metric}.*'
        
        else:
            if metric == 'auc':
                metric_name = metric_name + f'_auc_{auc_numbers[0]}.*'
            elif metric == 'auc_1':
                metric_name = metric_name + f'_auc_{auc_numbers[1]}.*'
    else:
        #For all non AUC metrics:
        metric_name = metric_name + f'_{metric}.*'
        
    return [i for i in input_dict if re.search(metric_name, i)][0]
        

#Run this 10 times
for i in range(repeats):
    full_model = make_2DNET('full')

    full_model.compile(optimizer = 'adam', loss='binary_crossentropy',  metrics=['accuracy', 
                            'mse', tf.keras.metrics.MeanAbsoluteError(), 
                            tf.keras.metrics.AUC(), tf.keras.metrics.AUC(curve='PR'), 
                            #tf.keras.metrics.F1(), 
                            tf.keras.metrics.Precision(),
                            tf.keras.metrics.Recall(),
                            tf.keras.metrics.PrecisionAtRecall(0.9)])               

    #Dont forget to add batch size back in 160
    #Now fit the model
    full_model_history = full_model.fit([all_train_array3d, all_train_array2d], [binary_death_train_outcomes, binary_LOS_train_outcomes, binary_deterioration_train_outcomes],
                                        epochs = 20,
                                        batch_size = 160,
                                        shuffle = True, 
                                        validation_data = ([all_test_array3d, all_test_array2d], [binary_death_test_outcomes, binary_LOS_test_outcomes, binary_deterioration_test_outcomes]),
                                        callbacks = [tf.keras.callbacks.EarlyStopping(patience=1)])
    y_pred1, y_pred2, y_pred3 = full_model.predict([all_test_array3d, all_test_array2d])
    recall_death_full.append(recall_score(np.argmax(binary_death_test_outcomes, axis = 1), np.argmax(y_pred1, axis = 1), average = 'micro'))
    recall_LOS_full.append(recall_score(np.argmax(binary_LOS_test_outcomes, axis = 1), np.argmax(y_pred2, axis = 1), average = 'micro'))
    recall_PEWS_full.append(recall_score(np.argmax(binary_deterioration_test_outcomes, axis = 1), np.argmax(y_pred3, axis = 1), average = 'micro'))
    precision_death_full.append(precision_score(np.argmax(binary_death_test_outcomes, axis = 1), np.argmax(y_pred1, axis = 1), average = 'micro'))
    precision_LOS_full.append(precision_score(np.argmax(binary_LOS_test_outcomes, axis = 1), np.argmax(y_pred2, axis = 1), average = 'micro'))
    precision_PEWS_full.append(precision_score(np.argmax(binary_deterioration_test_outcomes, axis = 1), np.argmax(y_pred3, axis = 1), average = 'micro'))
    F1_death_full.append(f1_score(np.argmax(binary_death_test_outcomes, axis = 1), np.argmax(y_pred1, axis = 1), average = 'micro'))
    F1_LOS_full.append(f1_score(np.argmax(binary_LOS_test_outcomes, axis = 1), np.argmax(y_pred2, axis = 1), average = 'micro'))
    F1_PEWS_full.append(f1_score(np.argmax(binary_deterioration_test_outcomes, axis = 1), np.argmax(y_pred3, axis = 1), average = 'micro'))
    
    #Calculate precision at recall 0.9
    prec, recall, thresholds_prc = precision_recall_curve(binary_deterioration_test_outcomes[:,1], y_pred3[:,1])
    thresholds_prc = np.append([0], thresholds_prc) #Thresholds is n-1 length of precision and recall
    threshold_90 = thresholds_prc[recall > 0.9][-1]

    #Now  calculate F1 at that level - should get precision at level of recall
    y_pred_tuned = (y_pred3[:,1] > threshold_90).astype(int)
    
    #Find unique head numbers:
    keys = [i for i in full_model_history.history.keys()]
    
    AUC_death_full.append(full_model_history.history[pull_keys(keys, 1, 'auc')][-1])
    AUC_LOS_full.append(full_model_history.history[pull_keys(keys, 2, 'auc')][-1])
    AUC_PEWS_full.append(roc_auc_score(binary_deterioration_test_outcomes[:,1], y_pred3[:, 1]))
    AUPRC_death_full.append(full_model_history.history[pull_keys(keys, 1, 'auprc')][-1])
    AUPRC_LOS_full.append(full_model_history.history[pull_keys(keys, 2, 'auprc')][-1])
    AUPRC_PEWS_full.append(average_precision_score(binary_deterioration_test_outcomes[:,1], y_pred3[:, 1]))
    acc_death_full.append(full_model_history.history[pull_keys(keys, 1, 'accuracy')][-1]) 
    acc_LOS_full.append(full_model_history.history[pull_keys(keys, 2, 'accuracy')][-1]) 
    acc_PEWS_full.append(accuracy_score(binary_deterioration_test_outcomes[:,1], np.round(y_pred3[:, 1]))) 
    MSE_death_full.append(full_model_history.history[pull_keys(keys, 1, 'mse')][-1]) 
    MSE_LOS_full.append(full_model_history.history[pull_keys(keys, 2, 'mse')][-1]) 
    MSE_PEWS_full.append(full_model_history.history[pull_keys(keys, 3, 'mse')][-1]) 
    MAE_death_full.append(full_model_history.history[pull_keys(keys, 1, 'mean_absolute_error')][-1]) 
    MAE_LOS_full.append(full_model_history.history[pull_keys(keys, 2, 'mean_absolute_error')][-1]) 
    MAE_PEWS_full.append(full_model_history.history[pull_keys(keys, 3, 'mean_absolute_error')][-1]) 
    prec_at_recall_death_full.append(full_model_history.history[pull_keys(keys, 1, 'precision_at_recall')][-1]) 
    prec_at_recall_LOS_full.append(full_model_history.history[pull_keys(keys, 2, 'precision_at_recall')][-1]) 
    prec_at_recall_PEWS_full.append(precision_score(binary_deterioration_test_outcomes[:,1], y_pred_tuned)) 




#Storage for individual models
AUC_death_individual = list()
AUC_LOS_individual = list()
AUC_PEWS_individual = list()
AUPRC_death_individual = list()
AUPRC_LOS_individual= list()
AUPRC_PEWS_individual = list()
acc_death_individual = list()
acc_LOS_individual = list()
acc_PEWS_individual= list()
MSE_death_individual = list()
MSE_LOS_individual = list()
MSE_PEWS_individual = list()
MAE_death_individual = list()
MAE_LOS_individual = list()
MAE_PEWS_individual = list()
recall_death_individual = list()
recall_LOS_individual = list()
recall_PEWS_individual = list()
precision_death_individual = list()
precision_LOS_individual = list()
precision_PEWS_individual = list()
F1_death_individual = list()
F1_LOS_individual = list()
F1_PEWS_individual = list()
prec_at_recall_death_individual = list()
prec_at_recall_LOS_individual = list()
prec_at_recall_PEWS_individual = list()
"""
#Mortality prediction
for i in range(10):
    mortality_model = make_2DNET('death')

    mortality_model.compile(optimizer = 'adam', loss='binary_crossentropy', 
                       metrics = ['mse', tf.keras.metrics.MeanAbsoluteError(), 
                                  tf.keras.metrics.AUC(), tf.keras.metrics.AUC(curve='PR'), 
                                  #tf.keras.metrics.F1(),
                                  tf.keras.metrics.Precision(),
                                  tf.keras.metrics.Recall(),
                                  tf.keras.metrics.PrecisionAtRecall(0.9)])               

    #Dont forget to add batch size back in 160
    #Now fit the model
    mortality_model_history = mortality_model.fit([all_train_array3d, all_train_array2d], [binary_death_train_outcomes],
                                        epochs = 20,
                                        batch_size = 160,
                                        shuffle = True, 
                                        validation_data = ([all_test_array3d, all_test_array2d], [binary_death_test_outcomes]),
                                        callbacks = [tf.keras.callbacks.EarlyStopping(patience=1)])
    y_pred1 = mortality_model.predict([all_test_array3d, all_test_array2d])  
    acc_death_individual.append(accuracy_score(np.argmax(binary_death_test_outcomes, axis = 1), np.argmax(y_pred1, axis = 1)))
    recall_death_individual.append(recall_score(np.argmax(binary_death_test_outcomes, axis = 1), np.argmax(y_pred1, axis = 1), average = 'micro'))
    precision_death_individual.append(precision_score(np.argmax(binary_death_test_outcomes, axis = 1), np.argmax(y_pred1, axis = 1), average = 'micro'))
    F1_death_individual.append(f1_score(np.argmax(binary_death_test_outcomes, axis = 1), np.argmax(y_pred1, axis = 1), average = 'micro'))

    keys = [i for i in mortality_model_history.history.keys()]
    AUC_death_individual.append(mortality_model_history.history[pull_keys(keys, False, 'auc')][-1])
    AUPRC_death_individual.append(mortality_model_history.history[pull_keys(keys, False, 'auprc')][-1])
    MSE_death_individual.append(mortality_model_history.history[pull_keys(keys, False, 'mse')][-1]) 
    MAE_death_individual.append(mortality_model_history.history[pull_keys(keys, False, 'mean_absolute_error')][-1]) 
    prec_at_recall_death_individual.append(mortality_model_history.history[pull_keys(keys, False, 'precision_at_recall')][-1]) 



#LOS prediction
for i in range(10):
    discharge_model = make_2DNET('discharge')

    discharge_model.compile(optimizer = 'adam', loss='binary_crossentropy', 
                       metrics = ['mse', tf.keras.metrics.MeanAbsoluteError(), 
                                  tf.keras.metrics.AUC(), tf.keras.metrics.AUC(curve='PR'), 
                            #tf.keras.metrics.F1(), 
                            tf.keras.metrics.Precision(),
                            tf.keras.metrics.Recall(),
                            tf.keras.metrics.PrecisionAtRecall(0.9)])               

    #Dont forget to add batch size back in 160
    #Now fit the model
    discharge_model_history = discharge_model.fit([all_train_array3d, all_train_array2d], [binary_LOS_train_outcomes],
                                        epochs = 20,
                                        batch_size = 160,
                                        shuffle = True, 
                                        validation_data = ([all_test_array3d, all_test_array2d], [binary_LOS_test_outcomes]),
                                        callbacks = [tf.keras.callbacks.EarlyStopping(patience=1)])
    y_pred1 = discharge_model.predict([all_test_array3d, all_test_array2d])  
    acc_LOS_individual.append(accuracy_score(np.argmax(binary_LOS_test_outcomes, axis = 1), np.argmax(y_pred1, axis = 1)))
    recall_LOS_individual.append(recall_score(np.argmax(binary_LOS_test_outcomes, axis = 1), np.argmax(y_pred1, axis = 1), average = 'micro'))
    precision_LOS_individual.append(precision_score(np.argmax(binary_LOS_test_outcomes, axis = 1), np.argmax(y_pred1, axis = 1), average = 'micro'))
    F1_LOS_individual.append(f1_score(np.argmax(binary_LOS_test_outcomes, axis = 1), np.argmax(y_pred1, axis = 1), average = 'micro'))

    keys = [i for i in discharge_model_history.history.keys()]
    AUC_LOS_individual.append(discharge_model_history.history[pull_keys(keys, False, 'auc')][-1])
    AUPRC_LOS_individual.append(discharge_model_history.history[pull_keys(keys, False, 'auprc')][-1])
    MSE_LOS_individual.append(discharge_model_history.history[pull_keys(keys, False, 'mse')][-1]) 
    MAE_LOS_individual.append(discharge_model_history.history[pull_keys(keys, False, 'mean_absolute_error')][-1]) 
    prec_at_recall_LOS_individual.append(discharge_model_history.history[pull_keys(keys, False, 'precision_at_recall')][-1]) 
"""

#Deterioration prediction
for i in range(repeats):
    deterioration_model = make_2DNET('PEWS')

    deterioration_model.compile(optimizer = 'adam', loss='binary_crossentropy',
                                metrics = ['mse', tf.keras.metrics.MeanAbsoluteError(), 
                                tf.keras.metrics.AUC(), tf.keras.metrics.AUC(curve='PR'),
                                #tf.keras.metrics.F1(), 
                                tf.keras.metrics.Precision(),
                                tf.keras.metrics.Recall(),
                                tf.keras.metrics.PrecisionAtRecall(0.9)])          

    #Dont forget to add batch size back in 160
    #Now fit the model
    deterioration_model_history = deterioration_model.fit([all_train_array3d, all_train_array2d], [binary_deterioration_train_outcomes],
                                        epochs = 20,
                                        batch_size = 160,
                                        shuffle = True, 
                                        validation_data = ([all_test_array3d, all_test_array2d], [binary_deterioration_test_outcomes]),
                                        callbacks = [tf.keras.callbacks.EarlyStopping(patience=1)])
    y_pred1 = deterioration_model.predict([all_test_array3d, all_test_array2d])  
    acc_PEWS_individual.append(accuracy_score(np.argmax(binary_deterioration_test_outcomes, axis = 1), np.argmax(y_pred1, axis = 1)))
    recall_PEWS_individual.append(recall_score(np.argmax(binary_deterioration_test_outcomes, axis = 1), np.argmax(y_pred1, axis = 1), average = 'micro'))
    precision_PEWS_individual.append(precision_score(np.argmax(binary_deterioration_test_outcomes, axis = 1), np.argmax(y_pred1, axis = 1), average = 'micro'))
    F1_PEWS_individual.append(f1_score(np.argmax(binary_deterioration_test_outcomes, axis = 1), np.argmax(y_pred1, axis = 1), average = 'micro'))

    #Calculate precision at recall 0.9
    prec, recall, thresholds_prc = precision_recall_curve(binary_deterioration_test_outcomes[:,1], y_pred1[:,1])
    thresholds_prc = np.append([0], thresholds_prc) #Thresholds is n-1 length of precision and recall
    threshold_90 = thresholds_prc[recall > 0.9][-1]

    #Now  calculate F1 at that level - should get precision at level of recall
    y_pred_tuned = (y_pred1[:,1] > threshold_90).astype(int)
    
    AUC_PEWS_individual.append(roc_auc_score(binary_deterioration_test_outcomes[:,1], y_pred1[:, 1]))
    AUPRC_PEWS_individual.append(average_precision_score(binary_deterioration_test_outcomes[:,1], y_pred1[:, 1]))
    acc_PEWS_individual.append(accuracy_score(binary_deterioration_test_outcomes[:,1], np.round(y_pred1[:, 1]))) 
    prec_at_recall_PEWS_individual.append(precision_score(binary_deterioration_test_outcomes[:,1], y_pred_tuned)) 

    keys = [i for i in deterioration_model_history.history.keys()]
    MSE_PEWS_individual.append(deterioration_model_history.history[pull_keys(keys, False, 'mse')][-1]) 
    MAE_PEWS_individual.append(deterioration_model_history.history[pull_keys(keys, False, 'mean_absolute_error')][-1]) 

conf_mat3 = confusion_matrix(np.argmax(y_pred1, axis = 1), np.argmax(all_test_outcomes[:, 8:11], axis = 1))

results = {'acc_death_individual_mean' : np.mean(acc_death_individual), 
            'acc_death_individual_std' : np.std(acc_death_individual),
            'acc_death_full_mean' : np.mean(acc_death_full), 
            'acc_death_full_std' : np.std(acc_death_full), 
            'acc_LOS_individual_mean' : np.mean(acc_LOS_individual), 
            'acc_LOS_individual_std' : np.std(acc_LOS_individual),
            'acc_LOS_full_mean' : np.mean(acc_LOS_full), 
            'acc_LOS_full_std' : np.std(acc_LOS_full),
            'acc_PEWS_individual_mean' : np.mean(acc_PEWS_individual), 
            'acc_PEWS_individual_std' : np.std(acc_PEWS_individual),
            'acc_PEWS_full_mean' : np.mean(acc_PEWS_full), 
            'acc_PEWS_full_std' : np.std(acc_PEWS_full),
            'AUC_death_individual_mean' : np.mean(AUC_death_individual), 
            'AUC_death_individual_std' : np.std(AUC_death_individual),
            'AUC_death_full_mean' : np.mean(AUC_death_full), 
            'AUC_death_full_std' : np.std(AUC_death_full), 
            'AUC_LOS_individual_mean' : np.mean(AUC_LOS_individual), 
            'AUC_LOS_individual_std' : np.std(AUC_LOS_individual),
            'AUC_LOS_full_mean' : np.mean(AUC_LOS_full), 
            'AUC_LOS_full_std' : np.std(AUC_LOS_full),
            'AUC_PEWS_individual_mean' : np.mean(AUC_PEWS_individual), 
            'AUC_PEWS_individual_std' : np.std(AUC_PEWS_individual),
            'AUC_PEWS_full_mean' : np.mean(AUC_PEWS_full), 
            'AUC_PEWS_full_std' : np.std(AUC_PEWS_full),
            'AUPRC_death_individual_mean' : np.mean(AUPRC_death_individual), 
            'AUPRC_death_individual_std' : np.std(AUPRC_death_individual),
            'AUPRC_death_full_mean' : np.mean(AUPRC_death_full), 
            'AUPRC_death_full_std' : np.std(AUPRC_death_full), 
            'AUPRC_LOS_individual_mean' : np.mean(AUPRC_LOS_individual), 
            'AUPRC_LOS_individual_std' : np.std(AUPRC_LOS_individual),
            'AUPRC_LOS_full_mean' : np.mean(AUPRC_LOS_full), 
            'AUPRC_LOS_full_std' : np.std(AUPRC_LOS_full),
            'AUPRC_PEWS_individual_mean' : np.mean(AUPRC_PEWS_individual), 
            'AUPRC_PEWS_individual_std' : np.std(AUPRC_PEWS_individual),
            'AUPRC_PEWS_full_mean' : np.mean(AUPRC_PEWS_full), 
            'AUPRC_PEWS_full_std' : np.std(AUPRC_PEWS_full),
            'MSE_death_individual_mean' : np.mean(MSE_death_individual), 
            'MSE_death_individual_std' : np.std(MSE_death_individual),
            'MSE_death_full_mean' : np.mean(MSE_death_full), 
            'MSE_death_full_std' : np.std(MSE_death_full), 
            'MSE_LOS_individual_mean' : np.mean(MSE_LOS_individual), 
            'MSE_LOS_individual_std' : np.std(MSE_LOS_individual),
            'MSE_LOS_full_mean' : np.mean(MSE_LOS_full), 
            'MSE_LOS_full_std' : np.std(MSE_LOS_full),
            'MSE_PEWS_individual_mean' : np.mean(MSE_PEWS_individual), 
            'MSE_PEWS_individual_std' : np.std(MSE_PEWS_individual),
            'MSE_PEWS_full_mean' : np.mean(MSE_PEWS_full), 
            'MSE_PEWS_full_std' : np.std(MSE_PEWS_full),
            'MAE_death_individual_mean' : np.mean(MAE_death_individual), 
            'MAE_death_individual_std' : np.std(MAE_death_individual),
            'MAE_death_full_mean' : np.mean(MAE_death_full), 
            'MAE_death_full_std' : np.std(MAE_death_full), 
            'MAE_LOS_individual_mean' : np.mean(MAE_LOS_individual), 
            'MAE_LOS_individual_std' : np.std(MAE_LOS_individual),
            'MAE_LOS_full_mean' : np.mean(MAE_LOS_full), 
            'MAE_LOS_full_std' : np.std(MAE_LOS_full),
            'MAE_PEWS_individual_mean' : np.mean(MAE_PEWS_individual), 
            'MAE_PEWS_individual_std' : np.std(MAE_PEWS_individual),
            'MAE_PEWS_full_mean' : np.mean(MAE_PEWS_full), 
            'MAE_PEWS_full_std' : np.std(MAE_PEWS_full), 
            'precision_death_individual_mean' : np.mean(precision_death_individual), 
            'precision_death_individual_std' : np.std(precision_death_individual),
            'precision_death_full_mean' : np.mean(precision_death_full), 
            'precision_death_full_std' : np.std(precision_death_full), 
            'precision_LOS_individual_mean' : np.mean(precision_LOS_individual), 
            'precision_LOS_individual_std' : np.std(precision_LOS_individual),
            'precision_LOS_full_mean' : np.mean(precision_LOS_full), 
            'precision_LOS_full_std' : np.std(precision_LOS_full),
            'precision_PEWS_individual_mean' : np.mean(precision_PEWS_individual), 
            'precision_PEWS_individual_std' : np.std(precision_PEWS_individual),
            'precision_PEWS_full_mean' : np.mean(precision_PEWS_full), 
            'precision_PEWS_full_std' : np.std(precision_PEWS_full), 
            'prec_at_recall_death_individual_mean' : np.mean(prec_at_recall_death_individual), 
            'prec_at_recall_death_individual_std' : np.std(prec_at_recall_death_individual),
            'prec_at_recall_death_full_mean' : np.mean(prec_at_recall_death_full), 
            'prec_at_recall_death_full_std' : np.std(prec_at_recall_death_full), 
            'prec_at_recall_LOS_individual_mean' : np.mean(prec_at_recall_LOS_individual), 
            'prec_at_recall_LOS_individual_std' : np.std(prec_at_recall_LOS_individual),
            'prec_at_recall_LOS_full_mean' : np.mean(prec_at_recall_LOS_full), 
            'prec_at_recall_LOS_full_std' : np.std(prec_at_recall_LOS_full),
            'prec_at_recall_PEWS_individual_mean' : np.mean(prec_at_recall_PEWS_individual), 
            'prec_at_recall_PEWS_individual_std' : np.std(prec_at_recall_PEWS_individual),
            'prec_at_recall_PEWS_full_mean' : np.mean(prec_at_recall_PEWS_full), 
            'prec_at_recall_PEWS_full_std' : np.std(prec_at_recall_PEWS_full), 
            'recall_death_individual_mean' : np.mean(recall_death_individual), 
            'recall_death_individual_std' : np.std(recall_death_individual),
            'recall_death_full_mean' : np.mean(recall_death_full), 
            'recall_death_full_std' : np.std(recall_death_full), 
            'recall_LOS_individual_mean' : np.mean(recall_LOS_individual), 
            'recall_LOS_individual_std' : np.std(recall_LOS_individual),
            'recall_LOS_full_mean' : np.mean(recall_LOS_full), 
            'recall_LOS_full_std' : np.std(recall_LOS_full),
            'recall_PEWS_individual_mean' : np.mean(recall_PEWS_individual), 
            'recall_PEWS_individual_std' : np.std(recall_PEWS_individual),
            'recall_PEWS_full_mean' : np.mean(recall_PEWS_full), 
            'recall_PEWS_full_std' : np.std(recall_PEWS_full), 
            'F1_death_individual_mean' : np.mean(F1_death_individual), 
            'F1_death_individual_std' : np.std(F1_death_individual),
            'F1_death_full_mean' : np.mean(F1_death_full), 
            'F1_death_full_std' : np.std(F1_death_full), 
            'F1_LOS_individual_mean' : np.mean(F1_LOS_individual), 
            'F1_LOS_individual_std' : np.std(F1_LOS_individual),
            'F1_LOS_full_mean' : np.mean(F1_LOS_full), 
            'F1_LOS_full_std' : np.std(F1_LOS_full),
            'F1_PEWS_individual_mean' : np.mean(F1_PEWS_individual), 
            'F1_PEWS_individual_std' : np.std(F1_PEWS_individual),
            'F1_PEWS_full_mean' : np.mean(F1_PEWS_full), 
            'F1_PEWS_full_std' : np.std(F1_PEWS_full)}

a_file = open("/mhome/damtp/q/dfs28/Project/PICU_project/files/2DCNN_results_binary_pSOFA", "w")
json.dump(results, a_file)
a_file.close()
results_df = pd.DataFrame({'metrics': results.keys(), 'results': results.values()})

file_suffix = f'{input_length}h_{length}h_{"no_zscore_" if no_z else ""}pSOFA'
results_df.to_csv("/mhome/damtp/q/dfs28/Project/PICU_project/files/2DCNN_results_binary_" + file_suffix + '.csv')
#tf.keras.utils.plot_model(full_model, to_file='/mhome/damtp/q/dfs28/Project/PICU_project/models/2dCNN_binary.pdf', show_shapes=True, expand_nested = True)
tf.keras.utils.plot_model(deterioration_model, to_file='/mhome/damtp/q/dfs28/Project/PICU_project/models/2dCNN_binary_single_output.pdf', show_shapes=True, expand_nested = True)
