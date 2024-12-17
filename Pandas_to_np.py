#!/usr/bin/env python3

#### Little script to convert big flowsheet file into np arrays for use
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import math
from datetime import datetime
from scipy import stats
import scipy
from progress.bar import Bar
import sys
from functools import reduce
import argparse
from tqdm import tqdm
from datetime import timedelta

#Make it so you can run this from the command line
parser = argparse.ArgumentParser(description="Allow running of Pandas_to_np with different input parameters",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-l", "--length", default=360, type=int, help="Length of input window in minutes")
parser.add_argument("-z", "--no_z", default=False, type=bool, help="Whether or not to use z-scores")
parser.add_argument("-p", "--perc", default=True, type=bool, help="Whether or not to use percentile")
args = vars(parser.parse_args())

no_z = args['no_z']
input_length = args['length']
perc = args['perc']

print(f'Running with {round(input_length/60)}h time input window, no z-scores: {no_z}, percentile: {perc}')

#To be able to run locally 
columns = ['taken_datetime', 'project_id', 'encounter_key', 'ALT', 'Albumin', 'AlkPhos',
            'AST', 'Aspartate', 'Amylase', 'APTT', 'Anion_gap', 'Base_excess', 'Basophils',
            'Bicarb', 'pH', 'Blood_culture', 'Cr', 'CRP', 'Ca2.', 'Cl', 'Eosinophils',
            'FHHb', 'FMetHb', 'FO2Hb', 'Glucose', 'HCT.', 'HCT', 'INR', 'Lactate', 
            'Lymphs', 'Mg', 'Monocytes', 'Neuts', 'P50', 'PaCO2', 'PcCO2', 'PmCO2', 
            'PaO2', 'PcO2', 'PmO2', 'PO2', 'PvCO2', 'PcO2.1', 'Phos', 'Plts', 'K.', 
            'PT', 'Retics', 'Na.', 'TT', 'Bili', 'WCC', 'count', 'Strong_ion_gap', 
            'Age_yrs', 'time_to_death', 'died', 'sex', 'ethnicity', 'Weight.g.', 
            'interpolated_wt_kg', 'Ventilation', 'HFO', 'Tracheostomy', 'ETCO2', 
            'FiO2', 'O2Flow', 'O2Flow.kg', 'IPAP', 'Ventilation_L_min', 'Ventilation.ml.', 
            'MeanAirwayPressure', 'EPAP', 'Ventilation_missing', 'O2Flow.kg_missing', 
            'IPAP_missing', 'EPAP_missing', 'FiO2_missing', 'HFO_missing', 'Tracheostomy_missing', 
            'Ventilation.ml._missing', 'MeanAirwayPressure_missing', 'ETCO2_missing', 
            'SysBP', 'DiaBP', 'MAP', 'SysBP_missing', 'DiaBP_missing', 'MAP_missing', 
            'HR', 'HR_missing', 'Comfort.Alertness', 'Comfort.BP', 'Comfort.Calmness', 
            'Comfort', 'Comfort.HR', 'Comfort.Resp', 'Comfort.Alertness_missing', 
            'Comfort.BP_missing', 'Comfort.Calmness_missing', 'Comfort.HR_missing', 
            'Comfort.Resp_missing', 'Comfort_missing', 'AVPU', 'GCS_V', 'GCS_E', 
            'GCS_M', 'GCS', 'GCS_missing', 'GCS_E_missing', 'GCS_M_missing', 
            'GCS_V_missing', 'AVPU_missing', 'CRT', 'CRT_missing', 'SpO2', 'ScvO2', 
            'SpO2_missing', 'ScvO2_missing', 'Height.cm.', 'interpolated_ht_m', 'input', 
            'output', 'Nappy_output', 'ECMO', 'ECMO_missing', 'Adj_adrenaline', 'Inotropes', 
            'Norad_adrenaline', 'EquivAVP', 'EquivDopamine', 'EquivMilrinone', 'Inotropes_kg', 
            'AVP_kg', 'Dopamine_kg', 'Norad_adrenaline_kg', 'Milrinone_kg', 'Inotropes_mcg_kg_min', 
            'EquivAVP_mcg_kg_min', 'EquivDopamine_mcg_kg_min', 'Norad_adrenaline_mcg_kg_min', 
            'EquivMilrinone_mcg_kg_min', 'Inotropes_missing', 'AVP_missing', 'Dopamine_missing', 
            'Norad_adrenaline_missing', 'Milrinone_missing', 'SF_ratio', 'pSOFA_resp', 'pSOFA_cardio', 
            'pSOFA_plt', 'pSOFA_bili', 'pSOFA_GCS', 'pSOFA_Cr', 'RR', 'RR_missing', 'dialysis', 
            'dialysis_missing', 'pSOFA', 'Temp', 'Temp_missing', 'Urine_output', 'Urine_output_kg', 
            'Urine_output_missing_kg', 'PEWS', 'hospital_no', 'Weight_z_scores', 'Height_z_scores', 
            'HR_zscore', 'RR_zscore', 'DBP_zscore', 'SBP_zscore', 'MAP_zscore']

#In reality import the big flowsheet to use thing
print('Starting run: ', datetime.now().strftime("%H:%M:%S"))
flowsheet = pd.read_csv('/Users/danstein/Documents/Masters/Course materials/Project/PICU project/new_project_files/flowsheet_zscores.csv.gz', parse_dates = ['taken_datetime'], usecols = columns)
print('Flowsheet loaded: ', datetime.now().strftime("%H:%M:%S"))

#Fix issue with ethnicity, sex and died
died = {'N': 0, 'Y': 1}
flowsheet['died'] = flowsheet['died'].replace(died)
sex = {'F': 0, 'M': 1, 'I': 2}
flowsheet['sex'] = flowsheet['sex'].replace(sex)
print('Correction to float applied: ', datetime.now().strftime("%H:%M:%S"))

#Make 3d training array
def make_3d_array(array, length, all_cols, point_cols, series_cols, percentile = True, normalise = True):
    """
    Function to make an pandas into a 3d np.array of slices and stack them \n
    Takes a dataframe, slices it by the length specified 
    Expects the time series to be longer than the number of vars
    Specify the length
    Percentile scales by rank - this should mean that fluctuations within the normal range are still visible
    """

    #Get the shape, work out what shape the new array should be
    all_used_cols = [i for i in array.columns if i in all_cols]
    print(all_used_cols)

    array_use = np.array(array.loc[:, all_used_cols])
    i_point_cols = [i for i, j in enumerate(all_used_cols) if j in point_cols]
    j_point_cols = [j for i, j in enumerate(all_used_cols) if j in point_cols]
    i_series_cols = [i for i, j in enumerate(all_used_cols) if j in series_cols]
    j_series_cols = [j for i, j in enumerate(all_used_cols) if j in series_cols]

    print('Converted to np array: ', datetime.now().strftime("%H:%M:%S"))

    #Monitor progress
    #Make this an np array with no nans
    if normalise:
        for i in tqdm(range(array_use.shape[1]), desc = 'Normalising data', colour = 'green'):
            nas = np.isnan(array_use[:, i])
            if any(nas):
                print(i)
                continue

            #Scale to percentile
            if percentile:
                array_use[nas == False, i] = stats.rankdata(array_use[nas == False, i])

            min = np.min(array_use[nas == False, i])
            max = np.max(array_use[nas == False, i])
            array_use[nas, i] = min

            #Now normalise 0-1
            array_use[:, i] -= min
            if not (max - min) == 0:
                array_use[:, i] /= (max - min)
        
    shape = array_use.shape
    z_dim = math.floor(np.max(shape)/length)
    
    #Get the slices to use
    to_use = np.array([], dtype = int)
    unique_patients = array['project_id'].unique()
    slicesPerPatient = np.zeros(len(unique_patients))
    
    for location, pt in enumerate(tqdm(unique_patients, desc = 'Getting usable slices')):
        
        #Get the positions for each patient
        pt_locs = np.where(array.project_id == pt)[0]
        assert len(np.unique(array.project_id[pt_locs])) <= 1, 'There should only be one patient per slice'
        
        #Get the number of slices per patient
        pt_slices = int(np.floor(len(array.project_id[pt_locs])/length))
        slicesPerPatient[location] = pt_slices
        
        #Slot in the locations of the patients
        to_use = np.concatenate([to_use, pt_locs[[i*180 for i in range(int(pt_slices))]]])

    x_dim = length
    y_dim = np.min(shape)
    array_3d = np.empty((len(to_use), len(i_series_cols), x_dim))
    array_2d = np.empty((len(to_use), len(i_point_cols)))
    array_characteristics = np.empty((len(to_use), len(i_series_cols)*2))
    splines = np.empty((len(to_use), 8*len(i_series_cols)))
    slopes = np.empty((len(to_use), len(i_series_cols)))
    std_errs = np.empty((len(to_use), len(i_series_cols)))
    r_values = np.empty((len(to_use), len(i_series_cols)))
    intercepts = np.empty((len(to_use), len(i_series_cols)))
    p_values = np.empty((len(to_use), len(i_series_cols)))

    #Outcomes
    outcomes = np.zeros((len(to_use), 17))
    pt_slices = list()

    #bar = Bar('Slicing and outcomes', max=len(to_use))
    for position, i in enumerate(tqdm(to_use, desc = 'Getting outcomes', colour = 'green')):
        start_position = i
        end_position = i + length
        end_position_pd = i + length -1
        temp_array = array_use[start_position:end_position, i_series_cols]
        array_3d[position, :, :] = temp_array.transpose()
        array_2d[position, :] = array_use[end_position - 1, i_point_cols]


        ##Build the outcomes
        #Make sure you can see which patients these are coming from
        patient_id = np.where(array.loc[end_position, 'project_id'] == unique_patients)[0]
        project_id = array.loc[end_position, 'project_id']
        outcomes[position, 0] = int(project_id[-4:])

        #Age of the patient
        outcomes[position, 1] = array.loc[end_position, 'Age_yrs']
        
        #Time to death as triplicate value - probably can't assume that death is within this admission?
        if array.loc[end_position, 'time_to_death'] <= 2/365:
            outcomes[position, 2] = 1
        elif array.loc[end_position, 'died'] == 1:
            outcomes[position, 3] = 1
        else: 
            outcomes[position, 4] = 1

        #Length of stay as triplicate
        end_of_section = array.loc[end_position, 'taken_datetime']
        all_dates = array.loc[array['project_id'] == project_id, 'taken_datetime']
        discharge_date = all_dates[all_dates.index[-1]]
        time_to_discharge = discharge_date - end_of_section

        #Correct for death prior to discharge

        #Now assign depending on how soon:
        if (time_to_discharge < np.timedelta64(2, 'D')) and (outcomes[position, 2] != 1):
            outcomes[position, 5] = 1
        elif time_to_discharge < np.timedelta64(7, 'D') and (array.loc[end_position, 'time_to_death'] > 7/365):
            outcomes[position, 6] = 1
        else:
            outcomes[position, 7] = 1

        #Now do something with deterioration as triplicate - get all pSOFA after end of this slice
        pt_locs = np.where(array['project_id'] == project_id)[0]

        #Get all slots from end of slice to end of patient
        next_slices = pt_locs[pt_locs >=  end_position]
        all_pSOFA_cardio = array.loc[next_slices, 'pSOFA_cardio'] #These should probably be named next_pSOFA/next_lactate etc
        all_lactate = array.loc[next_slices, 'Lactate']
        all_ECMO = array.loc[next_slices, 'ECMO']
        all_pSOFA = array.loc[next_slices, 'pSOFA']
        
        #Get maximum pSOFA from current slice
        max_pSOFA_cardio = np.max(array.loc[start_position:end_position_pd, 'pSOFA_cardio'])
        max_pSOFA = np.max(array.loc[start_position:end_position_pd, 'pSOFA'])
        max_lactate = np.max(array.loc[start_position:end_position_pd, 'Lactate'])
        max_ECMO = np.max(array.loc[start_position:end_position_pd, 'ECMO'])
        

        #Get all deterioration - now including ECMO
        worse_pSOFA_cardio = all_pSOFA_cardio > max_pSOFA_cardio
        worse_lactate = np.logical_and(all_lactate > max_lactate, all_lactate > 2)
        worse_ECMO = all_ECMO > max_ECMO
        all_worse = reduce(np.logical_or, (worse_pSOFA_cardio, worse_lactate, worse_ECMO))
        next_dates = all_dates[all_worse.index]
        worse_date = next_dates[all_worse]
        
        #Just do for pSOFA
        worse_pSOFA = all_pSOFA > max_pSOFA
        next_dates_pSOFA = all_dates[worse_pSOFA.index]
        worse_date_pSOFA = next_dates_pSOFA[worse_pSOFA]

        #Now for lactate, ECMO, pSOFA cardio
        next_dates_lactate = all_dates[worse_lactate.index]
        worse_date_lactate = next_dates_lactate[worse_lactate]
        next_dates_ECMO = all_dates[worse_ECMO.index]
        worse_date_ECMO = next_dates_ECMO[worse_ECMO]
        next_dates_pSOFA_cardio = all_dates[worse_pSOFA_cardio.index]
        worse_date_pSOFA_cardio = next_dates_pSOFA_cardio[worse_pSOFA_cardio]
        
        #Just work out time to deterioration
        if len(worse_date) > 0:
            time_to_deteriorate = worse_date[worse_date.index[0]] - end_of_section
            
            #Converting to hours allows easier comparisons later on (and we can use this for the standalone ttd value)
            hours_to_deteriorate = time_to_deteriorate.total_seconds()/3600
        else:
            #100 days in hours
            hours_to_deteriorate = 2400
        
        #Time to worse pSOFA
        if len(worse_date_pSOFA) > 0:
            time_to_worse_pSOFA = worse_date_pSOFA[worse_date_pSOFA.index[0]] - end_of_section
            hours_to_worse_pSOFA = time_to_worse_pSOFA.total_seconds()/3600
        else:
            hours_to_worse_pSOFA = 2400

        #For lactate and ECMO
        if len(worse_date_lactate) > 0:
            time_to_worse_lactate = worse_date_lactate[worse_date_lactate.index[0]] - end_of_section
            hours_to_worse_lactate = time_to_worse_lactate.total_seconds()/3600
        else:
            hours_to_worse_lactate = 2400
        
        if len(worse_date_ECMO) > 0:
            time_to_worse_ECMO = worse_date_ECMO[worse_date_ECMO.index[0]] - end_of_section
            hours_to_worse_ECMO = time_to_worse_ECMO.total_seconds()/3600
        else:
            hours_to_worse_ECMO = 2400

        if len(worse_date_pSOFA_cardio) > 0:
            time_to_worse_pSOFA_cardio = worse_date_pSOFA_cardio[worse_date_pSOFA_cardio.index[0]] - end_of_section
            hours_to_worse_pSOFA_cardio = time_to_worse_pSOFA_cardio.total_seconds()/3600
        else:
            hours_to_worse_pSOFA_cardio = 2400

        
        
        #Work out time to death, if less than to deteriorate mark as deteriorated
        if np.isnan(array.time_to_death[end_position]) == False:
            time_to_death = pd.Timedelta(array.loc[end_position, 'time_to_death']*365, 'D') 
            hours_to_death = time_to_death.total_seconds()/3600
            time_to_deteriorate = np.min([hours_to_death, hours_to_deteriorate])
            time_to_worse_pSOFA = np.min([hours_to_death, hours_to_worse_pSOFA])
        else:
            time_to_deteriorate = hours_to_deteriorate
            time_to_worse_pSOFA = hours_to_worse_pSOFA
            
        #Deterioration in hours
        assert time_to_deteriorate >= 0, 'Only interested in future deterioration'
        outcomes[position, 12] = time_to_deteriorate
        assert time_to_worse_pSOFA >= 0, 'Only interested in future deterioration (pSOFA)'
        outcomes[position, 13] = time_to_worse_pSOFA
        assert hours_to_worse_pSOFA_cardio >= 0, 'Only interested in future deterioration (pSOFA cardio)'
        outcomes[position, 14] = hours_to_worse_pSOFA_cardio
        assert hours_to_worse_lactate >= 0, 'Only interested in future lactate rise'
        outcomes[position, 15] = hours_to_worse_lactate
        assert hours_to_worse_ECMO >= 0, 'Only interested in future ECMO requirement'
        outcomes[position, 16] = hours_to_worse_ECMO
        
        #Currently setting cutoffs to <6h, 6-24h, >24h
        if time_to_deteriorate < 6 or array.loc[end_position, 'time_to_death'] <= 0.25/365:
            outcomes[position, 8] = 1
        elif time_to_deteriorate < 24 or array.loc[end_position, 'time_to_death'] <= 1/365:
            outcomes[position, 9] = 1
        else:
            outcomes[position, 10] = 1

        #Time to death as outcome (if doesn't die then 70yrs to death)
        if np.isnan(array.loc[end_position, 'time_to_death']):
            outcomes[position, 11] = 70
        else:
            outcomes[position, 11] = array.loc[end_position, 'time_to_death']

        ##Now get pointwise variables - could do median and mad depending on how scale to percentile looks?
        means = [np.mean(temp_array[:, i]) for i in range(np.shape(temp_array)[1])]
        st_devs = [np.std(temp_array[:, i]) for i in range(np.shape(temp_array)[1])]
        array_characteristics[position, :] = np.array(means + st_devs)
        
        ##Fit splines to approximate curve
        for j in range(temp_array.shape[1]):

            #Get polynomial values, smoothing seems to make this all the same length (8, will need to make this adaptable if change length)
            polynomials = scipy.interpolate.splrep(x = range(len(temp_array[:, j])), y = temp_array[:, j], s= 10000)[1]
            splines[position, range(j*8, (j+1)*8)] = polynomials[:8]

            #Now fit a least squares line and take the slope
            slope, intercept, r_value, p_value, std_err = stats.linregress(x = range(len(temp_array[:, j])), y = temp_array[:, j])
            slopes[position, j] = slope
            intercepts[position, j] = intercept
            r_values[position, j] = r_value
            p_values[position, j] = p_value
            std_errs[position, j] = std_err

    
    na_loc = np.isnan(array_3d)
    array_3d[na_loc] = 0
    na_loc2 = np.isnan(array_2d)
    array_2d[na_loc2] = 0
    na_loc_char = np.isnan(array_characteristics)
    array_characteristics[na_loc_char] = 0

    return array_3d, array_2d, array_characteristics, splines, outcomes, slopes, intercepts, r_values, p_values, std_errs, slicesPerPatient, all_used_cols, j_point_cols, j_series_cols

#Note that HCT. is HCT#

if args['no_z'] == False:
    point_cols = ['ALT', 'Albumin', 'AlkPhos','AST', 'Aspartate', 'Amylase', 'APTT', 'Anion_gap', 'Base_excess', 'Basophils', 'Bicarb',
                'pH', 'Blood_culture',  'Cr', 'CRP', 'Ca2.', 'Cl', 'Eosinophils', 'FHHb', 'FMetHb', 'FO2Hb', 'Glucose', 'HCT.', 'HCT', 'INR',
                'Lactate', 'Lymphs', 'Mg', 'Monocytes', 'Neuts', 'P50', 'PaCO2', 'PcCO2', 'PmCO2', 'PaO2', 'PcO2', 'PmO2', 'PO2', 
                'PvCO2', 'Phos', 'Plts', 'K.', 'PT', 'Retics', 'Na.', 'TT', 'Bili', 'WCC', 'Strong_ion_gap', 'Age_yrs', 'sex', 
                #'ethnicity', 
                'Weight_z_scores', 'Height_z_scores', 
                'interpolated_wt_kg', 'interpolated_ht_m'] #?Take out ethinicity?
    

    series_cols =  ['Ventilation', 'HFO', 'IPAP', 'EPAP', 'Tracheostomy', 'ETCO2', 'FiO2', 'O2Flow', 'O2Flow.kg', 'Ventilation.ml.',
                    'MeanAirwayPressure', 'Ventilation_missing', 'O2Flow.kg_missing', 'IPAP_missing', 'EPAP_missing', 'FiO2_missing', 'HFO_missing',
                    'Tracheostomy_missing', 'Ventilation.ml._missing', 'MeanAirwayPressure_missing', 'ETCO2_missing', 'SBP_zscore', 'DBP_zscore', 
                    'MAP_zscore', 'SysBP_missing', 'DiaBP_missing', 'MAP_missing', 'HR_zscore', 'HR_missing', 'Comfort.Alertness', 'Comfort.BP', 'Comfort.Calmness',
                    'Comfort', 'Comfort.HR', 'Comfort.Resp', 'AVPU', 'GCS_V', 'GCS_E', 'GCS_M', 'GCS', 'GCS_missing', 'AVPU_missing', 'CRT', 'CRT_missing',
                    'SpO2', 'SpO2_missing', 'interpolated_ht_m', 'ECMO', 'ECMO_missing', 'Inotropes_mcg_kg_min', 'EquivAVP_mcg_kg_min', 'EquivDopamine_mcg_kg_min',
                    'Norad_adrenaline_mcg_kg_min', 'EquivMilrinone_mcg_kg_min', 'Inotropes_missing', 'RR_zscore', 'RR_missing',  'AVP_missing', 'Dopamine_missing',
                    'Norad_adrenaline_missing', 'Milrinone_missing',
                    'SF_ratio', 'pSOFA_resp', 'pSOFA_cardio', 'pSOFA_plt', 'pSOFA_bili', 'pSOFA_GCS', 'pSOFA_Cr', 'pSOFA',
                    'dialysis', 'dialysis_missing', 'Temp', 'Temp_missing', 'Urine_output', 'Urine_output_kg', 'Urine_output_missing_kg', 'PEWS']

    other_cols = ['time_to_death', 'died']
else:
    #Note that HCT. is HCT#
    point_cols = ['ALT', 'Albumin', 'AlkPhos','AST', 'Aspartate', 'Amylase', 'APTT', 'Anion_gap', 'Base_excess', 'Basophils', 'Bicarb',
                'pH', 'Blood_culture', 'Cr', 'CRP', 'Ca2.', 'Cl', 'Eosinophils', 'FHHb', 'FMetHb', 'FO2Hb', 'Glucose', 'HCT.', 'HCT', 'INR',
                'Lactate', 'Lymphs', 'Mg', 'Monocytes', 'Neuts', 'P50', 'PaCO2', 'PcCO2', 'PmCO2', 'PaO2', 'PcO2', 'PmO2', 'PO2', 
                'PvCO2', 'Phos', 'Plts', 'K.', 'PT', 'Retics', 'Na.', 'TT', 'Bili', 'WCC', 'Strong_ion_gap', 'Age_yrs', 'sex', 
                'interpolated_wt_kg', 'interpolated_ht_m']


    series_cols =  ['Ventilation', 'HFO', 'IPAP', 'EPAP', 'Tracheostomy', 'ETCO2', 'FiO2', 'O2Flow', 'Ventilation.ml.',
                    'MeanAirwayPressure', 'Ventilation_missing', 'O2Flow.kg_missing', 'IPAP_missing', 'EPAP_missing', 'FiO2_missing', 'HFO_missing',
                    'Tracheostomy_missing', 'Ventilation.ml._missing', 'MeanAirwayPressure_missing', 'ETCO2_missing', 'SysBP', 'DiaBP', 
                    'MAP', 'SysBP_missing', 'DiaBP_missing', 'MAP_missing', 'HR', 'HR_missing', 'Comfort.Alertness', 'Comfort.BP', 'Comfort.Calmness',
                    'Comfort', 'Comfort.HR', 'Comfort.Resp', 'AVPU', 'GCS_V', 'GCS_E', 'GCS_M', 'GCS', 'GCS_missing', 'AVPU_missing', 'CRT', 'CRT_missing',
                    'SpO2', 'SpO2_missing', 'ECMO', 'ECMO_missing', 'Inotropes_mcg_kg_min', 'EquivAVP_mcg_kg_min', 'EquivDopamine_mcg_kg_min',
                    'Norad_adrenaline_mcg_kg_min', 'EquivMilrinone_mcg_kg_min', 'Inotropes_missing', 'RR', 'RR_missing', 'AVP_missing', 'Dopamine_missing',
                    'Norad_adrenaline_missing', 'Milrinone_missing',
                    'SF_ratio', 'pSOFA_resp', 'pSOFA_cardio', 'pSOFA_plt', 'pSOFA_bili', 'pSOFA_GCS', 'pSOFA_Cr', 'pSOFA',
                    'dialysis', 'dialysis_missing', 'Temp', 'Temp_missing', 'Urine_output', 'Urine_output_missing_kg', 'PEWS']

    other_cols = ['time_to_death', 'died']

all_cols = point_cols + series_cols + other_cols

#Now run it
file_start = '/Users/danstein/Documents/Masters/Course materials/Project/PICU project/new_project_files/all_outcomes_np_arrays'

if no_z == False:
    file_name = file_start + '_pSOFA'
else:
    file_name = file_start + '_no_zscore_pSOFA'
    
if not perc:
    file_name = file_name + '_no_perc'

if input_length == 180:
    file_name = file_name + '.npz'
else:
    file_name = file_name + f'_{round(input_length/60)}h.npz'

array3d, array2d, array_characteristics, splines, outcomes, slopes, intercepts, r_values, p_values, std_errs, slicesPerPatient, all_used_cols, j_point_cols, j_series_cols = make_3d_array(flowsheet, input_length, all_cols, point_cols, series_cols, normalise = perc)
print(f'Slicing done with length {input_length}: ', datetime.now().strftime("%H:%M:%S"))
np.savez(file_name, d3 = array3d, d2 = array2d, chars = array_characteristics, splines = splines, slopes = slopes, intercepts =  intercepts, r_values = r_values, p_values = p_values, std_errs = std_errs,
         outcomes = outcomes, per_pt = slicesPerPatient, all_used = np.array(all_used_cols), point_cols = np.array(j_point_cols), series_cols = np.array(j_series_cols))
print(f'Saved with length {input_length}: ', datetime.now().strftime("%H:%M:%S"))