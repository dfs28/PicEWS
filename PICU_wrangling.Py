### PICU data wrangling
# Setup

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from progress.bar import Bar
import os
from itertools import chain
import re
from datetime import datetime
from tqdm import tqdm, tqdm_pandas

#Starting message
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Starting run at", current_time)

#Import the files - start with the massive flowsheet one
flowsheet = pd.read_csv('/mhome/damtp/q/dfs28/Project/Project_data/files/caboodle_patient_selected_flowsheetrows_main_pivot.csv', sep= ',', parse_dates=['taken_datetime'])
demographics = pd.read_csv('/mhome/damtp/q/dfs28/Project/Project_data/files/caboodle_patient_demographics.csv', sep = ',', parse_dates = ['birth_date', 'death_date'])
medications = pd.read_csv('/mhome/damtp/q/dfs28/Project/Project_data/files/caboodle_patient_selected_medication_admins_main.csv', sep = ',', parse_dates = ['start_datetime', 'end_datetime'])
labs = pd.read_csv('/store/DAMTP/dfs28/updated_files/caboodle_patient_selected_lab_components_main_pivot.csv', sep = ',', parse_dates = ['collected_datetime', 'received_datetime', 'verified_datetime'])
stays = pd.read_csv('/store/DAMTP/dfs28/updated_files/caboodle_patient_ward_stays.csv', sep = ',', parse_dates = ['start_datetime', 'end_datetime'])
episodes = pd.read_csv('/store/DAMTP/dfs28/updated_files/caboodle_patient_episodes.csv', sep = ',', parse_dates = ['start_datetime', 'end_datetime'])

##### Make a whole load of helper functions to help with cleaning


## Some functions to help work out what variables are present/ needed

#Helper function for returning the column number as a name
def return_colname(original_column, sheet, give_inputs = False):
    """
    Function to give original column as string
    """
    
    #Get the unique values flexibly
    if type(original_column) == int:
        unique_inputs = sheet.loc[:, sheet.columns[original_column]].unique()
        
        #Make sure original column is a string
        original_column = sheet.columns[original_column]

    elif type(original_column) == str:
        unique_inputs = sheet.loc[:, (original_column)].unique()
    else:
        raise ValueError('original_column should either be a numbered column or the correct name of a column')

    if not give_inputs:
        return original_column
    else:
        return original_column, unique_inputs



#Print out all of the columns
def print_columns(sheet):
    for i in enumerate(sheet.columns):
        yield print(i)

def print_all(sheet):
    a = print_columns(sheet)
    length = len(sheet.columns)
    for i in range(length):
        next(a)



#Print out all of the unique parts of the columns
def print_uniquevars(sheet, column):
    
    column, unique_inputs = return_colname(column, sheet, True)

    for i in enumerate(unique_inputs):
        yield print(i)

def print_allunique(sheet, column, printAll = True):
    """
    Simple function that prints proportion of NaNs, absolute number of Nans and unique values
    Could also add in functionality to say what proportion of each exist
    """

    #Return colname
    column = return_colname(column, sheet)

    #Get number of Nas
    isnan = sheet.loc[:, column].isna() == False
    notNa = sum(isnan)
    propnotNa = notNa/sheet.shape[0]
    print('For {}'.format(column))
    print('{:2.2%} percent not Na'.format(propnotNa))
    print('{} of {} not Na'.format(notNa, sheet.shape[0]))

    #Print out the unique vals
    if printAll:
        a = print_uniquevars(sheet, column)
        length = len(sheet.loc[:, column].unique())
        for i in range(length):
            next(a)



## Bit of code to work out how many Nas there are per column

#Think about which variables are seen the most and the least
#Should help with what is worth going through
#Not run
#This takes ages so I've saved it, only rerun if needed
if (os.path.exists('/mhome/damtp/q/dfs28/Project/Project_data/files/not_NA.csv') == False):
    colnames = flowsheet.columns
    numNotNa = np.zeros([len(colnames) - 3])

    #Make a progress bar because this takes ages
    bar = Bar('Processing', max=len(colnames))

    #Work  through columns and get numbers of Nans
    for i in range(3, len(colnames)):
        isnan = flowsheet.iloc[:, (i)].isna() == False
        numNotNa[i - 3] = sum(isnan)
        bar.next()

    bar.finish()
    topcols = colnames[np.append(np.array([False, False, False]), numNotNa > 10000)]
    for i in range(len(topcols)):
        print(topcols[i])
    sortedNaNs = np.argsort(numNotNa)
    np.savetxt('/mhome/damtp/q/dfs28/Project/Project_data/files/not_NA.csv', sortedNaNs, delimiter=',')
else: 
    sortedNaNs = np.genfromtxt('/mhome/damtp/q/dfs28/Project/Project_data/files/not_NA.csv', delimiter=',')



## Work out which columns overlap
def overlapping_columns(sheet):
    """ 
    Function that returns columns which have other subsets
    """
    columns = sheet.columns
    subsets = list()
    for i in columns:
        subset = list()
        for j in (k for k in columns if k != i):

            #Work out if col j is a subset of col i
            j_locs = (sheet[j].isna() == False)
            all_same = all(sheet.loc[j_locs, j] == sheet.loc[j_locs, i])
            if all_same:
                subset.append(j)
        subsets.append(subset)

    return subsets



## Helper function to redefine and merge columns

#Some helper functions for redefining columns
def make_new_columns(original_column, sheet, new_cols, inputIsoutput = False, **input2output):
    """
    Function to take some columns and then make new ones  \n
    Original column can be a number or a string \n
    Sheet is a pd.DataFrame \n
    New cols (need to be named strings) are the new columns to be made \n
    If inputIsoutput = True then it just takes the input and puts it in the new column \n
    Input2output passed as nested lists for linking  \n
    Will probably want to add some sort of missingness new column \n
    """

    #Get the unique values flexibly
    original_column = return_colname(original_column, sheet)

    #Allocate the new values
    if not inputIsoutput:
        i = 0
        
        #Work through the lists
        for key, value in input2output.items():
            new_column = new_cols[i]
            if not new_column in sheet.columns:

                #Should probably consider whether default should be zeros?
                #Could maybe consider this to be na most of the time, then subsequently interpolate
                #Fill with zeros if backfill limit reached on interpolation
                #Could do interpolation but also consider missingness?
                sheet[new_column] = np.nan
            
            #Work  through the items in this list
            for j, k in enumerate(value):

                #Only take values that are in this column
                if isinstance(k, list):
                    for l, m in enumerate(k):
                        original_values = value[j]
                        sheet.loc[sheet[original_column].isin(original_values), (new_column)] = j
                else:         
                    original_value = value[j]
                    sheet.loc[sheet[original_column] == original_value, (new_column)] = j

            i += 1
    
    else:
        if not new_cols[0] in sheet.columns:
            sheet[new_cols[0]] = np.nan

        #Find the values that are Na
        na_locs = (sheet.loc[:, original_column].isna() == False)
        new_values = sheet.loc[na_locs, original_column]

        #Now assign
        sheet.loc[na_locs, new_cols[0]] = new_values


# This is a helper function for the above helper function
# It returns all non-Na unique values so they can be converted into a categorical variable

def make_allnonNa(sheet, column, which_item, length):
    """
    Simple function to make all the non-na values into a list
    """

    column, unique_values = return_colname(column, sheet, True)
    if unique_values.dtype == np.dtype('O'):
        unique_values = pd.Series(unique_values, dtype = 'string')
        unique_nonan = unique_values[unique_values.isna() == False].tolist()
    else:
        unique_nonan = unique_values[np.isnan(unique_values) == False].tolist()
    l = [None] * length
    l[which_item -1 ] = unique_nonan
    return l


# Function to correct FiO2 and make it into new col

def cor_FiO2(original_column, sheet, new_col):
    """
    Function to take columns containing FiO2 and make a new one \n
    It corrects FiO2 into a number between 0 and 1 \n
    Original column can be a number or a string \n
    Sheet is a pd.DataFrame \n
    New col (a named string)
    """

    #Get the unique values flexibly
    original_column = return_colname(original_column, sheet)

    if not new_col in sheet.columns:
        sheet[new_col] = np.nan

    #Find the values that are not Na
    na_locs = (sheet.loc[:, original_column].isna() == False)
    new_values = sheet.loc[na_locs, original_column]

    #Work through the values to make them into a number between 0 and 1
    toomuch_FiO2 = 100 < new_values
    new_values[toomuch_FiO2] = 100
    
    #Over 21
    percent_FiO2 = 21 <= new_values
    new_values[percent_FiO2] /= 100

    dec_FiO2 = 1 < new_values
    new_values[dec_FiO2] /= 10
    
    #Any that are less than 0.21 correct to 0.21
    lessthan21 = 0.21 > new_values
    new_values[lessthan21] = 0.21
    
    #Any remaining rogue ones: 
    rogue_FiO2 = 1 < new_values
    new_values[rogue_FiO2] = 0.21

    #Now assign
    sheet.loc[na_locs, new_col] = new_values


#Helper function to search through the outcomes to help me
def get_relevant_cols(sheet, query):
    """
    Function to pull out relevant colnames
    Will have it spit out how many Nas in the matching ones, 
    Also a short list of the unique values
    """

    #Pull the colnames and then work through them with a query
    colnames = sheet.columns
    relevant_columns = list()
    
    #Make the query
    prog = re.compile(query, flags = re.IGNORECASE)
    for i, j in enumerate(colnames):

        #Search for the query in colnames, if present store it
        result = re.search(prog, j)
        try:
            start = result.start()
            relevant_columns.append(i)
        except:
            pass
    
    for i in relevant_columns:
        print(i)
        print_allunique(sheet, i, False)
        unique_vars = sheet.iloc[:, i].unique()
        length = len(unique_vars)
        if length > 10:
            length = 10
        print(unique_vars[0:length])
        print('\n')
        

#Can consider some sort of helper function to return appropriate vars?

def conv_BP(original_column, sheet, SysDia, *new_cols):
    """
    Function to take some columns and then make new ones  \n
    Original column can be a number or a string \n
    Sheet is a pd.DataFrame \n
    New cols (need to be named strings) are the new columns to be made \n
    SysDia needs to be true or false
    If inputIsoutput = True then it just takes the input and puts it in the new column \n
    Input2output passed as nested lists for linking  \n
    Will probably want to add some sort of missingness new column \n
    """

    #Get the unique values flexibly
    original_column = return_colname(original_column, sheet)

    #If both need to split into sys and dia
    if SysDia in ['Three', 'Both', 'both', 'three', 3]:
        new_sheet = sheet.loc[:, (original_column)].str.split("/", n = 1, expand = True)
        new_sheet = new_sheet.astype('float')
        new_sheet['MAP'] = new_sheet.loc[:, 0]*(1/3) + new_sheet.loc[:, 1]*(2/3)
    else:
        new_sheet = sheet.loc[:, (original_column)]

    #Allocate the new values
    i = 0
        
    #Work through the new columns
    for value in new_cols:
        new_column = value
        
        #Make a new column if its not alread there
        if not new_column in sheet.columns:
            sheet[new_column] = np.nan

        #Use only the ones from the original columns that aren't Na
        if SysDia in ['Three', 'Both', 'both', 'three', 3]:
            na_locs = (new_sheet.iloc[:, (i)].isna() == False)
            new_values = new_sheet.loc[na_locs, (new_sheet.columns[i])]
        else:
            na_locs = (new_sheet.isna() == False)
            new_values = new_sheet[na_locs]

        #Now assign
        sheet.loc[na_locs, (new_column)] = new_values

        i += 1


def conv_column(original_column, sheet, new_col, func):
    """
    Function to take columns and convert using a new function \n
    Original column can be a number or a string \n
    Sheet is a pd.DataFrame \n
    New col (a named string)
    """

    #Get the unique values flexibly
    original_column = return_colname(original_column, sheet)

    if not new_col in sheet.columns:
        sheet[new_col] = np.nan

    #Find the values that are Na
    na_locs = (sheet.loc[:, original_column].isna() == False)
    new_values = sheet.loc[na_locs, original_column]

    #Apply the function
    new_values = func(new_values)

    #Now assign
    sheet.loc[na_locs, new_col] = new_values

def sum_inout(original_column, sheet, new_col):
    """
    Function to take a an existing column and add to it \n
    Original column can be a number or a string \n
    Sheet is a pd.DataFrame \n
    New col (a named string)
    """

    #Get the unique values flexibly
    original_column = return_colname(original_column, sheet)

    if not new_col in sheet.columns:
        sheet[new_col] = np.nan

    #Find the values that are Na
    na_locs = (sheet.loc[:, original_column].isna() == False)
    new_values = sheet.loc[na_locs, original_column]

    #If new column has nas where going to add make these zero
    both_na = sheet[new_col].isna() & na_locs
    sheet.loc[both_na, new_col] = 0

    #Now assign
    sheet.loc[na_locs, new_col] = sheet.loc[na_locs, new_col] + new_values

def bminusa(a, b):
    return b - a

def conv_2cols(original_column1, original_column2, sheet, new_col, func):
    """
    Function to take two existing columns and use function to apply them together to get a new col\n
    Original column can be a number or a string \n
    Sheet is a pd.DataFrame \n
    New col (a named string)
    """

    #Get the unique values flexibly
    original_column1 = return_colname(original_column1, sheet)
    original_column2 = return_colname(original_column2, sheet)

    if not new_col in sheet.columns:
        sheet[new_col] = np.nan

    #Find the values that are Na 1
    na_locs1 = (sheet.loc[:, original_column1].isna() == False)
    new_values1 = sheet[original_column1]

    #Find the values that are Na 2
    na_locs2 = (sheet.loc[:, original_column2].isna() == False)
    new_values2 = sheet[original_column2]

    #Find overlapping not nas
    both_nonan = na_locs1 & na_locs2

    #Calculate new values
    new_values = func(new_values1, new_values2)

    #Now assign
    sheet.loc[both_nonan, new_col] = new_values


### Get weights and heights for correction
def apply_wt_ht(original_column, sheet, new_column, scaling = 1):
    """
    Function to get wt or ht and apply them so there is an up to date weight or height \n
    Original column can be a number or a string \n
    Sheet is a pd.DataFrame \n
    Scaling is a scaling factor so can convert to kg if necessary \n
    new_col is a string
    """

    #Get the unique values flexibly
    original_column = return_colname(original_column, sheet)

    #Split by patient and work through them
    unique_patients = sheet['project_id'].unique()
    for i in tqdm(unique_patients):

        patient_locs = (sheet['project_id'] == i)
        
        #Interpolate weight linearly and fill in all missing values
        new_values = sheet.loc[patient_locs, original_column].interpolate('linear', limit = 100000, limit_direction = 'both')

        #Fill in new column with these weights, scaling
        sheet.loc[patient_locs, new_column] = new_values*scaling


def apply_demographic(original_column, demographic_sheet, flow_sheet, new_column, age = True):
    """
    Function to apply demographics to the new sheet \n
    Original column can be a number or a string \n
    Sheet is a pd.DataFrame \n
    new_col is a string 
    """

    #Make the column
    flow_sheet[new_column] = np.nan

    original_column = return_colname(original_column, demographic_sheet)

    unique_patients = flow_sheet['project_id'].unique()
    for i in unique_patients:

        #Get demographic value
        demographic_loc = (demographic_sheet['project_id'] == i)
        value = demographic_sheet.loc[demographic_loc, original_column]
        
        #Get location on flowsheet
        flowsheet_locs = (flow_sheet['project_id'] == i)

        #Make sequence of values
        length = sum(flowsheet_locs)
        new_values = np.repeat(value, length)

        if age:
            #Get date of variable in current time
            dates = flow_sheet.loc[flowsheet_locs, 'taken_datetime']
            dates = dates.dt.round('D')

            #Make sequence of birth date
            birth_dates = new_values
            dates.index = birth_dates.index

            #Calculate age and convert to years
            ages = dates - birth_dates
            new_values = ages / np.timedelta64(1, 'Y')
            new_values = abs(new_values)    

        #Fill in new column with these weights, scaling
        new_values.index = flow_sheet.loc[flowsheet_locs, new_column].index
        flow_sheet.loc[flowsheet_locs, new_column] = new_values

def get_unique_meds(medication_sheet, drug_name):
    """
    Function to query drug names
    """

    #Make the columns
    unique_medications = np.sort(medication_sheet['drug_name'].unique())
    drug_match = re.compile(drug_name)
    drug_names = [i for i in unique_medications if drug_match.match(i) != None]

    return drug_names

#all_doses = np.array([re.sub('(\d+\.*\d*)\s(.+)', r'\2', j) for i, j in enumerate(medications.loc[drug_locs, 'prn_dose']) if isinstance(j, str)])

def apply_drugs(medication_sheet, flow_sheet, drug_name, new_col, conc, dose_conv = 1, units = False):
    """
    Function to apply medications in new columns to sheet
    To use one formulation at a time
    Actually just leave this out - was using for phenylephrine but seems that it is given PRN mostly so not helpful
    """

    #Get drug name
    drug_name = get_unique_meds(medication_sheet, drug_name)
    drug_name = drug_name[0]
    
    #Get locations
    drug_locs = np.where(medication_sheet['drug_name'] == drug_name)

    #Get durations - these turn out all to be 0
    durations = medication_sheet.loc[drug_locs, 'end_datetime'] - medication_sheet.loc[drug_locs, 'start_datetime']
    durations = durations / np.timedelta64(1, 'h')
    durations = abs(durations)   

    #Now correct for time

    weight_doses = ['mg', 'mg/kg/min', 'micrograms', 'microg/kg/min', 'mL']
    unit_doses = ['Million Units', 'Units', 'Units/hr', 'Units/kg/hr', 'microg/kg/min', 'mL']
    mmol_doses = ['']

    for i in drug_locs:
        print(i)

    
    unique_patients = flow_sheet['project_id'].unique()

    for i in unique_patients:

        #Get demographic value
        demographic_loc = (demographic_sheet['project_id'] == i)
        value = demographic_sheet.loc[demographic_loc, original_column]
        
        #Get location on flowsheet
        flowsheet_locs = (flow_sheet['project_id'] == i)

        #Make sequence of values
        length = sum(flowsheet_locs)
        new_values = np.repeat(value, length)

        if age:
            #Get date of variable in current time
            dates = flow_sheet.loc[flowsheet_locs, 'taken_datetime']
            dates = dates.dt.round('D')

            #Make sequence of birth date
            birth_dates = new_values
            dates.index = birth_dates.index

            #Calculate age and convert to years
            ages = dates - birth_dates
            new_values = ages / np.timedelta64(1, 'Y')
            new_values = abs(new_values)    

        #Fill in new column with these weights, scaling
        new_values.index = flow_sheet.loc[flowsheet_locs, new_column].index
        flow_sheet.loc[flowsheet_locs, new_column] = new_values        


def fill_missing(original_column, sheet, new_col, missing = np.nan):
    """
    Function to take a an existing column and fill in missing values \n
    And make a new column  
    Original column can be a number or a string \n
    Sheet is a pd.DataFrame \n
    New col (a named string)
    """

    #Get the unique values flexibly
    original_column = return_colname(original_column, sheet)

    if not new_col in sheet.columns:
        sheet[new_col] = 0

    #Find the values that are Na, put 1s here
    na_locs = sheet.loc[:, original_column].isna()
    sheet.loc[na_locs, new_col] = 1
    sheet.loc[na_locs, original_column] = missing

def interpolate_cols(column, sheet, method, *args, how_far = 90, limit_direction = 'both', fill = 0):
    """ 
    Function to interpolate some columns \n
    Method, how_far, limit_direction inherit from pd.interpolate \n
    Fill then fills in the rest of the columns with a filler \n
    Do you want to make a new column to interpolate into? - xgboost can handle NaNs
    """

    #Get the unique values flexibly
    column = return_colname(column, sheet)

    #Work through the patients
    unique_patients = sheet['project_id'].unique()
    
    for i in tqdm(unique_patients):
        pt_locs = sheet['project_id'] == i
        
        #Now interpolate
        new_values = sheet.loc[pt_locs, column].interpolate(method = method,limit = how_far, limit_direction = limit_direction)
        sheet.loc[pt_locs, column] = new_values

    #Allow fill to 
    na_locs = sheet[column].isna() == True
    if callable(fill):
        fill(column, sheet, *args)
    else:
        sheet.loc[na_locs, column] = fill


#Median fill for BP and HR
HR_meansd = pd.read_csv('/mhome/damtp/q/dfs28/Project/PICU_project/files/HR_meansd.csv')
RR_meansd = pd.read_csv('/mhome/damtp/q/dfs28/Project/PICU_project/files/RR_meansd.csv')
MAP_male_meansd = pd.read_csv('/mhome/damtp/q/dfs28/Project/PICU_project/files/MAP_meansd_male.csv')
MAP_female_meansd = pd.read_csv('/mhome/damtp/q/dfs28/Project/PICU_project/files/MAP_meansd_female.csv')
DBP_male_meansd = pd.read_csv('/mhome/damtp/q/dfs28/Project/PICU_project/files/DBP_meansd_male.csv')
DBP_female_meansd = pd.read_csv('/mhome/damtp/q/dfs28/Project/PICU_project/files/DBP_meansd_female.csv')
SBP_male_meansd = pd.read_csv('/mhome/damtp/q/dfs28/Project/PICU_project/files/SBP_meansd_male.csv')
SBP_female_meansd = pd.read_csv('/mhome/damtp/q/dfs28/Project/PICU_project/files/SBP_meansd_female.csv')

def fill_median_HRBP(column, sheet, typeHRBP):
    """Function to fill in median values where BP missing despite interpolation
    Takes a column to fill, sheet, and type should be one of 'SBP', 'DBP' or 'MAP', 'BP', 'HR' or 'RR'
    """

    column = return_colname(column, sheet)
    na_locs = sheet[column].isna() == True

    #Set up the median values
    if typeHRBP in ['BP', 'SBP', 'DBP', 'MAP']:
        age = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 25]
    elif typeHRBP in ['HR', 'RR']:
        age = HR_meansd['age']
    else:
        raise ValueError('typeHRBP should be one of "SBP", "DBP" or "MAP", "BP", "HR" or "RR"')

    #Get ages
    age_range = np.full(sheet.shape[0], -1)
    median_values = list()
    for i, j in enumerate(age):
        locations = (sheet['Age_yrs'] < j) & (age_range == -1)
        age_range[locations] = i  
        
        #Now get median values
        notNas = (na_locs == False) & locations
        median = np.median(sheet.loc[notNas, column])
        median_values.append(median)

    #Now fill in the values
    median_BPs = np.array([median_values[j] for i, j in np.ndenumerate(age_range)])
    sheet.loc[na_locs, column] = median_BPs[na_locs]
    
def fill_median(column, sheet):
    """Function to fill in median values where normal missing
    Takes a column to fill, sheet
    """

    column = return_colname(column, sheet)
    na_locs = sheet[column].isna()

    #Get median
    median = np.median(sheet.loc[na_locs == False, column])
    sheet.loc[na_locs, column] = median

def remove_punct(column, new_column, sheet, punct_query):
    """
    Function to remove punctuation
    """
    column = return_colname(column, sheet)
    if not new_column in sheet.columns:
        sheet[new_column] = np.nan
    
    float_values = np.array([re.sub(punct_query, r'\2', str(row[column])) for index, row in sheet.iterrows()], dtype='f')
    na_locs = np.isnan(float_values) == False
    sheet.loc[na_locs, new_column] = float_values[na_locs]

### Add in lab components


if (os.path.exists('/store/DAMTP/dfs28/PICU_data/flowsheet_labs.csv.gz') == False):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Starting labs at", current_time)

    ## Now merge lab component columns into new cols
    #ALT
    remove_punct('ALT_Alanine Transaminase (ALT)_U/L', 'ALT', labs, '(>|<)(.+)')

    #Albumin
    remove_punct('ALB_Albumin_g/L', 'Albumin', labs, '(>|<)(.+)')

    #Alk phos
    remove_punct('ALP_Alkaline Phosphatase (ALP)_U/L', 'AlkPhos', labs, '(>|<)(.+)')

    #AST
    remove_punct('AST_Aspartate Transaminase (AST)_U/L', 'AST', labs, '(>|<)(.+)')

    #Aspartate
    make_new_columns('ASP_Aspartate_umol/L', labs, ['Aspartate'], True)

    #Amylase
    remove_punct('AMY_Amylase_U/L', 'Amylase', labs, '(>|<)(.+)')

    #APTT
    remove_punct('APTT_APTT_Seconds', 'APTT', labs, '(>|<)(.+)')
    remove_punct('APTT_APTT_sec', 'APTT', labs, '(>|<)(.+)')

    #AG
    remove_punct('POCGAP_Anion Gap POC_mmol/L', 'Anion_gap', labs, '(>|<)(.+)')

    #BE
    remove_punct('POCBEIM_BE (M) POC_mmol/L', 'Base_excess', labs, '(>|<)(.+)')
    make_new_columns('POCBASEEXC_Base Excess POC_mmol/L', labs, ['Base_excess'], True)

    #Basophils
    make_new_columns('BASOPHILS_Basophil_x10*9/L', labs, ['Basophils'], True)
    make_new_columns('BASOS1_Basophils_x10*9/L', labs, ['Basophils'], True)

    #Bicarb
    make_new_columns('POCBICARBI_Bicarb (M) POC_mmol/L', labs, ['Bicarb'], True)

    #pH
    make_new_columns('PH_Blood pH', labs, ['pH'], True)
    make_new_columns('POCPHC_pH (C) POC_pH', labs, ['pH'], True)
    make_new_columns('POCPHISTATM_pH (M) POC_pH', labs, ['pH'], True)
    remove_punct('POCPHT_PH (T) POC_pH', 'pH', labs, '(>|<)(.+)')
    remove_punct('POCPH_pH POC_pH', 'pH', labs, '(>|<)(.+)')

    #Blood cultures
    blood_culture = [['No growth', 'No growth at 48 hours - incubation and continuous monitoring is ongoing'], ['Positive']]
    make_new_columns('BOTRESANO2_Bottle Result Anaerobic', labs, ['Blood_culture'], newcols1 = blood_culture)
    make_new_columns('BOTRESO2_Bottle Result Aerobic', labs, ['Blood_culture'], newcols1 = blood_culture)

    #CRP
    remove_punct('CRP_C-reactive Protein (CRP)_mg/L', 'CRP', labs, '(>|<)(.+)')

    #Calcium - need to convert the POC potassium = have just scaled it to the ref ranges
    make_new_columns('CA_Calcium_mmol/L', labs, ['Ca2+'], True)
    def conv_Ca(a):
        return (a - 1)*(5/3) + 2.2

    conv_column('POCCALCIUM_Calcium POC_mmol/L', labs, 'Ca2+', conv_Ca)
    conv_column('POCCALI_Calcium POC_mmol/L', labs, 'Ca2+', conv_Ca)

    #Cl
    make_new_columns('CL_Chloride_mmol/L', labs, ['Cl'], True)
    make_new_columns('POCCL_Chloride POC_mmol/L', labs, ['Cl'], True)

    #Cr
    remove_punct('CREAT_Creatinine_umol/L', 'Cr', labs, '(>|<)(.+)')

    #Hb
    make_new_columns('POCCTHB_ctHb POC_g/L', labs, ['Hb'], True)
    make_new_columns('HB1_Hb_g/L', labs, ['Hb'], True)

    #Eos
    make_new_columns('EOSINO_Eosinophil_x10*9/L', labs, ['Eosinophils'], True)
    make_new_columns('EOS1_Eosinophils_x10*9/L', labs, ['Eosinophils'], True)

    #Fibrinogen
    remove_punct('FIB_Fibrinogen_g/L', 'Fibrinogen', labs, '(>|<)(.+)')

    #FHHb, FMetHb, FO2Hb
    make_new_columns('POCFHHB_FHHb POC_%', labs, ['FHHb'], True)
    make_new_columns('POCFMETHB_FMetHb POC_%', labs, ['FMetHb'], True)
    make_new_columns('POCFO2HB_FO2HB POC_%', labs, ['FO2Hb'], True)

    #Glucose
    make_new_columns('GLU_Glucose_mmol/L', labs, ['Glucose'], True)
    make_new_columns('POCGLUCOSE_Glucose POC_mmol/L', labs, ['Glucose'], True)

    #HCT
    remove_punct('POCHCT_HCT POC_%', 'HCT%', labs, '(>|<)(.+)')
    make_new_columns('HCT1_HCT_L/L', labs, ['HCT'], True)

    #INR
    remove_punct('INR_INR', 'INR', labs, '(>|<)(.+)')

    #Lactate
    make_new_columns('LAC_Lactate_mmol/L', labs, ['Lactate'], True)
    make_new_columns('LAC_Lactate_mmol/L', labs, ['Lactate'], True)
    remove_punct('POCLAC_Lactate (art) POC_mmol/L', 'Lactate', labs, '(>|<)(.+)')
    remove_punct('POCLAC_Lactate (cap) POC_mmol/L', 'Lactate', labs, '(>|<)(.+)')
    remove_punct('POCLAC_Lactate (mv) POC_mmol/L', 'Lactate', labs, '(>|<)(.+)')
    remove_punct('POCLAC_Lactate (ven) POC_mmol/L', 'Lactate', labs, '(>|<)(.+)')
    remove_punct('POCLAC_Lactate POC_mmol/L', 'Lactate', labs, '(>|<)(.+)')
    make_new_columns('BLAC_Blood Lactate_mmol/L', labs, ['Lactate'], True)

    #Lymphs
    make_new_columns('LYMPHS1_Lymphocyte_x10*9/L', labs, ['Lymphs'], True)
    make_new_columns('LYMPH_Lymphocyte_x10*9/L', labs, ['Lymphs'], True)
    make_new_columns('LYMPCNT_Lymphocyte Count_x10*9/L', labs, ['Lymphs'], True)

    #Mg
    make_new_columns('MG_Magnesium_mmol/L', labs, ['Mg'], True)

    #Monocytes
    make_new_columns('MONOS1_Monocyte_x10*9/L', labs, ['Monocytes'], True)
    make_new_columns('MONOCYTES_Monocytes_x10*9/L', labs, ['Monocytes'], True)

    #Neuts
    make_new_columns('NEUTS1_Neutrophils_x10*9/L', labs, ['Neuts'], True)
    make_new_columns('NEUTSABS_Neutrophils_x10*9/L', labs, ['Neuts'], True)

    #Various POC blood gas values
    make_new_columns('POCP50_p50 POC_kPa', labs, ['P50'], True)
    make_new_columns('POCPCO2_paCO2 POC_kPa', labs, ['PaCO2'], True)
    make_new_columns('POCPCO2_pcCO2 POC_kPa', labs, ['PcCO2'], True)
    make_new_columns('POCPCO2IC_pCO2 (C) POC_kPa', labs, ['PcCO2'], True)
    make_new_columns('POCPPCO2IM_pCO2 (M) POC_kPa', labs, ['PmCO2'], True)
    remove_punct('POCPO2_paO2 POC_kPa', 'PaO2', labs, '(>|<)(.+)')
    remove_punct('POCPO2_pcO2 POC_kPa', 'PcO2', labs, '(>|<)(.+)')
    make_new_columns('POCPCO2_pmvCO2 POC_kPa', labs, ['PvCO2'], True)
    make_new_columns('POCPO2_pmvO2 POC_kPa', labs, ['PvO2'], True)
    remove_punct('POCPO2IC_pO2 (C) POC_kPa', 'PcO2', labs, '(>|<)(.+)')
    remove_punct('POCPP02IM_pO2 (M) POC_kPa', 'PmO2', labs, '(>|<)(.+)')
    remove_punct('POCPO2_pO2 POC_kPa', 'PO2', labs, '(>|<)(.+)')
    remove_punct('POCPCO2_pvCO2 POC_kPa', 'PvCO2', labs, '(>|<)(.+)')
    remove_punct('POCPO2_pvO2 POC_kPa', 'PvO2', labs, '(>|<)(.+)')

    #Phos
    make_new_columns('PO5_Phosphate_mmol/L', labs, ['Phos'], True)
    make_new_columns('PLT1_Platelet Count_x10*9/L', labs, ['Plts'], True)

    #Potassium
    remove_punct('K1_Potassium_mmol/L', 'K+', labs, '(>|<)(.+)')
    remove_punct('POCPOTASSIU_Potassium (art) POC_mmol/L', 'K+', labs, '(>|<)(.+)')
    remove_punct('POCPOTASSIU_Potassium (cap) POC_mmol/L', 'K+', labs, '(>|<)(.+)')
    remove_punct('POCPOTASSIU_Potassium (mv) POC_mmol/L', 'K+', labs, '(>|<)(.+)')
    remove_punct('POCPOTASSIU_Potassium (ven) POC_mmol/L', 'K+', labs, '(>|<)(.+)')
    remove_punct('POCPOTASSIU_Potassium POC_mmol/L', 'K+', labs, '(>|<)(.+)')

    #PT
    remove_punct('PT_Prothrombin Time_sec', 'PT', labs, '(>|<)(.+)')
    remove_punct('PT_Prothrombin Time_Seconds', 'PT', labs, '(>|<)(.+)')

    #Retics
    remove_punct('RETIC_Reticulocyte (Retic) count_x10*9/L', 'Retics', labs, '(>|<)(.+)')
    remove_punct('RETIC_Reticulocyte (Retic) count_10*9/L', 'Retics', labs, '(>|<)(.+)')

    #RBC
    remove_punct('BFRBC_Red Blood Cell Count_x 10*6 /litre', 'RBC', labs, '(>|<)(.+)')

    #Na
    remove_punct('NA1_Sodium_mmol/L', 'Na+', labs, '(>|<)(.+)')
    remove_punct('POCSODIUM_Sodium POC_mmol/L', 'Na+', labs, '(>|<)(.+)')
    remove_punct('POCNAISTAT_Sodium POC_mmol/L', 'Na+', labs, '(>|<)(.+)')

    #TT
    remove_punct('TT_Thrombin Time_sec', 'TT', labs, '(>|<)(.+)')
    remove_punct('TT_Thrombin Time_Seconds', 'TT', labs, '(>|<)(.+)')

    #Bili
    remove_punct('TBIL_Total bilirubin_umol/L', 'Bili', labs, '(>|<)(.+)')
    remove_punct('CALBILI_Total bilirubin (BU + BC)_umol/L', 'Bili', labs, '(>|<)(.+)')
    remove_punct('POCBILIRUBIN_Total Bilirubin POC_µmol/L', 'Bili', labs, '(>|<)(.+)')

    #WCC
    remove_punct('BFWBC_White Blood Cell Count_x 10*6 /litre', 'WCC', labs, '(>|<)(.+)')
    remove_punct('WBC1_WBC_x10*9/L', 'WCC', labs, '(>|<)(.+)')
    remove_punct('WBCAUTOSYNC_White Blood Cells_x10*9/L', 'WCC', labs, '(>|<)(.+)')
    remove_punct('WCC_White Cell Count_x10*9/L', 'WCC', labs, '(>|<)(.+)')
    remove_punct('OFWBC_White Cell Count_x10*9/L', 'WCC', labs, '(>|<)(.+)')

    #Now only merge the columns we have just made
    labs['taken_datetime'] = labs['collected_datetime']
    lab_cols = ['project_id', 'encounter_key', 'taken_datetime', 'ALT', 'Albumin', 'AlkPhos', 'AST', 'Aspartate', 
            'Amylase', 'APTT', 'Anion_gap', 'Base_excess', 'Basophils', 
            'Bicarb', 'pH', 'Blood_culture', 'Cr', 'CRP', 'Ca2+', 'Cl', 'Eosinophils', 
            'FHHb', 'FMetHb', 'FO2Hb', 'Glucose', 'HCT%', 'HCT', 'INR', 
            'Lactate', 'Lymphs', 'Mg', 'Monocytes', 'Neuts', 'P50', 'PaCO2', 
            'PcCO2', 'PmCO2', 'PaO2', 'PcO2', 'PmO2', 'PO2', 'PvCO2', 
            'PcO2', 'Phos', 'Plts', 'K+', 'PT', 'Retics', 'Na+', 'TT', 'Bili', 'WCC']

    flowsheet = pd.merge(flowsheet, labs.loc[:, lab_cols], how = 'outer', on = ['project_id', 'encounter_key', 'taken_datetime'])
    print('Flowsheet merged')
    flowsheet.to_csv('/store/DAMTP/dfs28/PICU_data/flowsheet_labs.csv.gz')
    print('Flowsheet with labs saved')
else:
    print('Flowsheet with labs already produced')

if (os.path.exists('/store/DAMTP/dfs28/PICU_data/flowsheet_all_times.csv.gz') == False):
    flowsheet = pd.read_csv('/store/DAMTP/dfs28/PICU_data/flowsheet_labs.csv.gz', parse_dates = ['taken_datetime'])

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Merging all times at", current_time)

    ## First sort by date (for some reason the flowsheet rows aren't properly sorted)
    flowsheet = flowsheet.sort_values(['project_id', 'taken_datetime'])
    
    #Remove nas
    flowsheet = flowsheet.set_index(
        ["project_id", 'encounter_key', 'taken_datetime']
        ).groupby(
            level=0
            ).transform(
                lambda x: sorted(x, key=lambda k: pd.isna(k))
                ).dropna(
                    axis=0, how="all"
                    ).reset_index()
                
    #Drop duplicated time rows
    duplicate_times = flowsheet.loc[:, ['project_id', 'encounter_key', 'taken_datetime']].duplicated(keep = 'last')
    flowsheet = flowsheet.loc[duplicate_times == False, :]

    #### Now enforce a minute by minute time frame
    #Get the start and end of each patient
    start_dates = flowsheet.groupby('project_id').taken_datetime.min()
    end_dates = flowsheet.groupby('project_id').taken_datetime.max()
    unique_patients = flowsheet['project_id'].unique()
    
    #Now make a new series to merge to which contains all of the possible time points
    all_times = pd.DataFrame(pd.Series([], dtype = 'datetime64[ns]'), columns =['taken_datetime'])
    all_times = all_times.append(pd.DataFrame([], columns = ['project_id']))
    for index, patient in enumerate(tqdm(unique_patients)):
        times = pd.date_range(start = start_dates[patient], end = end_dates[patient], freq = 'min')
        patients = pd.Series([patient]*len(times))
        times_df = pd.DataFrame({'taken_datetime': times, 'project_id': patients})
        all_times = all_times.append(times_df)

    print('Adding all times')
    flowsheet = all_times.merge(flowsheet, how = 'left', on = ['taken_datetime', 'project_id'])
    
    #Sort by date
    flowsheet = flowsheet.sort_values(['project_id', 'taken_datetime'])
    
    print('Saving all times')
    flowsheet.to_csv('/store/DAMTP/dfs28/PICU_data/flowsheet_all_times.csv.gz')
    print('Times saved')
else:
    print('Times already added')
#### Now do some feature engineering


if (os.path.exists('/store/DAMTP/dfs28/PICU_data/flowsheet_labs_interpolated.csv.gz') == False):
    flowsheet = pd.read_csv('/store/DAMTP/dfs28/PICU_data/flowsheet_all_times.csv.gz', parse_dates = ['taken_datetime'])

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Interpolating labs at", current_time)

    #Interpolate labs maximum 4 days
    lab_vars = ['ALT', 'Albumin', 'AlkPhos', 'AST', 'Aspartate', 
                'Amylase', 'APTT', 'Anion_gap', 'Base_excess', 'Basophils', 
                'Bicarb', 'pH', 'Blood_culture', 'Cr', 'CRP', 'Ca2+', 'Cl', 'Eosinophils', 
                'FHHb', 'FMetHb', 'FO2Hb', 'Glucose', 'HCT%', 'HCT', 'INR', 
                'Lactate', 'Lymphs', 'Mg', 'Monocytes', 'Neuts', 'P50', 'PaCO2', 
                'PcCO2', 'PmCO2', 'PaO2', 'PcO2', 'PmO2', 'PO2', 'PvCO2', 
                'PcO2', 'Phos', 'Plts', 'K+', 'PT', 'Retics', 'Na+', 'TT', 'Bili', 'WCC']

    for i in lab_vars:
        interpolate_cols(i, flowsheet, 'pad', how_far = 5760, limit_direction = 'forward', fill = fill_median)

    #Calculate strong ion gap
    Ca_charge = (flowsheet['Ca2+'] * 4.008) - (flowsheet['Albumin']/10 + 4) * 0.2495 * (-0.2 * (flowsheet['pH'] - 7.4) + 0.46) * 2 
    SID_calc = flowsheet['Na+'] + flowsheet['K+'] + (Ca_charge + 2.0266 * flowsheet['Mg']) - flowsheet['Cl'] - flowsheet['Lactate']
    SID_effective = 17.6 * flowsheet['PaCO2'] * (7.59 / (10 ** (flowsheet['pH'] * -1))) + (flowsheet['Albumin'] * (0.123 * flowsheet['pH'] - 0.631)) + flowsheet['Phos'] * (0.309 * flowsheet['pH'] - 0.469)
    SIG = SID_calc - SID_effective
    flowsheet['Strong_ion_gap'] = SIG

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Writing lab interpolation at", current_time)

    flowsheet.to_csv('/store/DAMTP/dfs28/PICU_data/flowsheet_labs_interpolated.csv.gz')

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Labs written at", current_time)



### Get demographics
if os.path.exists('/store/DAMTP/dfs28/PICU_data/flowsheet_demographics_applied.csv.gz') == False:
    print('Reading in flowsheet')
    flowsheet = pd.read_csv('/store/DAMTP/dfs28/PICU_data/flowsheet_labs_interpolated.csv.gz', parse_dates = ['taken_datetime'])

    print('Applying demographics')
    apply_demographic('birth_date', demographics, flowsheet, 'Age_yrs')
    flowsheet['time_to_death'] = np.timedelta64(70, 'Y')
    apply_demographic('death_date', demographics, flowsheet, 'time_to_death')
    apply_demographic('deceased_flag', demographics, flowsheet, 'died', False)
    apply_demographic('sex', demographics, flowsheet, 'sex', False)
    apply_demographic('ethnicity_name', demographics, flowsheet, 'ethnicity', False)

    print('Saving applied demographics')
    flowsheet.to_csv('/store/DAMTP/dfs28/PICU_data/flowsheet_demographics_applied.csv.gz')

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Demographics saved at", current_time)
else:
    print('Demographics already applied')



if (os.path.exists('/store/DAMTP/dfs28/PICU_data/flowsheet_ventilation_interpolated.csv.gz') == False):

    flowsheet = pd.read_csv('/store/DAMTP/dfs28/PICU_data/flowsheet_demographics_applied.csv.gz', parse_dates = ['taken_datetime'])

    #Bodyweight
    #print_all(flowsheet)
    #get_relevant_cols(flowsheet, 'weight|kg|gram|wt')

    def kg2g(new_values):
        return new_values*1000

    #R DRUG CALCULATION WEIGHT_grams
    make_new_columns('R DRUG CALCULATION WEIGHT_grams', flowsheet, ['Weight(g)'], True)
    #WEIGHT/SCALE_grams
    make_new_columns('WEIGHT/SCALE_grams', flowsheet, ['Weight(g)'], True)
    #R GOSH POST TREATMENT WEIGHT_kg
    conv_column('R GOSH POST TREATMENT WEIGHT_kg', flowsheet, 'Weight(g)', kg2g)
    #R GOSH HD PRE TREATMENT WEIGHT_kg
    conv_column('R GOSH HD PRE TREATMENT WEIGHT_kg', flowsheet, 'Weight(g)', kg2g)

    apply_wt_ht('Weight(g)', flowsheet, 'interpolated_wt_kg', scaling = 0.001)


    ### May need to estimate bodyweight for patients with no bodyweight
    #Using formula from paper found
    na_locs = flowsheet['interpolated_wt_kg'].isna()
    over1s = na_locs & flowsheet['Age_yrs'] >= 1
    under1s = na_locs & (flowsheet['Age_yrs'] < 1)

    #Fill in weight estimations
    flowsheet.loc[over1s, 'interpolated_wt_kg'] = flowsheet.loc[over1s, 'Age_yrs']*2 + 10
    flowsheet.loc[under1s, 'interpolated_wt_kg'] = (flowsheet.loc[under1s, 'Age_yrs']*12 + 9)/2





    #### Start with ventilation and O2
    print_all(flowsheet)

    """
    Here I'm making a ventilation column:
    0 for no ventilation
    1 for on oxygen
    2 for NIV/Optiflow
    3 for intubated and ventilated

    High frequency oscillated ventilation - this is because I want to assume that anyone with a tube is ventilated
    and possibly anyone with an EtCO2 value is also ventilated - its not clear from the variables which come from which
    so I will be able to assume that more patients are ventilated without accidentally marking them as not on high frequency
    oscillatory ventilation
    True/False (1/0)

    An intubated column - probably don't need this as above
    0 for not
    1 for intubated

    An EtCO2 column:
    Na for no value, rest just ported from any EtCO2 columns

    An Oxygen flow column - this will then need to be corrected for bodyweight

    An FiO2 column (not going to try to convert between but planning to have missingness columns)

    A tracheostomy column (yes/no)

    Could probably consider adding some further columns including:
    PEEP/EPAP, IPAP, some other ventilation settings if they are available
    """
    #'R GOSH ICU AIRWAY STATUS' 14 ##### ?
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Starting ventilation at", current_time)
    
    ventilated14 = [['SVIA', 'NO'], 
                    ['Low Flow NC', 'Face Mask', 'SVIA;Low Flow NC', 'Face Mask;Low Flow NC', 'SVIA;Face Mask', 
                    'Low Flow NC;SVIA'], 
                    ['High Flow NC', 'CPAP', 'SVIA;CPAP', 'SVIA;High Flow NC', 'Face Mask;CPAP', 'High Flow NC;SVIA', 'CPAP;SVIA'], 
                    ['CMV', 'CMV;CPAP', 'SVIA;CMV', 'NO;CMV', 'CMV;NO', 'HFO', 'HFO;NO']]
    make_new_columns('R GOSH ICU AIRWAY STATUS', flowsheet, ['Ventilation'], newcol1 = ventilated14)
    HFO14 = [[], ['HFO', 'HFO;NO']]
    make_new_columns(14, flowsheet, ['HFO'], newcol1 = HFO14)
    vent_df = pd.DataFrame(flowsheet.Ventilation.value_counts())

    #'R GOSH IP PANDA AIRWAY' 15
    ventilated15 = [[], [], [], ['Intubation']]
    make_new_columns('R GOSH IP PANDA AIRWAY', flowsheet, ['Ventilation'], newcol1 = ventilated15)
    vent_df['R GOSH IP PANDA AIRWAY'] = flowsheet.Ventilation.value_counts()


    #'R GOSH NICU AIRWAY STATUS' 16
    ventilated16 = [[], [], [], ['Conventional ventilation']]
    make_new_columns('R GOSH NICU AIRWAY STATUS', flowsheet, ['Ventilation'], newcol1 = ventilated16)
    vent_df['R GOSH NICU AIRWAY STATUS'] = flowsheet.Ventilation.value_counts()

    #'R GOSH RESUS ABCDE AIRWAY' 17
    ventilated16 = [[],[], [], ['Intubated', 'Intubated;Opening manoeuvres', 'Intubated;Patent']]
    make_new_columns('R GOSH RESUS ABCDE AIRWAY', flowsheet, ['Ventilation'], newcol1 = ventilated16)
    vent_df['R GOSH RESUS ABCDE AIRWAY'] = flowsheet.Ventilation.value_counts()
    
    #'R AIRWAY LDA INSERTION ATTEMPTS' 19
    ventilated19 = [[], [], [], [1, 2, '1', '2', '3 or more']]
    make_new_columns('R AIRWAY LDA INSERTION ATTEMPTS', flowsheet, ['Ventilation'], newcol1 = ventilated19)
    vent_df['R AIRWAY LDA INSERTION ATTEMPTS'] = flowsheet.Ventilation.value_counts()
    
    #'R GOSH IP CATS AIRWAY STATUS' 20
    ventilated20 = [[], [], [],  ['Intubated']]
    make_new_columns('R GOSH IP CATS AIRWAY STATUS', flowsheet, ['Ventilation'], newcol1 = ventilated20)
    vent_df['R GOSH IP CATS AIRWAY STATUS'] = flowsheet.Ventilation.value_counts()
    
    #34, 'R GOSH IP ARE THEY INTUBATED AND VENTILATED?'
    ventilated34 = [[], [], ['No - Non-invasive Ventilation', 'No - Optiflow'], ['Yes - Invasive ventilation']]
    make_new_columns('R GOSH IP ARE THEY INTUBATED AND VENTILATED?', flowsheet, ['Ventilation'], newcol1 = ventilated34)
    vent_df['R GOSH IP ARE THEY INTUBATED AND VENTILATED?'] = flowsheet.Ventilation.value_counts()
    
    #136, 'R GOSH TRACHE CUFF'
    trache136 = [[], ['Inflated', 'Deflated', 'Deflated;Inflated', 'Inflated;Deflated', 'Deflated;Inflated;Leak present', 
                'Leak present;Inflated', 'Inflated;Leak present', 'Leak present', 'Inflated;Deflated;Leak present', 
                'Other (Comment)', 'Cuffless', 'Cuffless;Leak present', 'Leak present;Cuffless', 'Deflated;Leak present', 
                'Inflated;Other (Comment)', 'Leak present;Other (Comment)', 'Deflated;Leak present;Other (Comment)', 
                'Deflated;Other (Comment)', 'Deflated;Leak present;Inflated', 'Leak present;Deflated', 
                'Inflated;Leak present;Deflated', 'Deflated;Inflated;Other (Comment)']]
    make_new_columns('R GOSH TRACHE CUFF', flowsheet, ['Tracheostomy'], newcol1 = trache136)
    
    #204, 'R GOSH ICU END TIDAL CO2_% Should work out if this means they are ventilated
    make_new_columns('R GOSH ICU END TIDAL CO2_%', flowsheet, ['ETCO2'], True)

    #205, 'R VENT ETCO2_kPa Should work out if this means they are ventilated
    make_new_columns('R VENT ETCO2_kPa', flowsheet, ['ETCO2'], True)

    #206, 'R IP VENT ALARM ETCO2/TCM MAX_mmHg
    make_new_columns('R IP VENT ALARM ETCO2/TCM MAX_mmHg', flowsheet, ['ETCO2'], True)

    #210, 'R ETT TO LIP_cm
    #unique_vent210 = flowsheet.loc[:, 'R ETT TO LIP_cm'].unique()
    #vent210 = [[], [], [], unique_vent210[range(1, len(unique_vent210))].tolist()]
    #make_new_columns('R ETT TO LIP_cm', flowsheet, ['Ventilation'], newcol1 = vent210)
    #vent_df['R ETT TO LIP_cm'] = flowsheet.Ventilation.value_counts()
    
    #211, 'R ETT TO NOSE_cm'
    #unique_vent211 = flowsheet.loc[:, 'R ETT TO NOSE_cm'].unique()
    #vent211 = [[], [], [], unique_vent211[range(1, len(unique_vent211))].tolist()]
    #make_new_columns('R ETT TO NOSE_cm', flowsheet, ['Ventilation'], newcol1 = vent211)
    #vent_df['R ETT TO NOSE_cm'] = flowsheet.Ventilation.value_counts()
    
    #214, 'R AN AGENTS DESFLURANE EXPIRED_%'
    unique_vent214 = flowsheet.loc[:, 'R AN AGENTS DESFLURANE EXPIRED_%'].unique()
    vent214 = [[], [], [], unique_vent214[range(1, len(unique_vent214))].tolist()]
    make_new_columns('R AN AGENTS DESFLURANE EXPIRED_%', flowsheet, ['Ventilation'], newcol1 = vent214)
    vent_df['R AN AGENTS DESFLURANE EXPIRED_%'] = flowsheet.Ventilation.value_counts()
    
    #215, 'R AN AGENTS ISOFLURANE EXPIRED_%'
    unique_vent215 = flowsheet.loc[:, 'R AN AGENTS ISOFLURANE EXPIRED_%'].unique()
    vent215 = [[], [], [], unique_vent215[range(1, len(unique_vent215))].tolist()]
    make_new_columns('R AN AGENTS ISOFLURANE EXPIRED_%', flowsheet, ['Ventilation'], newcol1 = vent215)
    vent_df['R AN AGENTS ISOFLURANE EXPIRED_%'] = flowsheet.Ventilation.value_counts()
    
    #216, 'R AN EXPIRED MINUTE VOLUME_L/min'
    make_new_columns('R AN EXPIRED MINUTE VOLUME_L/min', flowsheet, ['Ventilation'], 
                        newcol1 = make_allnonNa(flowsheet, 'R AN EXPIRED MINUTE VOLUME_L/min', 4, 4))
    vent_df['R AN EXPIRED MINUTE VOLUME_L/min'] = flowsheet.Ventilation.value_counts()
    
    #217, 'R AN EXPIRED N2O_%'\\
    make_new_columns('R AN EXPIRED N2O_%', flowsheet, ['Ventilation'], newcol1 = make_allnonNa(flowsheet, 'R AN EXPIRED N2O_%', 4, 4))
    vent_df['R AN EXPIRED N2O_%'] = flowsheet.Ventilation.value_counts()
    
    #218, 'R AN AGENTS SEVOFLURANE EXPIRED_%'
    make_new_columns('R AN AGENTS SEVOFLURANE EXPIRED_%', flowsheet, ['Ventilation'], newcol1 = make_allnonNa(flowsheet, 'R AN AGENTS SEVOFLURANE EXPIRED_%', 4, 4))
    vent_df['R AN AGENTS SEVOFLURANE EXPIRED_%'] = flowsheet.Ventilation.value_counts()
    
    #219, 'R GOSH EXP TV LITRES_L' - should consider expired tidal volume?
    make_new_columns('R GOSH EXP TV LITRES_L', flowsheet, ['Ventilation'], newcol1 = make_allnonNa(flowsheet, 'R GOSH EXP TV LITRES_L', 4, 4))
    vent_df['R GOSH EXP TV LITRES_L'] = flowsheet.Ventilation.value_counts()
    
    #220, 'R VENT EXP TIDAL VOLUME_mL' - was this where the error was?
    make_new_columns('R VENT EXP TIDAL VOLUME_mL', flowsheet, ['Ventilation'], newcol1 = make_allnonNa(flowsheet, 'R VENT EXP TIDAL VOLUME_mL', 4, 4))
    vent_df['R VENT EXP TIDAL VOLUME_mL'] = flowsheet.Ventilation.value_counts()
    
    #234, 'R AN FIO2_%'
    cor_FiO2('R AN FIO2_%', flowsheet, 'FiO2')

    #235, 'R PERF FIO2_%'
    cor_FiO2('R PERF FIO2_%', flowsheet, 'FiO2')

    #238, 'R GOSH IP FIO2 SETTING_%'
    cor_FiO2('R GOSH IP FIO2 SETTING_%', flowsheet, 'FiO2')

    #239, 'R VENT INSP FLOWS_L/min'
    make_new_columns('R VENT INSP FLOWS_L/min', flowsheet, ['O2Flow'], True)

    #240, 'R GOSH VENT FLOW ML/KG_mL/Kg'
    make_new_columns('R GOSH VENT FLOW ML/KG_mL/Kg', flowsheet, ['O2Flow/kg'], True)

    #241, 'R IP VENT FLOW OBS_L/min'
    make_new_columns('R IP VENT FLOW OBS_L/min', flowsheet, ['O2Flow'], True)

    #242, 'R GOSH AIRVO FLOW_L/min' Highflow o2
    make_new_columns('R GOSH AIRVO FLOW_L/min' , flowsheet, ['O2Flow'], True)
    make_new_columns('R GOSH AIRVO FLOW_L/min' , flowsheet, ['Ventilation'], newcol1 = make_allnonNa(flowsheet, 'R GOSH AIRVO FLOW_L/min' , 2, 4))
    vent_df['R GOSH AIRVO FLOW_L/min'] = flowsheet.Ventilation.value_counts()
    
    #R GOSH VENT FABIAN O2 THERAPY FLOW_L/min HFNC/ NIV
    make_new_columns('R GOSH VENT FABIAN O2 THERAPY FLOW_L/min', flowsheet, ['O2Flow'], True)
    make_new_columns('R GOSH VENT FABIAN O2 THERAPY FLOW_L/min', flowsheet, ['Ventilation'], 
                    newcol1 = make_allnonNa(flowsheet, 'R GOSH VENT FABIAN O2 THERAPY FLOW_L/min', 2, 4))
    vent_df['R GOSH VENT FABIAN O2 THERAPY FLOW_L/min'] = flowsheet.Ventilation.value_counts()
    
    #(284, 'R GOSH HFO BASE FLOW READING')
    #print_allunique(flowsheet, 284)
    make_new_columns('R GOSH HFO BASE FLOW READING', flowsheet, ['HFO'], newcol1 = make_allnonNa(flowsheet, 'R GOSH HFO BASE FLOW READING', 2, 2))
    make_new_columns('R GOSH HFO BASE FLOW READING', flowsheet, ['O2Flow'], True)

    #285, 'R GOSH HFO BASE FLOW SETTING')
    make_new_columns('R GOSH HFO BASE FLOW SETTING', flowsheet, ['HFO'], newcol1 = make_allnonNa(flowsheet, 'R GOSH HFO BASE FLOW SETTING', 2, 2))
    make_new_columns('R GOSH HFO BASE FLOW SETTING', flowsheet, ['O2Flow'], True)

    #287, 'R GOSH HFO FREQUENCY READING_Hz')
    make_new_columns('R GOSH HFO FREQUENCY READING_Hz', flowsheet, ['HFO'], newcol1 = make_allnonNa(flowsheet, 'R GOSH HFO FREQUENCY READING_Hz', 2, 2))

    #(288, 'R GOSH IP HFV FREQUENCY SETTING_Hz')
    make_new_columns('R GOSH IP HFV FREQUENCY SETTING_Hz', flowsheet, ['HFO'], newcol1 = make_allnonNa(flowsheet, 'R GOSH IP HFV FREQUENCY SETTING_Hz', 2, 2))

    #(290, 'R GOSH IP HFV TIDAL VOLUME_mL')HFV is high frequency ventilation
    make_new_columns('R GOSH IP HFV TIDAL VOLUME_mL', flowsheet, ['HFO'], newcol1 = make_allnonNa(flowsheet, 'R GOSH IP HFV TIDAL VOLUME_mL', 2, 2))

    #(291, 'R GOSH HIGH FIO2_%') 
    cor_FiO2('R GOSH HIGH FIO2_%', flowsheet, 'FiO2')

    #R IP VENT HUMIDIFIER TEMP (SET)')
    make_new_columns("R IP VENT HUMIDIFIER TEMP (SET)", flowsheet, ['Ventilation'], 
                        newcol1 = make_allnonNa(flowsheet, 'R IP VENT HUMIDIFIER TEMP (SET)', 4, 4))
    vent_df["R IP VENT HUMIDIFIER TEMP (SET)"] = flowsheet.Ventilation.value_counts()
    
    #(317, 'R GOSH INSPRIATORY PRESSURE')
    make_new_columns('R GOSH INSPRIATORY PRESSURE', flowsheet, ['IPAP'], True)

    #(320, 'R VENT INSP TIME_Sec')
    make_new_columns('R VENT INSP TIME_Sec', flowsheet, ['Ventilation'], newcol1 = make_allnonNa(flowsheet,'R VENT INSP TIME_Sec',  4, 4))
    vent_df['R VENT INSP TIME_Sec'] = flowsheet.Ventilation.value_counts()
    
    #(322, 'R AN AGENTS DESFLURANE INSPIRED_%')
    make_new_columns('R AN AGENTS DESFLURANE INSPIRED_%', flowsheet, ['Ventilation'], newcol1 = make_allnonNa(flowsheet,'R AN AGENTS DESFLURANE INSPIRED_%',  4, 4))
    vent_df['R AN AGENTS DESFLURANE INSPIRED_%'] = flowsheet.Ventilation.value_counts()
    
    #(323, 'R FIO2_%')
    cor_FiO2('R FIO2_%', flowsheet, 'FiO2')

    #(324, 'R AN AGENTS ISOFLURANE INSPIRED_%')
    make_new_columns('R AN AGENTS ISOFLURANE INSPIRED_%', flowsheet, ['Ventilation'], newcol1 = make_allnonNa(flowsheet,  'R AN AGENTS ISOFLURANE INSPIRED_%',  4, 4))
    vent_df['R AN AGENTS ISOFLURANE INSPIRED_%'] = flowsheet.Ventilation.value_counts()
    
    #(325, 'R AN INSPIRED N2O_%')
    make_new_columns('R AN INSPIRED N2O_%', flowsheet, ['Ventilation'], newcol1 = make_allnonNa(flowsheet,  'R AN INSPIRED N2O_%',  4, 4))
    vent_df['R AN INSPIRED N2O_%'] = flowsheet.Ventilation.value_counts()
    
    #(326, 'R AN INSPIRED NO2_ppm')
    make_new_columns('R AN INSPIRED NO2_ppm', flowsheet, ['Ventilation'], newcol1 = make_allnonNa(flowsheet,  'R AN INSPIRED NO2_ppm',  4, 4))
    vent_df['R AN INSPIRED NO2_ppm'] = flowsheet.Ventilation.value_counts()
    
    #(327, 'R AN AGENTS NITRIC OXIDE_ppm')
    make_new_columns('R AN AGENTS NITRIC OXIDE_ppm', flowsheet, ['Ventilation'], newcol1 = make_allnonNa(flowsheet,  'R AN AGENTS NITRIC OXIDE_ppm',  4, 4))
    vent_df['R AN AGENTS NITRIC OXIDE_ppm'] = flowsheet.Ventilation.value_counts()
    
    #(330, 'R AN AGENTS SEVOFLURANE INSPIRED_%')
    make_new_columns('R AN AGENTS SEVOFLURANE INSPIRED_%', flowsheet, ['Ventilation'], newcol1 = make_allnonNa(flowsheet,  'R AN AGENTS SEVOFLURANE INSPIRED_%',  4, 4))
    vent_df['R AN AGENTS SEVOFLURANE INSPIRED_%'] = flowsheet.Ventilation.value_counts()
    
    #(333, 'R GOSH IPAP')
    make_new_columns('R GOSH IPAP', flowsheet, ['IPAP'], True)

    #(332, 'R GOSH ASTRAL IPAP_cm H2O')
    make_new_columns('R GOSH ASTRAL IPAP_cm H2O', flowsheet, ['IPAP'], True)

    #(334, 'R GOSH NIPPY IPAP_cmH2O')
    make_new_columns('R GOSH NIPPY IPAP_cmH2O', flowsheet, ['IPAP'], True)

    #(335, 'R GOSH STELLA IPAP_cm H2O')
    make_new_columns('R GOSH STELLA IPAP_cm H2O', flowsheet, ['IPAP'], True)
    make_new_columns('R IP VENT MV LOW_L/min', flowsheet, ['Ventilation_L_min'], True)
    make_new_columns('R GOSH VENT RATE', flowsheet, ['Ventilation(ml)'], True)
    make_new_columns('R GOSH VENT RATE OBSERVED', flowsheet, ['Ventilation(ml)'], True)

    #(387, 'R VENT MAP_cm H2O')
    make_new_columns('R VENT MAP_cm H2O', flowsheet, ['MeanAirwayPressure'], True)

    #(388, 'R IP VENT MEAN AIRWAY PRESSURE (SET)')
    make_new_columns('R IP VENT MEAN AIRWAY PRESSURE (SET)', flowsheet, ['MeanAirwayPressure'], True)

    #(424, 'R AN AGENTS O2_L/min')
    make_new_columns('R AN AGENTS O2_L/min', flowsheet, ['O2Flow'], True)

    #(428, 'R RT OXYGEN FLOW RATE_L/min')
    make_new_columns('R RT OXYGEN FLOW RATE_L/min', flowsheet, ['O2Flow'], True)

    # 429, 'R OXYGEN FLOW RATE_L/min')
    make_new_columns('R OXYGEN FLOW RATE_L/min', flowsheet, ['O2Flow'], True)

    #(442, 'R GOSH AIRVO OXYGEN %_%')
    cor_FiO2('R GOSH AIRVO OXYGEN %_%', flowsheet, 'FiO2')

    #(492, 'R GOSH ASTRAL PEEP_cm H2O')
    make_new_columns('R GOSH ASTRAL PEEP_cm H2O', flowsheet, ['EPAP'], True)

    #(493, 'R GOSH PEEP_cmH2O')
    make_new_columns('R GOSH PEEP_cmH2O', flowsheet, ['EPAP'], True)

    #(494, 'R VENT PEEP_cm H2O')
    make_new_columns('R VENT PEEP_cm H2O', flowsheet, ['EPAP'], True)

    #(496, 'R GOSH IP VENT PEEP SETTING_cm H2O')
    make_new_columns('R GOSH IP VENT PEEP SETTING_cm H2O', flowsheet, ['EPAP'], True)

    #(497, 'R AN VENT PEEP_cm H20')
    make_new_columns('R AN VENT PEEP_cm H20', flowsheet, ['EPAP'], True)

    #Now use some columns to update others
    make_new_columns('ETCO2', flowsheet, ['Ventilation'], newcol1 = make_allnonNa(flowsheet,  'ETCO2',  4, 4))
    vent_df['ETCO2'] = flowsheet.Ventilation.value_counts()
    
    make_new_columns('HFO', flowsheet, ['Ventilation'], newcol1 = make_allnonNa(flowsheet,  'HFO',  4, 4))
    vent_df['HFO'] = flowsheet.Ventilation.value_counts()
    
    #Some weight corrected values
    O2nas = flowsheet['O2Flow/kg'].isna()
    flowsheet.loc[O2nas, 'O2Flow/kg'] = flowsheet.loc[O2nas, 'O2Flow']/flowsheet.loc[O2nas, 'interpolated_wt_kg'] 

    #198, 'R GOSH ASTRAL EPAP_cm H2O') - Astral is NIV
    unique_vent198 = flowsheet.loc[:, 'R GOSH ASTRAL EPAP_cm H2O'].unique()
    vent198 = [[], [], unique_vent198[range(1, len(unique_vent198))].tolist(), []]
    make_new_columns('R GOSH ASTRAL EPAP_cm H2O', flowsheet, ['Ventilation'], newcol1 = vent198)
    make_new_columns('R GOSH ASTRAL EPAP_cm H2O', flowsheet, ['EPAP'], True)
    vent_df['R GOSH ASTRAL EPAP_cm H2O'] = flowsheet.Ventilation.value_counts()
    
    #199, 'R GOSH EPAP'
    make_new_columns('R GOSH EPAP', flowsheet, ['EPAP'], True)

    #200, 'R GOSH NIPPY EPAP_cmH2O
    unique_vent200 = flowsheet.loc[:, 'R GOSH NIPPY EPAP_cmH2O'].unique()
    vent200 = [[], [], unique_vent200[range(1, len(unique_vent200))].tolist(), []]
    make_new_columns('R GOSH NIPPY EPAP_cmH2O', flowsheet, ['Ventilation'], newcol1 = vent200)
    make_new_columns(200, flowsheet, ['EPAP'], True)
    vent_df['R GOSH NIPPY EPAP_cmH2O'] = flowsheet.Ventilation.value_counts()

    #127, 'R GOSH CPAP CMH2O_cmH2O'
    make_new_columns('R GOSH CPAP CMH2O_cmH2O', flowsheet, ['IPAP'], True)
    make_new_columns('R GOSH CPAP CMH2O_cmH2O', flowsheet, ['EPAP'], True)
    unique_vent127 = flowsheet.loc[:, 'R GOSH CPAP CMH2O_cmH2O'].unique()
    vent127 = [[], [], unique_vent127[range(1, len(unique_vent127))].tolist(), []]
    make_new_columns('R GOSH CPAP CMH2O_cmH2O', flowsheet, ['Ventilation'], newcol1 = vent127)
    vent_df['R GOSH CPAP CMH2O_cmH2O'] = flowsheet.Ventilation.value_counts()
    
    #Moved this here as seems most reliable
    ventilated14 = [['SVIA', 'NO'], 
                    ['Low Flow NC', 'Face Mask', 'SVIA;Low Flow NC', 'Face Mask;Low Flow NC', 'SVIA;Face Mask', 
                    'Low Flow NC;SVIA'], 
                    ['High Flow NC', 'CPAP', 'SVIA;CPAP', 'SVIA;High Flow NC', 'Face Mask;CPAP', 'High Flow NC;SVIA', 'CPAP;SVIA'], 
                    ['CMV', 'CMV;CPAP', 'SVIA;CMV', 'NO;CMV', 'CMV;NO', 'HFO', 'HFO;NO']]
    make_new_columns('R GOSH ICU AIRWAY STATUS', flowsheet, ['Ventilation'], newcol1 = ventilated14)
    HFO14 = [[], ['HFO', 'HFO;NO']]
    make_new_columns(14, flowsheet, ['HFO'], newcol1 = HFO14)
    
    ##Now fill missing values
    fill_missing('Ventilation', flowsheet, 'Ventilation_missing')
    fill_missing('O2Flow/kg', flowsheet, 'O2Flow/kg_missing')
    fill_missing('IPAP', flowsheet, 'IPAP_missing')
    fill_missing('EPAP', flowsheet, 'EPAP_missing')
    fill_missing('FiO2', flowsheet, 'FiO2_missing')
    fill_missing('HFO', flowsheet, 'HFO_missing')
    fill_missing('Tracheostomy', flowsheet, 'Tracheostomy_missing')
    fill_missing('Ventilation(ml)', flowsheet, 'Ventilation(ml)_missing')
    fill_missing('MeanAirwayPressure', flowsheet, 'MeanAirwayPressure_missing')
    fill_missing('ETCO2', flowsheet, 'ETCO2_missing')

    #Now do interpolation
    interpolate_cols('Ventilation', flowsheet, 'pad', how_far = 90, limit_direction = 'forward', fill = 0)
    interpolate_cols('O2Flow/kg', flowsheet, 'linear', how_far = 90, limit_direction = 'both', fill = 0)
    interpolate_cols('O2Flow', flowsheet, 'linear', how_far = 90, limit_direction = 'both', fill = 0)
    interpolate_cols('IPAP', flowsheet, 'linear', how_far = 90, limit_direction = 'both', fill = 0)
    interpolate_cols('EPAP', flowsheet, 'linear', how_far = 90, limit_direction = 'both', fill = 0)
    interpolate_cols('FiO2', flowsheet, 'linear', how_far = 90, limit_direction = 'both', fill = 0.21)
    interpolate_cols('HFO', flowsheet, 'pad', how_far = 90, limit_direction = 'forward', fill = 0)
    interpolate_cols('Tracheostomy', flowsheet, 'pad', how_far = 90, limit_direction = 'forward', fill = 0)
    interpolate_cols('Ventilation(ml)', flowsheet, 'linear', how_far = 90, limit_direction = 'both', fill = 0)
    interpolate_cols('MeanAirwayPressure', flowsheet, 'linear', how_far = 90, limit_direction = 'both', fill = 0)
    interpolate_cols('ETCO2', flowsheet, 'linear', how_far = 90, limit_direction = 'both', fill = 0)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Saving ventilation at", current_time)

    flowsheet.to_csv('/store/DAMTP/dfs28/PICU_data/flowsheet_ventilation_interpolated.csv.gz')
else:
    print('Ventilation already interpolated')

#### Work on BP now

"""
This column is just going to be BP as pulled from all other columns containing what appears to be BP
I also wanted to include CVP but I'm unclear about these values - they appear to be in the 100s which would be 
an unreasonable value

For some reason there appears to be a problem with the MAP column - I if this is difficult to solve I will just 
calculate this from the SBP and DBP columns - should possibly change the function so it doesn't automatically produce
a MAP/ have the function calls not to produce MAP
"""

if (os.path.exists('/store/DAMTP/dfs28/PICU_data/flowsheet_pSOFA_interpolated.csv.gz') == False):
    flowsheet = pd.read_csv('/store/DAMTP/dfs28/PICU_data/flowsheet_ventilation_interpolated.csv.gz', parse_dates = ['taken_datetime'])
    
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Starting BP at", current_time)

    #Make a unified BP column
    #4, 'R AN ARTERIAL BLOOD PRESSURE'
    conv_BP('R AN ARTERIAL BLOOD PRESSURE', flowsheet, 3, 'SysBP', 'DiaBP')

    #5, 'R AN MEAN ARTERIAL BLOOD PRESSURE_mmHg'
    conv_BP('R AN MEAN ARTERIAL BLOOD PRESSURE_mmHg', flowsheet, False, 'MAP')

    #26, 'R AN CENTRAL AORTIC PRESSURE (CAP)'
    conv_BP('R AN CENTRAL AORTIC PRESSURE (CAP)', flowsheet, 3, 'SysBP', 'DiaBP')

    #27, 'R AN CENTRAL AORTIC PRESSURE (CAP) MEAN_mmHg'
    conv_BP('R AN CENTRAL AORTIC PRESSURE (CAP) MEAN_mmHg', flowsheet, False, 'MAP')

    #37, 'R GOSH IP MEAN ARTERIAL BLOOD PRESSURE_mmHg')
    conv_BP('R GOSH IP MEAN ARTERIAL BLOOD PRESSURE_mmHg', flowsheet, False, 'MAP')

    #39, 'R ARTERIAL LINE BLOOD PRESSURE')
    conv_BP('R ARTERIAL LINE BLOOD PRESSURE', flowsheet, 3, 'SysBP', 'DiaBP')

    #(40, 'R MAP A-LINE_mmHg')
    conv_BP('R MAP A-LINE_mmHg', flowsheet, False, 'MAP')
    
    conv_BP('R MAP', flowsheet, False, 'MAP')

    #(81, 'BLOOD PRESSURE')
    conv_BP('BLOOD PRESSURE', flowsheet, 3, 'SysBP', 'DiaBP')

    #(411, 'R AN NIBP MEAN_mmHg')
    conv_BP('R AN NIBP MEAN_mmHg', flowsheet, False, 'MAP')

    ##### Now fix the problem with MAP
    #Pull out where MAP is a **/** format
    slash = re.compile('.*/+.*')
    strings = [i for i, j in enumerate(flowsheet['MAP']) if not slash.match(str(j)) == None]
    strings = np.array(strings)

    if np.shape(strings)[0] > 0:
        #Find the location where sysbp and diabp are missing but map not
        diaNAs = np.where(flowsheet['DiaBP'].isna())
        sysNAs = np.where(flowsheet['SysBP'].isna())
        BP_NAs = np.union1d(diaNAs, sysNAs)
        MAPnotBP = np.intersect1d(BP_NAs, strings)
        #Can see actually none here that are relevant

        #Now just fix the problem for the MAPs that are strings
        newBPs = flowsheet.loc[strings, 'MAP'].str.split("/", n = 1, expand = True)
        newBPs = newBPs.astype('float')
        flowsheet.loc[strings, 'MAP'] = newBPs.loc[:, 0]*(1/3) + newBPs.loc[:, 1]*(2/3)

    #Now fill in where SBP and DBP not NA but MAP is:
    diaBPs = np.where(flowsheet.DiaBP.isna() == False)
    sysBPs = np.where(flowsheet.SysBP.isna() == False)
    dsBPs = np.intersect1d(diaBPs, sysBPs)
    MAPs = np.where(flowsheet.MAP.isna() == False)
    BPnotMAP = np.setdiff1d(dsBPs, MAPs)
    flowsheet.MAP[BPnotMAP] = flowsheet.DiaBP[BPnotMAP] + (flowsheet.SysBP[BPnotMAP] - flowsheet.DiaBP[BPnotMAP])/3

    #Fill in where missing 
    fill_missing('SysBP', flowsheet, 'SysBP_missing')
    fill_missing('DiaBP', flowsheet, 'DiaBP_missing')
    fill_missing('MAP', flowsheet, 'MAP_missing')

    interpolate_cols('SysBP', flowsheet, 'linear', 'SBP', how_far = 90, limit_direction = 'both', fill = fill_median_HRBP)
    interpolate_cols('DiaBP', flowsheet, 'linear', 'DBP', how_far = 90, limit_direction = 'both', fill = fill_median_HRBP)
    interpolate_cols('MAP', flowsheet, 'linear', 'MAP', how_far = 90, limit_direction = 'both', fill = fill_median_HRBP)
    
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("BP now interpolated:", current_time)

    #### Now do HR
    get_relevant_cols(flowsheet, 'HR| rate')

    #R GOSH IP HEART RATE ECG_beats per minute
    make_new_columns('R GOSH IP HEART RATE ECG_beats per minute', flowsheet, ['HR'], True)
    #R AN HEART RATE ECG_beats per minute
    make_new_columns('R AN HEART RATE ECG_beats per minute', flowsheet, ['HR'], True)
    #R AN SPO2 HR PULSE
    make_new_columns('R AN SPO2 HR PULSE', flowsheet, ['HR'], True)
    #R GOSH IP HEART RATE PLETHYSMOGRAM
    make_new_columns('R GOSH IP HEART RATE PLETHYSMOGRAM', flowsheet, ['HR'], True)

    fill_missing('HR', flowsheet, 'HR_missing')

    #Should probably do something as above with median HR
    interpolate_cols('HR', flowsheet, 'linear', 'HR', how_far = 90, limit_direction = 'both', fill = fill_median_HRBP)





    #GCS/ AVPU
    print_all(flowsheet)
    #get_relevant_cols(flowsheet, 'GCS|AVPU|Comfort|Consciousness|Alert|Pupils|GLASGOW COMA SCALE|PEWS')

    #Comfort score
    #R GOSH COMFORT ALERTNESS 22
    make_new_columns('R GOSH COMFORT ALERTNESS', flowsheet, ['Comfort:Alertness'], True)
    #R GOSH COMFORT BP 84
    make_new_columns('R GOSH COMFORT BP', flowsheet, ['Comfort:BP'], True)
    #R GOSH COMFORT CALMNESS
    make_new_columns('R GOSH COMFORT CALMNESS', flowsheet, ['Comfort:Calmness'], True)
    #R GOSH COMFORT SCORE
    make_new_columns('R GOSH COMFORT SCORE', flowsheet, ['Comfort'], True)
    #R GOSH IP COMFORT SCORE FOR GOSH GO
    make_new_columns('R GOSH IP COMFORT SCORE FOR GOSH GO', flowsheet, ['Comfort'], True)
    #R GOSH COMFORT HR
    make_new_columns('R GOSH COMFORT HR', flowsheet, ['Comfort:HR'], True)
    #R GOSH COMFORT RESP NO VENT - Consider this as surrogate for no vent/ vent below?
    make_new_columns('R GOSH COMFORT RESP NO VENT', flowsheet, ['Comfort:Resp'], True)
    #R GOSH COMFORT RESP VENT
    make_new_columns('R GOSH COMFORT RESP VENT', flowsheet, ['Comfort:Resp'], True)
    
    fill_missing('Comfort:Alertness', flowsheet, 'Comfort:Alertness_missing')
    fill_missing('Comfort:BP', flowsheet, 'Comfort:BP_missing')
    fill_missing('Comfort:Calmness', flowsheet, 'Comfort:Calmness_missing')
    fill_missing('Comfort:HR', flowsheet, 'Comfort:HR_missing')
    fill_missing('Comfort:Resp', flowsheet, 'Comfort:Resp_missing')
    fill_missing('Comfort', flowsheet, 'Comfort_missing')
    interpolate_cols('Comfort:Alertness', flowsheet, 'pad', how_far = 90, limit_direction = 'forward', fill = 4)
    interpolate_cols('Comfort:BP', flowsheet, 'pad', how_far = 90, limit_direction = 'forward', fill = 2)
    interpolate_cols('Comfort:Calmness', flowsheet, 'pad', how_far = 90, limit_direction = 'forward', fill = 1)
    interpolate_cols('Comfort:Resp', flowsheet, 'pad', how_far = 90, limit_direction = 'forward', fill = 2)
    interpolate_cols('Comfort:HR', flowsheet, 'pad', how_far = 90, limit_direction = 'forward', fill = 2)
    interpolate_cols('Comfort', flowsheet, 'pad', how_far = 90, limit_direction = 'forward', fill = 20)
    #Alertness
    #make_new_columns(867, flowsheet, ['Comfort:Alertness'], True)

    #R GOSH IP AVPU SCORE
    make_new_columns('R GOSH IP AVPU SCORE', flowsheet, ['AVPU'], newcols1 = [['U'], ['P'], ['V'], ['A']])
    #R PEDSCPN GLASGOW COMA SCALE (AGE 6 MONTHS TO 2 YEARS) BEST AUD
    make_new_columns('R PEDSCPN GLASGOW COMA SCALE (AGE 6 MONTHS TO 2 YEARS) BEST AUD', flowsheet, ['GCS_V'], True)
    #R PEDSCPN GLASGOW COMA SCALE (AGE GREATER THAN 2 YEARS) BEST AU
    make_new_columns('R PEDSCPN GLASGOW COMA SCALE (AGE GREATER THAN 2 YEARS) BEST AU', flowsheet, ['GCS_V'], True)
    #R NSR GLASGOW COMA SCALE BEST EYE RESPONSE
    make_new_columns('R NSR GLASGOW COMA SCALE BEST EYE RESPONSE', flowsheet, ['GCS_E'], True)

    #R PEDSCPN GLASGOW COMA SCALE (AGE 6 MONTHS TO 2 YEARS) BEST MOT
    motor60 = [[], ['1', 1], ['2', 2], [3.0, '3'], [4.0, '4', 'P'], ['5', 5.0], [6.0, '6']]
    make_new_columns('R PEDSCPN GLASGOW COMA SCALE (AGE 6 MONTHS TO 2 YEARS) BEST MOT', flowsheet, ['GCS_M'], newcols1 = motor60)
    #R PEDSCPN GLASGOW COMA SCALE (AGE GREATER THAN 2 YEARS) BEST MO
    make_new_columns('R PEDSCPN GLASGOW COMA SCALE (AGE GREATER THAN 2 YEARS) BEST MO', flowsheet, ['GCS_M'], newcols1 = motor60)
    #R GLASGOW COMA SCALE BEST MOTOR RESPONSE
    make_new_columns('R GLASGOW COMA SCALE BEST MOTOR RESPONSE', flowsheet, ['GCS_M'], newcols1 = motor60)
    #R NSR GLASGOW COMA SCALE BEST MOTOR RESPONSE
    make_new_columns('R NSR GLASGOW COMA SCALE BEST MOTOR RESPONSE', flowsheet, ['GCS_M'], newcols1 = motor60)

    #R GLASGOW COMA SCALE BEST VERBAL RESPONSE
    make_new_columns('R GLASGOW COMA SCALE BEST VERBAL RESPONSE', flowsheet, ['GCS_V'], newcols1 = motor60)

    #R GLASGOW COMA SCALE EYE OPENING
    eye223 = [[], ['1', 1], ['2', 2, 'P'], [3.0, '3'], [4.0, '4']]
    make_new_columns('R GLASGOW COMA SCALE EYE OPENING', flowsheet, ['GCS_E'], newcols1 = eye223)

    #R PEDSCPN GLASGOW COMA SCALE (AGE 6 MONTHS TO 2 YEARS) EYE OPEN
    make_new_columns('R PEDSCPN GLASGOW COMA SCALE (AGE 6 MONTHS TO 2 YEARS) EYE OPEN', flowsheet, ['GCS_E'], newcols1 = eye223)

    #R PEDSCPN GLASGOW COMA SCALE (AGE GREATER THAN 2 YEARS) EYE OPE
    make_new_columns('R PEDSCPN GLASGOW COMA SCALE (AGE GREATER THAN 2 YEARS) EYE OPE', flowsheet, ['GCS_E'], newcols1 = eye223)

    #R GLASGOW COMA SCALE SCORE
    make_new_columns('R GLASGOW COMA SCALE SCORE', flowsheet, ['GCS'], True)

    #R NSR GLASGOW COMA SCALE SCORE
    make_new_columns('R NSR GLASGOW COMA SCALE SCORE', flowsheet, ['GCS'], True)
    #R PEDSCPN GLASGOW COMA SCALE SCORE (AGE 6 MONTHS TO 2 YEARS)
    make_new_columns('R PEDSCPN GLASGOW COMA SCALE SCORE (AGE 6 MONTHS TO 2 YEARS)', flowsheet, ['GCS'], True)
    #R PEDSCPN GLASGOW COMA SCALE SCORE (AGE GREATER THAN 2 YEARS)
    make_new_columns('R PEDSCPN GLASGOW COMA SCALE SCORE (AGE GREATER THAN 2 YEARS)', flowsheet, ['GCS'], True)


    #Make GCS column sum of other columns where not filled
    GCS_nas = flowsheet['GCS'].isna()
    flowsheet.loc[GCS_nas, 'GCS'] = flowsheet.loc[GCS_nas, 'GCS_E'] + flowsheet.loc[GCS_nas, 'GCS_M'] + flowsheet.loc[GCS_nas, 'GCS_V']

    fill_missing('GCS', flowsheet, 'GCS_missing')
    fill_missing('GCS_E', flowsheet, 'GCS_E_missing')
    fill_missing('GCS_M', flowsheet, 'GCS_M_missing')
    fill_missing('GCS_V', flowsheet, 'GCS_V_missing')
    fill_missing('AVPU', flowsheet, 'AVPU_missing')
    interpolate_cols('GCS', flowsheet, 'pad', how_far = 90, limit_direction = 'forward', fill = 15)
    interpolate_cols('GCS_E', flowsheet, 'pad', how_far = 90, limit_direction = 'forward', fill = 4)
    interpolate_cols('GCS_M', flowsheet, 'pad', how_far = 90, limit_direction = 'forward', fill = 6)
    interpolate_cols('GCS_V', flowsheet, 'pad', how_far = 90, limit_direction = 'forward', fill = 5)
    interpolate_cols('AVPU', flowsheet, 'pad', how_far = 90, limit_direction = 'forward', fill = 3)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("GCS now interpolated at", current_time)

    #CRT
    print_all(flowsheet)
    get_relevant_cols(flowsheet, 'Cap|Refill|CRT|Time')
    #R CAPILLARY REFILL: GENERAL
    crt98 = [[], [], ['Less than/equal to 2 seconds (All extremities)'], [], ['Greater than 2 seconds (All extremities)']]
    make_new_columns('R CAPILLARY REFILL: GENERAL', flowsheet, ['CRT'], newcols1 = crt98)

    #R CAPILLARY REFILL: ICU
    crt99 = [[], [], ['<2', 'Brisk'], ['2 - 3'], ['3 - 5', 'Sluggish'], ['>5']]
    make_new_columns('R CAPILLARY REFILL: ICU', flowsheet, ['CRT'], newcols1 = crt99)

    #R CAPILLARY REFILL: SECONDS
    crt100 = [[], [], ['Less than 2 seconds'], ['3 seconds'], ['4 seconds'], ['5 seconds'], ['Greater than 6 seconds']]
    make_new_columns('R CAPILLARY REFILL: SECONDS', flowsheet, ['CRT'], newcols1 = crt100)

    #R PVS CAPILLARY REFILL LLE
    crt357 = [[], [1, '2'], [2, '2'], [3, '3'], [4, '4', '4+']]
    make_new_columns('R PVS CAPILLARY REFILL LLE', flowsheet, ['CRT'], newcols1 = crt357)

    #R PVS CAPILLARY REFILL RLE
    make_new_columns('R PVS CAPILLARY REFILL RLE', flowsheet, ['CRT'], newcols1 = crt357)

    fill_missing('CRT', flowsheet, 'CRT_missing')
    interpolate_cols('CRT', flowsheet, 'linear', how_far = 90, limit_direction = 'both', fill = 2)






    ####Sats
    print_all(flowsheet)
    get_relevant_cols(flowsheet, 'sat|spo2|oxygen|\%|percent')
    # R EVENT SPO2
    make_new_columns('R EVENT SPO2', flowsheet, ['SpO2'], True)
    #R PERF NORMALISED SAO2_%
    make_new_columns('R PERF NORMALISED SAO2_%', flowsheet, ['SpO2'], True)
    #R GOSH CV SATURATION NEW_%
    make_new_columns('R GOSH CV SATURATION NEW_%', flowsheet, ['ScvO2'], True)
    #PERF PRIME BG POC SO2_%
    make_new_columns('PERF PRIME BG POC SO2_%', flowsheet, ['SpO2'], True)
    #ANAESTHESIA PULSE OXIMETRY_%
    make_new_columns('ANAESTHESIA PULSE OXIMETRY_%', flowsheet, ['SpO2'], True)
    #PULSE OXIMETRY_%
    make_new_columns('PULSE OXIMETRY_%', flowsheet, ['SpO2'], True)
    #R GOSH IP PULSE OXIMETRY ARTERIAL SPO2L_%
    make_new_columns('R GOSH IP PULSE OXIMETRY ARTERIAL SPO2L_%', flowsheet, ['SpO2'], True)
    #R GOSH IP PULSE OXIMETR ARTERIAL SPO2R_%
    make_new_columns('R GOSH IP PULSE OXIMETR ARTERIAL SPO2R_%', flowsheet, ['SpO2'], True)
    #R PULSE OXIMETRY PRE-DUCTAL_%
    make_new_columns('R PULSE OXIMETRY PRE-DUCTAL_%', flowsheet, ['SpO2'], True)
    #R CV SYSTEMIC ARTERIAL O2 SATURATION_%
    make_new_columns('R CV SYSTEMIC ARTERIAL O2 SATURATION_%', flowsheet, ['SpO2'], True)

    fill_missing('SpO2', flowsheet, 'SpO2_missing')
    fill_missing('ScvO2', flowsheet, 'ScvO2_missing')
    interpolate_cols('SpO2', flowsheet, 'linear', how_far = 90, limit_direction = 'both', fill = 95)
    interpolate_cols('ScvO2', flowsheet, 'linear', how_far = 90, limit_direction = 'both', fill = fill_median)



    #Height
    get_relevant_cols(flowsheet, 'height|ht|metres|cm')
    #HEIGHT_centimetres
    make_new_columns('HEIGHT_centimetres', flowsheet, ['Height(cm)'], True)
    #R GOSH IP HEIGHT FOR WEIGHT
    make_new_columns('R GOSH IP HEIGHT FOR WEIGHT', flowsheet, ['Height(cm)'], True)

    apply_wt_ht('Height(cm)', flowsheet, 'interpolated_ht_m', scaling = 0.01)


    #Input/ output
    #Input
    get_relevant_cols(flowsheet, 'input|intake|fluid|output|nappy')
    get_relevant_cols(flowsheet, 'loss')
    flowsheet['input'] = 0
    flowsheet['output'] = 0

    #R DRY NAPPY WEIGHT_g 182
    #R NAPPY WITH STOOL WEIGHT_g
    conv_2cols('R DRY NAPPY WEIGHT_g', 'R NAPPY WITH STOOL WEIGHT_g', flowsheet, 'Nappy_output', bminusa)
    #R NAPPY WEIGHT WITH URINE_g
    conv_2cols('R DRY NAPPY WEIGHT_g', 'R NAPPY WEIGHT WITH URINE_g', flowsheet, 'Nappy_output', bminusa)
    #401 For R NAPPY WEIGHT URINE AND STOOL_g
    conv_2cols('R DRY NAPPY WEIGHT_g', 'R NAPPY WEIGHT URINE AND STOOL_g', flowsheet, 'Nappy_output', bminusa)
    sum_inout('Nappy_output', flowsheet, 'output')

    #R GOSH IP DIET TOTAL FLUID
    sum_inout('R GOSH IP DIET TOTAL FLUID', flowsheet, 'input')

    #R ONC APHERESIS FLUID BALANCE
    sum_inout('R ONC APHERESIS FLUID BALANCE', flowsheet, 'input')

    #INTRAVENOUS INTAKE_mL
    sum_inout('INTRAVENOUS INTAKE_mL', flowsheet, 'input')

    #R AN NG/OG TUBE OUTPUT_mL
    sum_inout('R AN NG/OG TUBE OUTPUT_mL', flowsheet, 'output')

    #OTHER OUTPUT_mL
    sum_inout('OTHER OUTPUT_mL', flowsheet, 'output')

    #R CHEST TUBE OUTPUT_mL
    sum_inout('R CHEST TUBE OUTPUT_mL', flowsheet, 'output')

    #R GOSH LDA WOUND OUTPUT VOLUME_mL
    sum_inout('R GOSH LDA WOUND OUTPUT VOLUME_mL', flowsheet, 'output')

    #R DRAIN OUTPUT_mL
    sum_inout('R DRAIN OUTPUT_mL', flowsheet, 'output')

    #R GOSH URINE DRAIN OUTPUT_mL
    sum_inout('R GOSH URINE DRAIN OUTPUT_mL', flowsheet, 'output')

    #R OSTOMY OUTPUT TYPE_mL
    sum_inout('R OSTOMY OUTPUT TYPE_mL', flowsheet, 'output')

    #R TUBE OUTPUT_mL
    sum_inout('R TUBE OUTPUT_mL', flowsheet, 'output')

    #R URINE OUTPUT_m
    sum_inout('R URINE OUTPUT_mL', flowsheet, 'output')

    #R CRRT PROGRAMMED FLUID LOSS_mL/hr - need to check if this is every min or hour
    sum_inout('R CRRT PROGRAMMED FLUID LOSS_mL/hr', flowsheet, 'output')

    #R GOSH IP CRRT FLUID REMOVAL_mL
    sum_inout('R GOSH IP CRRT FLUID REMOVAL_mL', flowsheet, 'output')

    #URINE OUTPUT_mL
    sum_inout('URINE OUTPUT_mL', flowsheet, 'output')

    #R GOSH IP I/O ORAL INTAKE VOL ML_ml
    sum_inout('R GOSH IP I/O ORAL INTAKE VOL ML_ml', flowsheet, 'input')

    #(718, 'R GOSH IP CONT FEED TOTAL_mL')
    #(719, 'R GOSH IP CONTINUED FEED TOTAL VOLUME_mL')
    """(720, 'R GOSH IP IV FLUID MAINTENANCE ACCUMULATIVE VOLUME_mL')
    (721, 'R GOSH IP TPN ALL IN ONE INFUSION ACCUMULATIVE VOLUME_mL')
    (722, 'R GOSH IP TPN AQUEOUS INFUSION ACCUMULATIVE VOLUME_mL')
    (723, 'R GOSH IP TPN LIPIDS INFUSION ACCUMULATIVE VOLUME_mL')
    (745, 'R GOSH TOTAL VOLUME TPN_mL')
    (775, 'R CATHETER URINE RETURNED')
    (773, 'R URINE AMOUNT')
    (791, 'G TPN ALL IN ONE INFUSION VOLUME_mL')
    (792, 'R CHEMO IV VOLUME')
    (793, 'R GOSH IP I/O ORAL INTAKE VOL ML_ml')
    (794, 'R GOSH IP IO CONT FEED VOL_mL')
    (796, 'R IP BLOOD ADMINISTRATION TOTAL VOLUME_mL')
    (797, 'R MAINTENANCE IV BOLUS VOLUME')
    (798, 'R MAINTENANCE IV VOLUME_mL')
    """
    #Will need to double check about duplicates here
    #May need to normalise by bodyweight?

    #ECMO moved here as ECMO = maximal inotropy
    #ECMO
    get_relevant_cols(flowsheet, 'ECMO')
    make_new_columns('R ECMO  FLOW PROBE', flowsheet, ['ECMO'], new_cols1 = make_allnonNa(flowsheet, 147, 2, 2))
    make_new_columns('R GOSH LDA ECMO LINE STATUS', flowsheet, ['ECMO'], new_cols1 = make_allnonNa(flowsheet, 185, 2, 2))
    make_new_columns('R ECMO PUMP FLOW (L/MIN)_L/min', flowsheet, ['ECMO'], new_cols1 = make_allnonNa(flowsheet, 186, 2, 2))
    make_new_columns('R ECMO CLAMPS CHECK', flowsheet, ['ECMO'], new_cols1 = make_allnonNa(flowsheet, 188, 2, 2))
    make_new_columns('GOSH ICU ECMO HCT', flowsheet, ['ECMO'], new_cols1 = make_allnonNa(flowsheet, 190, 2, 2))
    make_new_columns('R ECMO ON/OFF', flowsheet, ['ECMO'], new_cols1 = make_allnonNa(flowsheet, 279, 2, 2))
    make_new_columns('R ECMO HEAT EXCHANGER WATER LEVEL', flowsheet, ['ECMO'], new_cols1 = make_allnonNa(flowsheet, 280, 2, 2))
    make_new_columns('R HOURS ON ECMO', flowsheet, ['ECMO'], new_cols1 = make_allnonNa(flowsheet, 295, 2, 2))
    make_new_columns('R GOSH ECMO NUMBER CIRCUITS', flowsheet, ['ECMO'], new_cols1 = make_allnonNa(flowsheet, 423, 2, 2))
    fill_missing('ECMO', flowsheet, 'ECMO_missing')    

    #Other organ support
    print_all(flowsheet)
    """
    (803, 'R GOSH ADRENALINE HOURLY VOLUME_mL')
    (835, 'R GOSH MILRINONE HOURLY VOLUME_mL')
    (837, 'R GOSH NORADRENALINE HOURLY VOLUME_mL')
    (853, 'R GOSH VASOPRESSIN HOURLY VOLUME_mL')
    (815, 'R DOPAMINE VOLUME_mL')
    Adrenaline 1:1 with norad
    Milrinone not sure yet
    Vasopressin 0.4:1 norad
    Possibly just assume equivalence across milrinone range? like bottom dose = bottom of norad?
    Bottom dose of milrinone is 30/mcg/kg/hr, of norad is 20/ng/kg/min or 1.2mcg/kg/hr
    Therefore milrinone to norad conversion would be 1.2/30 or 1/25
    """
    ## Make changes to adrenaline according to locations of timings (some doses 1:10000, some 1:1000)
    flowsheet['Adj_adrenaline'] = flowsheet['R GOSH ADRENALINE HOURLY VOLUME_mL']

    #Get locations where lower dose adrenaline given
    adrenaline_lowdose = re.compile('(ADRENALINE).*(\(1:10,000\)).*')
    adrenaline_locs = [i for i, j in enumerate(medications['drug_name']) if adrenaline_lowdose.match(j) != None]

    #Work through this list and then change all the matched ones in flowsheet
    for i in adrenaline_locs:
        #Pull patient and time
        patient = medications.loc[i, 'project_id']
        start_time = medications.loc[i, 'start_datetime']
        end_time = medications.loc[i, 'end_datetime']

        #Find locations in flowsheet that match patient and time
        patient_matches = flowsheet['project_id'] == patient
        time_matches = (start_time <= flowsheet['taken_datetime']) & (flowsheet['taken_datetime'] <= end_time)
        true_matches = patient_matches & time_matches

        #Now adjust
        flowsheet.loc[true_matches, 'Adj_adrenaline'] /= 10
        

    sum_inout('Adj_adrenaline', flowsheet, 'Inotropes')
    sum_inout('Adj_adrenaline', flowsheet, 'Norad_adrenaline')
    sum_inout('R GOSH NORADRENALINE HOURLY VOLUME_mL', flowsheet, 'Inotropes')
    sum_inout('R GOSH NORADRENALINE HOURLY VOLUME_mL', flowsheet, 'Norad_adrenaline')

    def corAVP(input):
        return input*2.5
    conv_column('R GOSH VASOPRESSIN HOURLY VOLUME_mL', flowsheet, 'EquivAVP', corAVP)
    
    sum_inout('EquivAVP', flowsheet, 'Inotropes')

    def corDopamine(input):
        #Dopamine given at 40mg/ml
        return (input/100)*40
    conv_column('R DOPAMINE VOLUME_mL', flowsheet, 'EquivDopamine', corDopamine)
    sum_inout('EquivDopamine', flowsheet, 'Inotropes')

    #Can consider changing this to be /10 as that is what Michael has done
    #Milrinone is 1mg/ml - (0.1%, 1:1000) - so this should be *10 not /10
    def corMilrinone(input):
        return input*10
    conv_column('R GOSH MILRINONE HOURLY VOLUME_mL', flowsheet, 'EquivMilrinone', corMilrinone)
    sum_inout('EquivMilrinone', flowsheet, 'Inotropes')

    #Add in phenylephrine - actually don't as appears only to be used PRN - almost impossible for useful data to be got

    def aoverb(a, b):
        return a/b
    conv_2cols('Inotropes', 'interpolated_wt_kg', flowsheet, 'Inotropes_kg', aoverb)
    conv_2cols('EquivAVP', 'interpolated_wt_kg', flowsheet, 'AVP_kg', aoverb)
    conv_2cols('EquivDopamine', 'interpolated_wt_kg', flowsheet, 'Dopamine_kg', aoverb)
    conv_2cols('Norad_adrenaline', 'interpolated_wt_kg', flowsheet, 'Norad_adrenaline_kg', aoverb)
    conv_2cols('EquivMilrinone', 'interpolated_wt_kg', flowsheet, 'Milrinone_kg', aoverb)
    
    #Now in mcg/kg/min
    flowsheet['Inotropes_mcg_kg_min'] = flowsheet.Inotropes_kg/60
    flowsheet['EquivAVP_mcg_kg_min'] = flowsheet.AVP_kg/60
    flowsheet['EquivDopamine_mcg_kg_min'] = flowsheet.Dopamine_kg/60
    flowsheet['Norad_adrenaline_mcg_kg_min'] = flowsheet.Norad_adrenaline_kg/60
    flowsheet['EquivMilrinone_mcg_kg_min'] = flowsheet.Milrinone_kg/60

    #Take maximum corrected intrope to be when patients are on ECMO
    maximum_inotropy = np.max(flowsheet['Inotropes_kg'])
    on_ECMO = flowsheet['ECMO'].isna() == False
    flowsheet.loc[on_ECMO, 'Inotropes_kg'] = maximum_inotropy

    #Only filling and interpolating Inotropy, interoplating ECMO here as don't want this to affect intropy/ viceversa
    fill_missing('Inotropes_mcg_kg_min', flowsheet, 'Inotropes_missing')
    fill_missing('EquivAVP_mcg_kg_min', flowsheet, 'AVP_missing')
    fill_missing('EquivDopamine_mcg_kg_min', flowsheet, 'Dopamine_missing')
    fill_missing('Norad_adrenaline_mcg_kg_min', flowsheet, 'Norad_adrenaline_missing')
    fill_missing('EquivMilrinone_mcg_kg_min', flowsheet, 'Milrinone_missing')
    
    interpolate_cols('ECMO', flowsheet, 'pad', how_far = 90, limit_direction = 'forward', fill = 0)
    interpolate_cols('Inotropes_mcg_kg_min', flowsheet, 'linear', how_far = 90, limit_direction = 'both', fill = 0)
    interpolate_cols('EquivAVP_mcg_kg_min', flowsheet, 'linear', how_far = 90, limit_direction = 'both', fill = 0)
    interpolate_cols('EquivDopamine_mcg_kg_min', flowsheet, 'linear', how_far = 90, limit_direction = 'both', fill = 0)
    interpolate_cols('Norad_adrenaline_mcg_kg_min', flowsheet, 'linear', how_far = 90, limit_direction = 'both', fill = 0)
    interpolate_cols('EquivMilrinone_mcg_kg_min', flowsheet, 'linear', how_far = 90, limit_direction = 'both', fill = 0)

    #Presumably ECMO is also maximum ventilation so:
    flowsheet.loc[flowsheet['ECMO'] == 1, 'Ventilation'] = 3

    #Print where we've got to
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Inotropes done:", current_time)
    
    ##Now we can calculate pSOFA
    #First spo2:fiO2 ratio
    #Now make the ratio
    flowsheet['SF_ratio'] = np.NaN
    flowsheet.SF_ratio = flowsheet.SpO2/flowsheet.FiO2
    
    #Make ventilation == 1 where FiO2 > 0.23
    no_resp_support = np.where(flowsheet.Ventilation == 0)
    on_oxygen = np.where(flowsheet.FiO2 > 0.23)
    needs_resp_support = np.intersect1d(no_resp_support, on_oxygen)
    flowsheet.loc[needs_resp_support, 'Ventilation'] = 1
    
    flowsheet.to_csv('/store/DAMTP/dfs28/PICU_data/flowsheet_pSOFA_interpolated.csv.gz')

if (os.path.exists('/store/DAMTP/dfs28/PICU_data/flowsheet_nearly_interpolated.csv.gz') == False):
    flowsheet = pd.read_csv('/store/DAMTP/dfs28/PICU_data/flowsheet_pSOFA_interpolated.csv.gz', parse_dates = ['taken_datetime'])

    
    #Now make the pSOFA resp criteria
    #First small function to determine how high the score should be
    def pResp(ratio, Vent):
        if Vent <= 1:
            if ratio < 264:
                return 2
            elif ratio < 292:
                return 1
            elif ratio >=292:
                return 0
            else:
                return np.NaN
        else:
            if ratio <148:
                return 4
            elif ratio >=148:
                return 3
            else:
                return np.NaN
            
        
    #Now make the pSOFA score for resp
    flowsheet['pSOFA_resp'] = np.NaN
    
    #It seems to use too much memory to do this all at once so:
    unique_id = flowsheet.project_id.unique()
    bar = Bar('Processing pSOFA_resp', max=len(unique_id))
    for i in unique_id:
        project_locs = np.where(flowsheet.project_id == i)
        pSOFA_per_patient = flowsheet.loc[project_locs[0], :].apply(lambda x: pResp(x.SF_ratio, x.Ventilation), 1)
        flowsheet.loc[project_locs[0], 'pSOFA_resp'] = pSOFA_per_patient
        bar.next()
    bar.finish()
        
    
    

    ## Now pSOFA for inotropes
    def pInotrope(MAP, dopamine, nor_adrenaline, milrinone, age):
        #First if they are on inotropes/ vasopressors
        if nor_adrenaline > 0.1:
            return 4
        elif dopamine > 0.15: #Using the corrected values
            return 4
        elif dopamine > 0.05: #Using the corrected values
            return 3
        elif nor_adrenaline > 0:
            return 3
        elif dopamine > 0:
            return 2
        elif milrinone > 0:
            return 2 #Need to double check this with Sam
        else:
            #Get the normal value
            map_ages = np.array([-1, 1/12, 1, 2, 5, 12, 18, 25]) #Using -1 because there is a child with age 0 apparently
            map_normals = np.array([46, 55, 60, 62, 65, 67, 70, 70])
            normal_map = map_normals[map_ages < age][-1]
            
            #Now score
            if MAP < normal_map:
                return 1
            elif MAP >= normal_map:
                return 0
            else:
                return np.NaN 
            
    flowsheet['pSOFA_cardio'] = np.NaN
    
    #It seems to use too much memory to do this all at once so:
    bar = Bar('Processing pSOFA_cardio', max=len(unique_id))
    for i in unique_id:
        project_locs = np.where(flowsheet.project_id == i)
        pCardio_per_patient = flowsheet.loc[project_locs[0], :].apply(lambda x: pInotrope(x.MAP,
                                                                                          x.EquivDopamine_mcg_kg_min*100,
                                                                                          x.Norad_adrenaline_mcg_kg_min, 
                                                                                          x.EquivMilrinone_mcg_kg_min,
                                                                                          x.Age_yrs), 1)
        flowsheet.loc[project_locs[0], 'pSOFA_cardio'] = pCardio_per_patient
        bar.next()
    bar.finish()
        
    #pSOFA for Platelet count
    plt_normals = np.array([150, 100,  50,  20,   0])
    pSOFA_plt = np.nan
    
    #For Bilirubin
    normal_bili = np.array([0, 1.2, 2, 6, 12]) * 17.104
    pSOFA_bili = np.nan
    
    #For GCS
    normal_gcs = np.array([15, 13, 10, 6, 0])
    pSOFA_GCS = np.nan
    
    #For creatinine
    normal_creatinine = pd.DataFrame(np.array(([0, 0.8, 1.0, 1.2, 1.6], 
                                                [0, 0.3, 0.5, 0.8, 1.2], 
                                                [0, 0.4, 0.6, 1.1, 1.5], 
                                                [0, 0.6, 0.9, 1.6, 2.3], 
                                                [0, 0.7, 1.1, 1.8, 2.6], 
                                                [0, 1.0, 1.7, 2.9, 4.2], 
                                                [0, 1.2, 2.0, 3.5, 5]))*88.4, 
                                     index = [1/12, 1, 2, 5, 12, 18, 25])
    
    def pCreatinine(creatinine: float, age: float) -> int:
        #Find the relevant row of the index
        index_position = np.where(age < normal_creatinine.index)[0][0]
        correct_row = normal_creatinine.index[index_position]
        
        #Now return the pSOFA score relevant to that (just the correct column)
        return np.where(normal_creatinine.loc[correct_row, :] < creatinine)[0][-1]
    pSOFA_cr = np.nan
    
    bar = Bar('Processing rest of pSOFA', max=len(unique_id))                                  
    for i in unique_id:
        project_locs = np.where(flowsheet.project_id == i)
        
        #Platelets
        pPlt_per_patient = flowsheet.loc[project_locs[0], 'Plts'].map(lambda x: np.where(x >= plt_normals)[0][0])
        flowsheet.loc[project_locs[0], 'pSOFA_plt'] = pPlt_per_patient
        
        #Bili
        pBili_per_patient = flowsheet.loc[project_locs[0], 'Bili'].map(lambda x: np.where(x >= normal_bili)[0][-1])
        flowsheet.loc[project_locs[0], 'pSOFA_bili'] = pBili_per_patient
        
        #GCS
        pGCS_per_patient = flowsheet.loc[project_locs[0], 'GCS'].map(lambda x: np.where(x >= normal_gcs)[0][0])
        flowsheet.loc[project_locs[0], 'pSOFA_GCS'] = pGCS_per_patient
        
        #Creatinine
        pCr_per_patient = flowsheet.loc[project_locs[0], :].apply(lambda x: pCreatinine(x.Cr,
                                                                                          x.Age_yrs), 1)
        flowsheet.loc[project_locs[0], 'pSOFA_Cr'] = pCr_per_patient
        bar.next()
    bar.finish()

    #Adding up all the pSOFA scores has to happen after we have data for dialysis so this has moved
    
    ### Resp rate
    #get_relevant_cols(flowsheet, 'rate|RR|resp')

    #378 R GOSH VENT RATE
    #print_allunique(flowsheet, 378)

    #567 ANAESTHESIA ECG RESPIRATION RATE
    make_new_columns('ANAESTHESIA ECG RESPIRATION RATE', flowsheet, ['RR'], True)

    #566 ANAESTHESIA AIRWAY RESPIRATION RATE
    make_new_columns('ANAESTHESIA AIRWAY RESPIRATION RATE', flowsheet, ['RR'], True)

    #559 RESPIRATIONS
    make_new_columns('RESPIRATIONS', flowsheet, ['RR'], True)

    #379 For R GOSH VENT RATE OBSERVED
    fill_missing('RR', flowsheet, 'RR_missing')

    #Need to do something about normals here too
    interpolate_cols('RR', flowsheet, 'linear', 'RR', how_far = 90, limit_direction = 'both', fill = fill_median_HRBP)
    print('Saving interpolated columns')
    flowsheet.to_csv('/store/DAMTP/dfs28/PICU_data/flowsheet_nearly_interpolated.csv.gz')

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Everything up to resp rate interpolated", current_time)
else:
    print('Interpolation already nearly done')




### Ecmo and dialysis

if os.path.exists('/store/DAMTP/dfs28/PICU_data/flowsheet_with_PEWS.csv.gz') == False:
    flowsheet = pd.read_csv('/store/DAMTP/dfs28/PICU_data/flowsheet_nearly_interpolated.csv.gz', parse_dates = ['taken_datetime'])

    #Dialysis
    get_relevant_cols(flowsheet, 'CRRT|dialysis|PD|HDF')

    #231 R GOSH CRRT FILTER TYPE
    make_new_columns('R GOSH CRRT FILTER TYPE', flowsheet, ['dialysis'], new_cols1 = make_allnonNa(flowsheet, 'R GOSH CRRT FILTER TYPE', 2, 2))

    #232 R GOSH CRRT FF_%
    make_new_columns('R GOSH CRRT FF_%', flowsheet, ['dialysis'], new_cols1 = make_allnonNa(flowsheet, 'R GOSH CRRT FF_%', 2, 2))

    #576 R CRRT PROGRAMMED FLUID LOSS_mL/hr
    make_new_columns('R CRRT PROGRAMMED FLUID LOSS_mL/hr', flowsheet, ['dialysis'], new_cols1 = make_allnonNa(flowsheet, 'R CRRT PROGRAMMED FLUID LOSS_mL/hr', 2, 2))

    #577 R GOSH IP CRRT FLUID REMOVAL_mL
    make_new_columns('R GOSH IP CRRT FLUID REMOVAL_mL', flowsheet, ['dialysis'], new_cols1 = make_allnonNa(flowsheet, 'R GOSH IP CRRT FLUID REMOVAL_mL', 2, 2))

    #578 R GOSH CRRT END OF CYCLE VOL_mL
    make_new_columns('R GOSH CRRT END OF CYCLE VOL_mL', flowsheet, ['dialysis'], new_cols1 = make_allnonNa(flowsheet, 'R GOSH CRRT END OF CYCLE VOL_mL', 2, 2))

    #733 R GOSH PD TOTAL VOLUME_mL
    make_new_columns('R GOSH PD TOTAL VOLUME_mL', flowsheet, ['dialysis'], new_cols1 = make_allnonNa(flowsheet, 'R GOSH PD TOTAL VOLUME_mL', 2, 2))

    #766 R GOSH UF VOLUME HD/HDF
    make_new_columns('R GOSH UF VOLUME HD/HDF', flowsheet, ['dialysis'], new_cols1 = make_allnonNa(flowsheet, 'R GOSH UF VOLUME HD/HDF', 2, 2))

    #769 R GOSH CRRT UF
    make_new_columns('R GOSH CRRT UF', flowsheet, ['dialysis'], new_cols1 = make_allnonNa(flowsheet, 'R GOSH CRRT UF', 2, 2))
    fill_missing('dialysis', flowsheet, 'dialysis_missing')
    interpolate_cols('dialysis', flowsheet, 'pad', how_far = 90, limit_direction = 'forward', fill = 0)

    ## Fix issue with dialysis not appearing in pSOFA score - these patients should have the max socre
    flowsheet.loc[flowsheet.dialysis == 1, 'pSOFA_Cr'] = 4
    
    #Now add up pSofa scores
    flowsheet['pSOFA'] = flowsheet.pSOFA_resp + flowsheet.pSOFA_cardio + flowsheet.pSOFA_bili + flowsheet.pSOFA_plt + flowsheet.pSOFA_GCS + flowsheet.pSOFA_Cr

    ### Don't forget temperature
    get_relevant_cols(flowsheet, 'temp|deg')
    make_new_columns('R GOSH IP AXILLA TEMPERATURE', flowsheet, ['Temp'], True)
    make_new_columns('R CORE (BODY) TEMPERATURE', flowsheet, ['Temp'], True)
    make_new_columns('R AN NASOPHARYNGEAL TEMPERATURE', flowsheet, ['Temp'], True)
    make_new_columns('R GOSH IP NASOPHARYNGEAL TEMPERATURE', flowsheet, ['Temp'], True)
    make_new_columns('R AN ESOPHAGEAL TEMPERATURE', flowsheet, ['Temp'], True)
    make_new_columns('R GOSH IP OESOPHAGEAL TEMPERATURE', flowsheet, ['Temp'], True)
    make_new_columns('R AN RECTAL TEMPERATURE', flowsheet, ['Temp'], True)
    make_new_columns('R AN SKIN TEMPERATURE', flowsheet, ['Temp'], True)
    make_new_columns('TEMPERATURE', flowsheet, ['Temp'], True)
    make_new_columns('R AN TEMPERATURE (NON-SPECIFIC)', flowsheet, ['Temp'], True)
    fill_missing('Temp', flowsheet, 'Temp_missing')
    interpolate_cols('Temp', flowsheet, 'linear', how_far = 90, limit_direction = 'both', fill = fill_median)

    ### Urine output
    make_new_columns('R URINE OUTPUT_mL', flowsheet, ['Urine_output'], True)
    make_new_columns('URINE OUTPUT_mL', flowsheet, ['Urine_output'], True)
    flowsheet['Urine_output_kg'] =  flowsheet['Urine_output']/flowsheet['interpolated_wt_kg'] 
    fill_missing('Urine_output_kg', flowsheet, 'Urine_output_missing_kg')
    interpolate_cols('Urine_output_kg', flowsheet, 'linear', how_far = 90, limit_direction = 'both', fill = 0)
    
    #Use the urine output kg to interpolate urine output
    urine_na_locs = flowsheet.Urine_output.isna()
    flowsheet.loc[urine_na_locs, 'Urine_output'] = flowsheet.loc[urine_na_locs, 'Urine_output_kg']*flowsheet.loc[urine_na_locs, 'interpolated_wt_kg']

    ## Do something with PEWS and PIM score
    get_relevant_cols(flowsheet, 'PEWS')
    
    #Will have to come back to this, can use it just as PEWS in mean time
    pews = ['R GOSH PEWS BP SCORE', 'R GOSH PEWS CRT SCORE', 'R GOSH PEWS HR SCORE', 'R GOSH PEWS RESP SCORE', 'R GOSH PEWS OX SCORE', 'R GOSH PEWS SPO2 SCORE', 'R GOSH PEWS WOB SCORE']
    
    #I think that most PEWS charts make you put zero if not available anyway - assume GCS basically not there because mostly NA
    for i in pews:
        interpolate_cols(i, flowsheet, 'pad', how_far = 90, limit_direction = 'forward', fill = 0)
    flowsheet['PEWS'] = flowsheet['R GOSH PEWS BP SCORE'] + flowsheet['R GOSH PEWS WOB SCORE'] + flowsheet['R GOSH PEWS SPO2 SCORE'] + flowsheet['R GOSH PEWS RESP SCORE'] + flowsheet['R GOSH PEWS OX SCORE'] + flowsheet['R GOSH PEWS CRT SCORE'] + flowsheet['R GOSH PEWS HR SCORE']


    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Temp and PEWS done:", current_time)

    #Saving PEWS
    flowsheet.to_csv('/store/DAMTP/dfs28/PICU_data/flowsheet_with_PEWS.csv.gz')

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Temp and PEWS saved:", current_time)

#Now PIM score
""" Probably come back to this?
get_relevant_cols(flowsheet, 'PIM')
for i in unique_patients:
PIM2 val = (0.01395 * (absolute (SBP-120))) + (3.0791 * Pupils) + (0.2888 * (100 * FiO2/PaO2)) + 
(0.1040 * (absolute Base Excess)) + (1.3352 * MechVent) − (0.9282 * Elective) − (1.0244 * Recovery) + (0.7507 * Bypass) + (1.6829 * HRdiag) − (1.5770 * LRdiag) − 4.8841. 
Ref is downloaded Egyptian paper
The PIM2 risk of death = exp PIM2 val/(1 + expPIM2 val)
"""

if os.path.exists('/store/DAMTP/dfs28/PICU_data/flowsheet_final_output.csv.gz') == False:
    flowsheet = pd.read_csv('/store/DAMTP/dfs28/PICU_data/flowsheet_with_PEWS.csv.gz', parse_dates = ['taken_datetime'])

    #Now bin all bits that are not in the PICU admission - should preserve pre-existing bloods etc
    #Frames for start and end dates
    episode_times = pd.DataFrame([], columns = ['project_id', 'episode', 'start_dates', 'end_dates'])
    new_dtypes = {"project_id": object, "episode": int, 'start_dates': 'datetime64', 'end_dates':'datetime64'}
    episode_times = episode_times.astype(new_dtypes)
    start_dates = list()
    end_dates = list()
    #start_dates = pd.Series([], dtype = 'datetime64[ns]')
    #end_dates = pd.Series([], dtype = 'datetime64[ns]')

    #Plan to split up by episode however it appears that there is only one episode per patient in this case
    #flowsheet['Episode'] = 0

    #Fill them
    PICU_rows = stays['ward_code'].isin(['PICU', 'NICU', 'CICU', 'OTTER RECOV', 'TURTL RECOV'])
    unique_patients = episodes['project_id'].unique()

    for i in tqdm(unique_patients):
        pt_locs = stays['project_id'] == i

        #Get the rows which are PICU/NICU/CICU rows
        PICU_pt_locs = PICU_rows & pt_locs
        start_times = stays.loc[PICU_pt_locs, 'start_datetime']
        end_times = stays.loc[PICU_pt_locs, 'end_datetime']

        #Mark 2 different episodes if gap in stays
        if len(start_times) > 1:
            
            #Get gaps by finding difference between beginnings and ends
            middle_starts = start_times[1:]
            middle_ends = end_times[:-1]
            middle_ends.index = middle_starts.index
            gaps = middle_starts - middle_ends
            
            #Only consider separate admissions as split by a day - could just be going elsewhere for a procedure
            big_gaps = gaps > np.timedelta64(1, 'D')
            #if any(gaps > np.timedelta64(6, 'h')):
            #    print(gaps)
            #    print(i)
            #I noted issue with duplicate patients on different encounters - will need to fix

            if any(big_gaps):
                when_gap = np.where(big_gaps)
                gapped_pts.append(i)
                print(i, when_gap)

                #Put the start and end of it on so can do before and after
                start_gap_end = list([0])
                start_gap_end.append([i for i in when_gap.ndenumerate()])
                start_gap_end.append(int(len(big_gaps)))
                
                #Plan to work through the gaps if this occurs - none in this sheet
        
            else:
                #Take the start of the first admission as start and end of last as end
                #Looks like there are some places where single patients (e.g 1498 1499) have some sort of overlap but not sure why - have to leave this for now
                start_dates.append(start_times[start_times.index[0]])
                end_dates.append(end_times[end_times.index[-1]])
                #start_dates = start_dates.append(pd.Series(np.datetime64(start_times[start_times.index[0]]), dtype = 'datetime64[ns]'))
                #end_dates = end_dates.append(pd.Series(end_times[-1:], dtype = 'datetime64[ns]'))
        else:
            start_dates.append(start_times[start_times.index[0]])
            end_dates.append(end_times[end_times.index[0]])
            #start_dates = start_dates.append(pd.Series(start_times, dtype = 'datetime64[ns]'))
            #end_dates = end_dates.append(pd.Series(end_times, dtype = 'datetime64[ns]'))

    #Now make a new series to merge to which contains all of the possible time points
    all_times = pd.DataFrame(pd.Series([], dtype = 'datetime64[ns]'), columns =['taken_datetime'])
    all_times = pd.concat([all_times, pd.DataFrame([], columns = ['project_id'])])
    for i in tqdm(range(len(unique_patients))):
        times = pd.date_range(start = start_dates[i], end = end_dates[i], freq = 'min')
        patient = pd.Series([unique_patients[i]]*len(times))
        times_df = pd.DataFrame({'taken_datetime': times, 'project_id': patient})
        all_times = pd.concat([all_times, times_df])

    #Now remove all extraneous time points
    #Can do this with an if a not in b remove? then we don't have to mess around with the index
    all_times_merge_cols = ['taken_datetime', 'project_id']
    flowsheet_merge = flowsheet.loc[:, all_times_merge_cols].reset_index().merge(all_times, how = 'inner', on = all_times_merge_cols).set_index('index')
    
    #Merge back on the hospital numbers
    hospital_numbers_df = flowsheet.loc[:, 
                                        ['project_id', 'encounter_key']
                                        ].merge(
                                            demographics.loc[:, ['project_id', 'hospital_no']],
                                            on = ['project_id'],
                                            how = 'left')
    flowsheet.loc[hospital_numbers_df.index, 'hospital_no'] = hospital_numbers_df.hospital_no
    flowsheet = flowsheet.loc[flowsheet_merge.index, :]

    ### Save the data so that can run childsds in R
    flowsheet.to_csv('/store/DAMTP/dfs28/PICU_data/flowsheet_final_output_pSOFA.csv.gz')

### Now bin all data that isn't within the PICU admission time periods


### Now only print columns we are interested in
cols = ['project_id', 'encounter_key', 'taken_datetime', 'R GOSH IP PCCMDS AIRWAY/VENTILATION', 'R GOSH PCCMDS ANALGESIA/SEDATION', 'R GOSH IP PCCMDS BASIC', 
        'R GOSH IP PCCMDS CARDIOVASCULAR', 'R GOSH PCCMDS ISOLATION', 'R GOSH IP PCCMDS METABOLIC', 'R GOSH IP PCCMDS NEUROLOGICAL', 'R GOSH IP PCCMDS OTHER', 
        'R GOSH IP PCCMDS RENAL', 'R GOSH ICU PCCMDS CHECK', 'R GOSH IP GLAMORGAN PERFUSION', 'R GOSH IP GLAMORGAN PYREXIA', 'R GOSH PEWS ESCALATION',
        'R GOSH PEWS SCORE', 'R IP PEWS SCORE', 'R PIM PAO2 PRE ICU ADMISSION KPA_kPa', 'R PIM BASE EXCESS ART/CAP PRE ICU ADMISSION_mmol/L', 'R PIM PRE ICU CONTACT DATE',
        'R PIM PRE ICU CONTACT LOCATION', 'R PIM PRE ICU CONTACT TIME', 'R PIM PRE ICU ADMISSION', 'R PIM FIO2 PRE ICU ADMISSION_%', 'R PIM LACTATE PRE ICU ADMISSION_mmol/L',
        'R PIM PRE ICU MECHANICAL VENTILATION', 'R PIM PUPILS PRE-ICU', 'R PIM SYSTOLIC BLOOD PRESSURE PRE ICU ADMISSION_mmHg', 'R GOSH IP GLAMORGAN TOTAL SCORE', 
        'R GOSH IP ICU GLUCOSE MG/KG/HR_mg/kg/hr', 'R PERF TOTAL HAEMOGLOBIN (HB)', 'R CRRT INTAKE TO BE REMOVED_mL', 'R GOSH IP HD TOTAL LITRES OF DIALYSATE_L',
        'R GOSH EPIDURAL LEVO RUNNING VOL_mL', 'R GOSH LEVO EPIDURAL DO NOT USE_mL', 'R GOSH LEVO/MORPHINE EPIDURAL RUNNING VOL_mL', 'R GOSH ICU WITHDRAWAL TOTAL SCORE', 
        'R GOSH IP WITHDRAWAL TOTAL SCORE', 'R GOSH ACD-A TOTAL VOLUME_mL', 'R GOSH ADENOSINE TOTAL VOLUME_mL', 'R GOSH ADRENALINE TOTAL VOLUME_mL', 'R GOSH ALTEPLASE TOTAL VOLUME_mL', 
        'R GOSH AMINOPHYLLINE TOTAL VOLUME_mL', 'R GOSH AMIODARONE TOTAL VOLUME_mL', 'R GOSH ARGININE TOTAL VOLUME_mL', 'R GOSH CA CHLOR14.7 CVVH TOTAL VOLUME_mL', 
        'R GOSH CA GLUC10 CVVH TOTAL VOLUME_mL', 'R GOSH CALCIUM CHLOR TOTAL VOLUME_mL', 'R GOSH CLONIDINE TOTAL VOLUME_mL', 'R GOSH CRT HEPARIN TOTAL VOL_mL', 
        'R GOSH DEXMEDETOMIDINE TOTAL VOLUME_mL', 'R GOSH DINOPROSTONE TOTAL VOLUME_mL', 'R GOSH DISODIUM PHOSPHATE 21.49% TOTAL VOLUME_mL', 'R GOSH DOPAMINE TOTAL VOLUME_mL', 
        'R GOSH EPOPROSTENOL TOTAL VOLUME_mL', 'R GOSH ESMOLOL TOTAL VOLUME_mL', 'R GOSH FENTANYL TOTAL VOLUME_mL', 'R GOSH FUROSEMIDE TOTAL VOLUME_mL', 
        'R GOSH GLYCERYL TRINITRATE TOTAL VOLUME_mL', 'R GOSH HEPARIN - ART FLUSH NACL 0.9% TOTAL VOLUME_mL', 'R GOSH HEPARIN FLUSH TOTAL VOLUME_mL', 
        'R GOSH HEPARIN LA FLUSH TOTAL VOLUME_mL', 'R GOSH HEPARIN TOTAL VOLUME_mL', 'R GOSH HEPARIN UAC FLUSH TOTAL VOLUME_mL', 'R GOSH HYDRALAZINE TOTAL VOLUME_mL', 
        'R GOSH INSULIN TOTAL VOLUME_mL', 'R GOSH IP CONT FEED TOTAL_mL', 'R GOSH IP CONTINUED FEED TOTAL VOLUME_mL', 'R GOSH IP IV FLUID MAINTENANCE ACCUMULATIVE VOLUME_mL', 
        'R GOSH IP TPN ALL IN ONE INFUSION ACCUMULATIVE VOLUME_mL', 'R GOSH IP TPN AQUEOUS INFUSION ACCUMULATIVE VOLUME_mL', 'R GOSH IP TPN LIPIDS INFUSION ACCUMULATIVE VOLUME_mL', 
        'R GOSH KETAMINE TOTAL VOLUME_mL', 'R GOSH LABETALOL TOTAL VOLUME_mL', 'R GOSH MAGNESIUM TOTAL VOLUME_mL', 'R GOSH MIDAZOLAM TOTAL VOLUME_mL', 'R GOSH MILRINONE TOTAL VOLUME_mL', 
        'R GOSH MORPHINE SULPHATE TOTAL VOLUME_mL', 'R GOSH NORADRENALINE TOTAL VOLUME_mL', 'R GOSH OCTREOTIDE TOTAL VOLUME_mL', 'R GOSH PD TOTAL VOLUME_mL', 
        'R GOSH PENTOXIFYLLINE TOTAL VOLUME_mL', 'R GOSH PROPOFOL TOTAL VOLUME_mL', 'R GOSH REMIFENTANIL TOTAL VOLUME_mL', 'R GOSH ROCURONIUM TOTAL VOLUME_mL', 
        'R GOSH SALBUTAMOL TOTAL VOLUME_mL', 'R GOSH SODIUM BENZOATE TOTAL VOLUME_mL', 'R GOSH SODIUM BICARB 8.4% TOTAL VOL_mL', 'R GOSH SODIUM BICARBONATE TOTAL VOLUME_mL', 
        'R GOSH SODIUM NITROPRUSSIDE TOTAL VOLUME_mL', 'R GOSH SODIUM PHENYLBUTYRATE STREPTOKINASE TOTAL VOLUME_mL', 'R GOSH TACROLIMUS TOTAL VOLUME_mL', 'R GOSH THIOPENTAL TOTAL VOLUME_mL', 
        'R GOSH TOTAL VOLUME TPN_mL', 'R GOSH TRANEXAMIC ACID TOTAL VOLUME_mL', 'R GOSH TROMETAMOL TOTAL VOLUME_mL', 'R GOSH VASOPRESSIN TOTAL VOLUME_mL', 'R GOSH VECURONIUM TOTAL VOLUME_mL', 
        'R PCA FENTANYL (20 MCG/ML) HOURLY VOLUME_mL', 'R GOSH PCA MORPHINE/KET TOTAL VOLUME HOURLY_mL', 'R GOSH PCA NON FORMULARY VOLUME HOURLY_mL', 
        'R PCA FENTANYL/KETAMINE TOTAL VOLUME HOURLY (ML)_mcg', 'R PCA MORPHINE (1 MG/ML) TOTAL VOLUME HOURLY_mL', 'R PCA FENTANY (20 MCG/ML) CUMULATIVE VOLUME_mL', 
        'R GOSH PCA MORPHINE/KET TOTAL VOLUME RUNNING_mL', 'R PCA FENTANYL AND KETAMINE TOTAL VOLUME RUNNING (ML)_mL', 'R PCA MORPHINE (1 MG/ML) TOTAL VOLUME RUNNING_mL', 
        'R GOSH PEWS WOB SCORE', 'R GOSH PEWS CRT SCORE', 'G GOSH IP PEWS GLASGOW COMA SCALE', 'R GOSH PEWS HR SCORE', 'R GOSH PEWS OX SCORE', 'R GOSH PEWS ESCALATION', 
        'R GOSH PEWS SCORE', 'R IP PEWS SCORE', 'R GOSH PEWS RESP SCORE', 'R GOSH PEWS SPO2 SCORE', 'R GOSH PEWS WOB SCORE']

print('Using only columns that I need')

#Tidy up columns
new_cols = [i for i in flowsheet.columns[870:]]
new_cols = np.union1d(new_cols, cols)
flowsheet_output = flowsheet.loc[:, new_cols]
flowsheet_sample = flowsheet_output.iloc[1:500000, :]

print('Saving output')
flowsheet_sample.to_csv('/store/DAMTP/dfs28/PICU_data/flowsheet_sample_final_pSOFA.csv.gz')
flowsheet_output.to_csv('/store/DAMTP/dfs28/PICU_data/flowsheet_output_touse_pSOFA.csv.gz')
print('Done!')