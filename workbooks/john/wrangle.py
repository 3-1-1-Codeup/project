import pandas as pd
import numpy as np
import os

from pydataset import data
from datetime import date 
from scipy import stats

# turn off pink warning boxes
import warnings
warnings.filterwarnings("ignore")

import sklearn

from sklearn.model_selection import train_test_split

# Train/Split the data~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def split(df, stratify_by= None):
    """
    Crude train, validate, test split
    To stratify, send in a column name
    """
    if stratify_by == None:
        train, test = train_test_split(df, test_size=.2, random_state=319)
        train, validate = train_test_split(train, test_size=.3, random_state=319)
    else:
        train, test = train_test_split(df, test_size=.2, random_state=319, stratify=df[stratify_by])
        train, validate = train_test_split(train, test_size=.3, random_state=319, stratify=train[stratify_by])
    return train, validate, test

# Create X_train, y_train, etc...~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def seperate_y(train, validate, test):
    '''
    This function will take the train, validate, and test dataframes and seperate the target variable into its
    own panda series
    '''
    
    X_train = train.drop(columns=['logerror'])
    y_train = train.logerror
    X_validate = validate.drop(columns=['logerror'])
    y_validate = validate.logerror
    X_test = test.drop(columns=['logerror'])
    y_test = test.logerror
    return X_train, y_train, X_validate, y_validate, X_test, y_test

# Scale the data~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def scale_data(X_train, X_validate, X_test):
    '''
    This function will scale numeric data using Min Max transform after 
    it has already been split into train, validate, and test.
    '''
    
    
    obj_col = ['fips', 'yearbuilt', 'airconditioningdesc', 'heatingorsystemdesc', 'bath_bed_ratio']
    num_train = X_train.drop(columns = obj_col)
    num_validate = X_validate.drop(columns = obj_col)
    num_test = X_test.drop(columns = obj_col)
    
    
    # Make the thing
    scaler = sklearn.preprocessing.MinMaxScaler()
    
   
    # we only .fit on the training data
    scaler.fit(num_train)
    train_scaled = scaler.transform(num_train)
    validate_scaled = scaler.transform(num_validate)
    test_scaled = scaler.transform(num_test)
    
    # turn the numpy arrays into dataframes
    train_scaled = pd.DataFrame(train_scaled, columns=num_train.columns)
    validate_scaled = pd.DataFrame(validate_scaled, columns=num_train.columns)
    test_scaled = pd.DataFrame(test_scaled, columns=num_train.columns)
    
    
    return train_scaled, validate_scaled, test_scaled

# Combo Train & Scale Function~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def split_seperate_scale(df, stratify_by= None):
    '''
    This function will take in a dataframe
    seperate the dataframe into train, validate, and test dataframes
    seperate the target variable from train, validate and test
    then it will scale the numeric variables in train, validate, and test
    finally it will return all dataframes individually
    '''
    
    # split data into train, validate, test
    train, validate, test = split(df, stratify_by= None)
    
     # seperate target variable
    X_train, y_train, X_validate, y_validate, X_test, y_test = seperate_y(train, validate, test)
    
    
    # scale numeric variable
    train_scaled, validate_scaled, test_scaled = scale_data(X_train, X_validate, X_test)
    
    return train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test, train_scaled, validate_scaled, test_scaled

