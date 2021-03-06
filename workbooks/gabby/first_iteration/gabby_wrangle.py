import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def get_311_data():
    '''
    This function uses pandas read .csv to read in the downloaded .csv 
    from: https://data.sanantonio.gov/dataset/service-calls/resource/20eb6d22-7eac-425a-85c1-fdb365fd3cd7
    after the .csv is read in, it returns it as a data frame.
    '''
    df= pd.read_csv('service_calls.csv')
    return df

#-----------------------------------------------------------------------------

# Set the index
def drop_and_index(df):
    """
    This function will take in one positional argurment:
    1.  311 df
    This function will perform the following operations to the df:
    1.  Drop category, drop report starting date, and drop report
    ending date
    2.  Set CASEID as the index
    """
    # Drop category, report starting date, and report ending date
    df.drop(columns=['Category', 
                 'Report Starting Date', 
                 'Report Ending Date' ], inplace=True)
    # Set index to case id
    df.set_index('CASEID')
    return df

#-----------------------------------------------------------------------------

# Handle null values
def handle_nulls(df):
    '''This funciton takes in data frame
    drops nulls for specified features
    replaces nulls with "Unknown" for specific feature
    removes observations where City Council is the Department'''
    # drop null values
    df.dropna(subset = ['SLA_Date', 'XCOORD', 'YCOORD'], inplace = True)
    # replace null values
    df.fillna({'Dept': 'Unknown'}, inplace = True)
    # remove city council department
    df = df[df.Dept != 'City Council']
    # return df
    return df

#-----------------------------------------------------------------------------

# Create delay columns
def create_delay_columns(df):
    '''This funciton takes in the dataframe
    reformats specified columns into datetime format
    create 2 columns to see time a case was open 
    and the time it was given for a resolution to be found
    create another feature for how long it took 
        compare to due date for a resolution to be found
    bin how long it took compared to due date
    fill nulls with "Still Open"
    return df
    '''
    # make sure the open and closed date columns are formatted in datetime format
    df['OPENEDDATETIME'] = pd.to_datetime(df['OPENEDDATETIME'])
    df['CLOSEDDATETIME'] = pd.to_datetime(df['CLOSEDDATETIME'])
    df['SLA_Date'] = pd.to_datetime(df['SLA_Date'])
    # create new number of days open feature
    df['days_open'] = df['CLOSEDDATETIME'] - df['OPENEDDATETIME']
    # Create new column to hold days before needed resolution
    df['resolution_days_due'] = df['SLA_Date'] - df['OPENEDDATETIME']
    # Convert to string format insted of timedelta
    df['days_open'] = df.days_open // pd.Timedelta('1d')
    # Convert to string format insted of timedelta
    df['resolution_days_due'] = df.resolution_days_due // pd.Timedelta('1d')
    # create new feature to show how long it took to resolve compared to resolution due date
    df['days_before_or_after_due'] = df.resolution_days_due - df.days_open
        # postitive means before days before due data and negative means number of days after due
    # bin how long it took compare to due date to get level of delay
    df['level_of_delay'] = pd.cut(df.days_before_or_after_due, 
                                bins = [-700,-500,-300,-100,0,100,300,500],
                                labels = ['Extremely Late Response', 'Very Late Response', 
                                          'Late Response', "On Time Response", "Early Response", 
                                          'Very Early Response', 'Extremely Early Response'])
    # add a new category for level of delay
    df["level_of_delay"] = df['level_of_delay'].cat.add_categories('Still Open')
    # replace nulls in level of delay with the new "Still Open" category
    df["level_of_delay"].fillna("Still Open", inplace=True)
    # return new df
    return df

#-----------------------------------------------------------------------------

def clean_reason(df):
    '''
    This function will take in the service call df and replace the content of REASONNAME column with condensed names
    '''
    # replace with waste
    df['REASONNAME'] = df['REASONNAME'].replace(['Waste Collection', 'Solid Waste', 'Brush'], 'waste')
    # replace with code
    df['REASONNAME'] = df['REASONNAME'].replace(['Code Enforcement', 'Code Enforcement (Internal)', 'Code Enforcement (IntExp)'], 'code')
    # replace with
    df['REASONNAME'] = df['REASONNAME'].replace(['Field Operations', 'Vector'], 'field')
    # replace with
    df['REASONNAME'] = df['REASONNAME'].replace('Miscellaneous', 'misc')
    # replace with
    df['REASONNAME'] = df['REASONNAME'].replace(['Traffic Operations', 'Traffic Engineering Design', 'Traffic Issue Investigation'], 'traffic')
    # replace with
    df['REASONNAME'] = df['REASONNAME'].replace(['Streets', 'Signals', 'Signs and Markings'], 'streets')
    # replace with
    df['REASONNAME'] = df['REASONNAME'].replace('Trades', 'trades')
    # replace with
    df['REASONNAME'] = df['REASONNAME'].replace(['Stormwater', 'Storm Water', 'Storm Water Engineering'], 'storm')
    # replace with
    df['REASONNAME'] = df['REASONNAME'].replace(['Small Business', 'Food Establishments', 'Shops (Internal)', 'Shops'], 'business')
    # replace with
    df['REASONNAME'] = df['REASONNAME'].replace('Workforce Development', 'workforce')
    # replace with
    df['REASONNAME'] = df['REASONNAME'].replace(['Customer Service', '311 Call Center', 'Director\'s Office Horizontal'], 'customer_service')
    # replace with
    df['REASONNAME'] = df['REASONNAME'].replace(['Land Development', 'Clean and Green', 'Urban Forestry', 'Natural Resources', 'Park Projects', 'Tree Crew', 'District 2', 'Clean and Green Natural Areas'], 'land')
    # replace with
    df['REASONNAME'] = df['REASONNAME'].replace('Facility License', 'license')
    # replace with
    df['REASONNAME'] = df['REASONNAME'].replace(['Dangerous Premise','Historic Preservation', 'Engineering Division'], 'buildings')
    # replace with
    df['REASONNAME'] = df['REASONNAME'].replace(['Graffiti (IntExp)', 'General Sanitation', 'Graffiti'], 'cleanup')
    # replace with
    df['REASONNAME'] = df['REASONNAME'].replace(['Waste Collection', 'Solid Waste'], 'waste')
    # replace with
    df['REASONNAME'] = df['REASONNAME'].replace(['Waste Collection', 'Solid Waste'], 'waste')
    return df

#-----------------------------------------------------------------------------

# rename columns
def clean_column_names(df):
    '''This function reads in a dataframe as a positional argument, makes the column names easier to call and
    more python friendly. It also extracts the zip code from the address column. It then returns a cleaned data 
    frame.'''
    df= df.rename(columns={
                    'Category':'category', 'OPENEDDATETIME':'open_date', 'Dept': 'dept',
                    'SLA_Date':'due_date', 'CLOSEDDATETIME': 'closed_date', 'Late (Yes/No)': 'is_late',
                    'OBJECTDESC': 'address', 'REASONNAME': 'call_reason', 'TYPENAME': 'case_type', 
                    'Council District': 'council_district', 'CASEID': 'case_id',
                    'CaseStatus': 'case_status', 'SourceID':'source_id', 'XCOORD': 'longitude', 'YCOORD': 'latitude',
                    'Report Starting Date': 'report_start_date', 'Report Ending Date': 'report_end_date'
                      })
    df['zipcode'] = df['address'].str.extract(r'(\d{5}\-?\d{0,4})')
    return df

#-----------------------------------------------------------------------------

# clean the whole df
def clean_311(df):
    '''Takes in all previous funcitons to clean the whole df'''
    # Drop columns and set index
    df = drop_and_index(df)
    # hadle null values
    df = handle_nulls(df)
    # creating delay involved columns
    df = create_delay_columns(df)
    # merge reasons
    df = clean_reason(df)
    # rename columns
    df = clean_column_names(df)
    # return df
    return df
#~~~~~~~~~~~~~~~~~~

def clean_reason(df):
    '''
    This function will take in the service call df and replace the content of REASONNAME column with condensed names
    '''
    df['REASONNAME'] = df['REASONNAME'].replace(['Waste Collection', 'Solid Waste', 'Brush'], 'waste')
    df['REASONNAME'] = df['REASONNAME'].replace(['Code Enforcement', 'Code Enforcement (Internal)', 'Code Enforcement (IntExp)'], 'code')
    df['REASONNAME'] = df['REASONNAME'].replace(['Field Operations', 'Vector'], 'field')
    df['REASONNAME'] = df['REASONNAME'].replace('Miscellaneous', 'misc')
    df['REASONNAME'] = df['REASONNAME'].replace(['Traffic Operations', 'Traffic Engineering Design', 'Traffic Issue Investigation'], 'traffic')
    df['REASONNAME'] = df['REASONNAME'].replace(['Streets', 'Signals', 'Signs and Markings'], 'streets')
    df['REASONNAME'] = df['REASONNAME'].replace('Trades', 'trades')
    df['REASONNAME'] = df['REASONNAME'].replace(['Stormwater', 'Storm Water', 'Storm Water Engineering'], 'storm')
    df['REASONNAME'] = df['REASONNAME'].replace(['Small Business', 'Food Establishments', 'Shops (Internal)', 'Shops'], 'business')
    df['REASONNAME'] = df['REASONNAME'].replace('Workforce Development', 'workforce')
    df['REASONNAME'] = df['REASONNAME'].replace(['Customer Service', '311 Call Center', 'Director\'s Office Horizontal'], 'customer_service')
    df['REASONNAME'] = df['REASONNAME'].replace(['Land Development', 'Clean and Green', 'Urban Forestry', 'Natural Resources', 'Park Projects', 'Tree Crew', 'District 2', 'Clean and Green Natural Areas'], 'land')
    df['REASONNAME'] = df['REASONNAME'].replace('Facility License', 'license')
    df['REASONNAME'] = df['REASONNAME'].replace(['Dangerous Premise','Historic Preservation', 'Engineering Division'], 'buildings')
    df['REASONNAME'] = df['REASONNAME'].replace(['Graffiti (IntExp)', 'General Sanitation', 'Graffiti'], 'cleanup')
    df['REASONNAME'] = df['REASONNAME'].replace(['Waste Collection', 'Solid Waste'], 'waste')
    df['REASONNAME'] = df['REASONNAME'].replace(['Waste Collection', 'Solid Waste'], 'waste')
    return df
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def split_separate_scale(df, stratify_by= None):
    '''
    This function will take in a dataframe
    seperate the dataframe into train, validate, and test dataframes
    seperate the target variable from train, validate and test
    then it will scale the numeric variables in train, validate, and test
    finally it will return all dataframes individually
    '''
    # split data into train, validate, test
    train, validate, test = split(df, random_state= 123, stratify_by= None)
     # seperate target variable
    X_train, y_train, X_validate, y_validate, X_test, y_test = separate_y(train, validate, test)
    # scale numeric variable
    train_scaled, validate_scaled, test_scaled = scale_data(X_train, X_validate, X_test)
    return train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test, train_scaled, validate_scaled, test_scaled

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def mother_function():
    df= pd.read_csv('service_calls.csv')
    # Drop category, report starting date, and report ending date
    df.drop(columns=['Category', 
                 'Report Starting Date', 
                 'Report Ending Date' ], inplace=True)
    # Set index to case id
    df.set_index('CASEID')
    # drop null values
    df.dropna(subset = ['SLA_Date', 'XCOORD', 'YCOORD'], inplace = True)
    # replace null values
    df.fillna({'Dept': 'Unknown'}, inplace = True)
    # remove city council department
    df = df[df.Dept != 'City Council']
   # make sure the open and closed date columns are formatted in datetime format
    df['OPENEDDATETIME'] = pd.to_datetime(df['OPENEDDATETIME'])
    df['CLOSEDDATETIME'] = pd.to_datetime(df['CLOSEDDATETIME'])
    df['SLA_Date'] = pd.to_datetime(df['SLA_Date'])
    # create new number of days open feature
    df['days_open'] = df['CLOSEDDATETIME'] - df['OPENEDDATETIME']
    # Create new column to hold days before needed resolution
    df['resolution_days_due'] = df['SLA_Date'] - df['OPENEDDATETIME']
    # Convert to string format insted of timedelta
    df['days_open'] = df.days_open // pd.Timedelta('1d')
    # Convert to string format insted of timedelta
    df['resolution_days_due'] = df.resolution_days_due // pd.Timedelta('1d')
    # create new feature to show how long it took to resolve compared to resolution due date
    df['days_before_or_after_due'] = df.resolution_days_due - df.days_open
        # postitive means before days before due data and negative means number of days after due
    # bin how long it took compare to due date to get level of delay
    df['level_of_delay'] = pd.cut(df.days_before_or_after_due, 
                                bins = [-700,-500,-300,-100,0,100,300,500],
                                labels = ['Extremely Late Response', 'Very Late Response', 
                                          'Late Response', "On Time Response", "Early Response", 
                                          'Very Early Response', 'Extremely Early Response'])
    # add a new category for level of delay
    df["level_of_delay"] = df['level_of_delay'].cat.add_categories('Still Open')
    # replace nulls in level of delay with the new "Still Open" category
    df["level_of_delay"].fillna("Still Open", inplace=True)
   # replace with waste
    df['REASONNAME'] = df['REASONNAME'].replace(['Waste Collection', 'Solid Waste', 'Brush'], 'waste')
    # replace with code
    df['REASONNAME'] = df['REASONNAME'].replace(['Code Enforcement', 'Code Enforcement (Internal)', 'Code Enforcement (IntExp)'], 'code')
    # replace with
    df['REASONNAME'] = df['REASONNAME'].replace(['Field Operations', 'Vector'], 'field')
    # replace with
    df['REASONNAME'] = df['REASONNAME'].replace('Miscellaneous', 'misc')
    # replace with
    df['REASONNAME'] = df['REASONNAME'].replace(['Traffic Operations', 'Traffic Engineering Design', 'Traffic Issue Investigation'], 'traffic')
    # replace with
    df['REASONNAME'] = df['REASONNAME'].replace(['Streets', 'Signals', 'Signs and Markings'], 'streets')
    # replace with
    df['REASONNAME'] = df['REASONNAME'].replace('Trades', 'trades')
    # replace with
    df['REASONNAME'] = df['REASONNAME'].replace(['Stormwater', 'Storm Water', 'Storm Water Engineering'], 'storm')
    # replace with
    df['REASONNAME'] = df['REASONNAME'].replace(['Small Business', 'Food Establishments', 'Shops (Internal)', 'Shops'], 'business')
    # replace with
    df['REASONNAME'] = df['REASONNAME'].replace('Workforce Development', 'workforce')
    # replace with
    df['REASONNAME'] = df['REASONNAME'].replace(['Customer Service', '311 Call Center', 'Director\'s Office Horizontal'], 'customer_service')
    # replace with
    df['REASONNAME'] = df['REASONNAME'].replace(['Land Development', 'Clean and Green', 'Urban Forestry', 'Natural Resources', 'Park Projects', 'Tree Crew', 'District 2', 'Clean and Green Natural Areas'], 'land')
    # replace with
    df['REASONNAME'] = df['REASONNAME'].replace('Facility License', 'license')
    # replace with
    df['REASONNAME'] = df['REASONNAME'].replace(['Dangerous Premise','Historic Preservation', 'Engineering Division'], 'buildings')
    # replace with
    df['REASONNAME'] = df['REASONNAME'].replace(['Graffiti (IntExp)', 'General Sanitation', 'Graffiti'], 'cleanup')
    # replace with
    df['REASONNAME'] = df['REASONNAME'].replace(['Waste Collection', 'Solid Waste'], 'waste')
    # replace with
    df['REASONNAME'] = df['REASONNAME'].replace(['Waste Collection', 'Solid Waste'], 'waste')
    df= df.rename(columns={
                    'Category':'category', 'OPENEDDATETIME':'open_date', 'Dept': 'dept',
                    'SLA_Date':'due_date', 'CLOSEDDATETIME': 'closed_date', 'Late (Yes/No)': 'is_late',
                    'OBJECTDESC': 'address', 'REASONNAME': 'call_reason', 'TYPENAME': 'case_type', 
                    'Council District': 'council_district', 'CASEID': 'case_id',
                    'CaseStatus': 'case_status', 'SourceID':'source_id', 'XCOORD': 'longitude', 'YCOORD': 'latitude',
                    'Report Starting Date': 'report_start_date', 'Report Ending Date': 'report_end_date'
                      })

    #make zipcode variable
    df['zipcode'] = df['address'].str.extract(r'(\d{5}\-?\d{0,4})')
        #drop nulls in days before or after due
    df.dropna(subset = ['days_before_or_after_due', 'zipcode'], inplace = True)
    
     # split data into train, validate, test
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify= None)
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123, 
                                       stratify= None)
    return df, train, validate, test


