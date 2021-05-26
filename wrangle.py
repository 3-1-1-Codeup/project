import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.preprocessing
import seaborn as sns
import matplotlib.pyplot as plt


#-----------------------------------------------------------------------------

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
    df.set_index('CASEID', inplace=True)
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
    # replace null values in days open with 0
    df['days_open'] = df['days_open'].fillna(0)
    # add 1 to resolution days to offset future issues with upcoming feature
    df['resolution_days_due'] = df['resolution_days_due'] + 1
    # create new feature to show how long it took to resolve compared to resolution due date
    df['pct_time_of_used'] = df.days_open / df.resolution_days_due
    # bin the new feature
    df['level_of_delay'] = pd.cut(df.pct_time_of_used, 
                            bins = [-20.0,0.5,0.75,1.0,15,800],
                            labels = ['Very Early Response', 
                                      'Early Response', "On Time Response", "Late Response", 
                                      'Very Late Response'])
    # drop nulls in these columns
    df.dropna(subset=['days_open'], how='all', inplace=True)
    df.dropna(subset=['level_of_delay'], how='all', inplace=True)
    # return df
    return df

#-----------------------------------------------------------------------------

def handle_outliers(df):
    '''removes outiers from df'''
    # remove outliers from days_open
    df = df[df.days_open < 1400]
    # return df
    return df

#-----------------------------------------------------------------------------

def create_dummies(df):
    '''This function creates dummy variables for Council Districts'''
    # Drop district 0
    df = df[df['Council District'] != 0]
    # set what we are going to create these dummies from
    dummy_df =  pd.get_dummies(df['Council District'])
    # Name the new columns
    dummy_df.columns = ['district_1', 'district_2', 
                        'district_3', 'district_4', 'district_5',
                        'district_6', 'district_7', 'district_8',
                        'district_9', 'district_10']
    # add the dummies to the data frame
    df = pd.concat([df, dummy_df], axis=1)
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
    df['zipcode'] = df['address'].str.extract(r'.*(\d{5}?)$')  
    #drop zipcode nulls after obtaining zipcode
    df.dropna(subset=['zipcode'], how='all', inplace=True)         
    return df

#-----------------------------------------------------------------------------

# clean the whole df
def first_iteration_clean_311(df):
    '''Takes in all previous funcitons to clean the whole df'''
    # Drop columns and set index
    df = drop_and_index(df)
    # hadle null values
    df = handle_nulls(df)
    # creating delay involved columns
    df = create_delay_columns(df)
    # handle outliers
    df = handle_outliers(df)
    # make dummies
    df = create_dummies(df)
    # merge reasons
    df = clean_reason(df)
    # rename columns
    df = clean_column_names(df)
    df.to_csv('first_iteration_clean_311.csv')
    # return df
    return df


# Train/Split the data~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def split(df, stratify_by= 'level_of_delay'):
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

def separate_y(train, validate, test):
    '''
    This function will take the train, validate, and test dataframes and separate the target variable into its
    own panda series
    '''
    
    X_train = train.drop(columns=['level_of_delay'])
    y_train = train.level_of_delay
    X_validate = validate.drop(columns=['level_of_delay'])
    y_validate = validate.level_of_delay
    X_test = test.drop(columns=['level_of_delay'])
    y_test = test.level_of_delay
    return X_train, y_train, X_validate, y_validate, X_test, y_test

# Scale the data~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def scale_data(X_train, X_validate, X_test):
    '''
    This function will scale numeric data using Min Max transform after 
    it has already been split into train, validate, and test.
    '''
    
    
    obj_col = ['open_date', 'due_date', 'closed_date', 'is_late', 'dept', 'call_reason', 'case_type', 'case_status', 'source_id', 'address', 'zipcode']
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

def split_separate_scale(df, stratify_by= 'level_of_delay'):
    '''
    This function will take in a dataframe
    separate the dataframe into train, validate, and test dataframes
    separate the target variable from train, validate and test
    then it will scale the numeric variables in train, validate, and test
    finally it will return all dataframes individually
    '''
    
    # split data into train, validate, test
    train, validate, test = split(df, stratify_by= 'level_of_delay')
    
     # seperate target variable
    X_train, y_train, X_validate, y_validate, X_test, y_test = separate_y(train, validate, test)
    
    
    # scale numeric variable
    train_scaled, validate_scaled, test_scaled = scale_data(X_train, X_validate, X_test)
    
    return train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test, train_scaled, validate_scaled, test_scaled

#------------------------------------------------------------------------------------------------------------------------------------------------
def get_sq_miles(df):
    """
    This function will apply the square miles per district
    to each district.
    """
    # Creating a dictionary with square miles from city of San Antonio to convert to dataframe
    sq_miles = {1: 26.00, 2: 59.81, 3: 116.15, 4: 65.21, 5: 22.24, 6: 38.44, 7: 32.82,
                         8: 71.64, 9: 48.71, 10: 55.62}
    # Converting to dataframe
    sq_miles = pd.DataFrame(list(sq_miles.items()),columns = ['council_district','square_miles'])
    #Merging with the original dataframe
    df = df.merge(sq_miles, on = 'council_district', how ='left')
    return df
#-------------------------------------------------------------------------------------------------------------------------------------------------

def extract_time(df):
    '''
    This function will take in a dataframe and return it with new features extracted from the open_date column
    - open_month: which month the case was opened in
    - open_year: which year the case was opened in
    - open_week: which week the case was opened in
    '''
    
    # extract month from open_date
    df['open_month'] = df.open_date.dt.month
    
    # extract year from open_date
    df['open_year'] = df.open_date.dt.year
    
    # extract week from open_date
    df['open_week'] = df.open_date.dt.week
    
    return df
    
#------------------------------------------------------------------------------------------------------------------------------------------
def find_voter_info(df):
    '''This function reads in a dataframe. Using the Council District column, it appends the voter turn out and
    number of registered voters for each district. It does NOT take into account district 0 due to that being a
    filler district for outside jurisdictions. It then appends the info onto the dataframe and returns it for later use.'''
    conditions = [
    (df['Council District'] == 1),
    (df['Council District'] == 2), 
    (df['Council District'] == 3),
    (df['Council District'] == 4),
    (df['Council District'] == 5),
    (df['Council District'] == 6),
    (df['Council District'] == 7),
    (df['Council District'] == 8),
    (df['Council District'] == 9),
    (df['Council District'] == 10)
    ]
    # create a list of the values we want to assign for each condition
    voter_turnout = [0.148, 0.086, 0.111, 0.078, 0.085, 0.124, 0.154, 0.137, 0.185, 0.154]
    registered_voters= [68081, 67656, 69022, 66370, 61418, 80007, 83287, 97717, 99309, 91378]
    # create a new column and use np.select to assign values to it using our lists as arguments
    df['voter_turnout_2019'] = np.select(conditions, voter_turnout)
    df['num_of_registered_voters'] = np.select(conditions, registered_voters)
    return df
#------------------------------------------------------------------------------------------------------------------------------------------
def add_per_cap_in(df):
    '''
    This function takes in the original cleaned dataframe and adds a per capita
    income metric by council district
    '''
    # Creating a dictionary with per_capita values from city of San Antonio to convert to dataframe
    per_cap_in = {1: 23967, 2: 19055, 3: 18281, 4: 18500, 5: 13836, 6: 23437, 7: 25263,
                         8: 35475, 9: 42559, 10: 30240}
    # Converting to dataframe
    per_cap_in = pd.DataFrame(list(per_cap_in.items()),columns = ['council_district','per_capita_income'])
    #Merging with the original dataframe
    df = df.merge(per_cap_in, on = 'council_district', how ='left')
    return df
#------------------------------------------------------------------------------------------------------------------------------------------
# clean the whole df
def clean_311(df):
    '''Takes in all previous funcitons to clean the whole df'''
    # Drop columns and set index
    df = drop_and_index(df)
    # hadle null values
    df = handle_nulls(df)
    # creating delay involved columns
    df = create_delay_columns(df)
    # handle outliers
    df = handle_outliers(df)
    # make dummies
    df = create_dummies(df)
    # merge reasons
    df = clean_reason(df)
    #add voter information
    df= find_voter_info(df)
    # rename columns
    df = clean_column_names(df)
    # add date/time information
    df= extract_time(df)
    #add per capita information
    df= add_per_cap_in(df)

    #add per sqmiles info
    df = get_sq_miles(df)

    #make clean csv with all changes
    df.to_csv('second_clean_311.csv')
    # return df
    return df