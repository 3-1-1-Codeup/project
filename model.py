import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn import preprocessing

#-----------------------------------------------------------------------------

# Model Prep

def dummy_dept(df):
    # dummy dept feature
    dummy_df =  pd.get_dummies(df['dept'])
    # Name the new columns
    dummy_df.columns = ['animal_care_services', 'code_enforcement_services', 
                        'customer_services', 'development_services', 
                        'metro_health', 'parks_and_rec',
                        'solid_waste_management', 'trans_and_cap_improvements', 
                        'unknown_dept']
    # add the dummies to the data frame
    df = pd.concat([df, dummy_df], axis=1)
#-----------------------------------------------------------------------------    
def dummy_call_reason(df):
    # dummy dept feature
    dummy_df =  pd.get_dummies(df['call_reason'])
    # Name the new columns
    dummy_df.columns = ['buildings', 'business', 'cleanup', 'code',
                        'customer_service', 'field', 'land',
                        'license', 'misc', 'storm', 'streets', 'trades', 
                        'traffic', 'waste']
    # add the dummies to the data frame
    df = pd.concat([df, dummy_df], axis=1)
    return df
#-----------------------------------------------------------------------------
    def make_source_id_dummies(df):
    '''This function takes in the cleaned dataframe, makes dummy variables of the source id column, readds the names of the
    dummy columns and returns the concatenated dummy dataframe to the original dataframe.'''
    #make dummies
    dummy_df = pd.get_dummies(df['source_id'])
    #add back column names
    dummy_df.columns = ['web_portal', '311_mobile_app', 
                        'constituent_call', 'interal_services_requests']
    # concatenate dummies to the cleaned data frame
    df = pd.concat([df, dummy_df], axis=1)
    return df