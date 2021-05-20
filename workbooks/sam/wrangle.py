import pandas as pd

''' Reads the data from a csv into a pandas dataframe'''

def get_311():
    return pd.read_csv('311_service_calls.csv')



def handle_nulls(df):
    
    '''
    This function takes in the dataframe and addresses any null/ignorably small values
    '''

    # Drops rows with null values that were unresolvable, such as SLA_Date, XCOORD, and YCOORD.
    df.dropna(subset = ['SLA_Date', 'XCOORD', 'YCOORD'], inplace = True)
    # Fills null Dept values with "Unknown".
    df.fillna({'Dept': 'Unknown'}, inplace = True)
    # Drops issues with dept assigned to "City Council", as there was only 1
    df = df[df.Dept != 'City Council']
    return df
