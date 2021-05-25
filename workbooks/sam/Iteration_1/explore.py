import json
import folium
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def create_sa_zip_map(table, zips, mapped_feature, add_text = ''):

    '''
    This function takes in a dataframe referencing zipcodes in San Antonio 
    and plots quantitative variables for those zipcodes with a colored map.
    Table being the dataframe referenced, zips(str) being the column referencing 
    the zipcodes, mapped_feature(str) being the column for exploring,
    and add_text(str) being the desired title of the map created. The file is 
    saved as an HTML locally.
    '''

    # Acquires the data from the .geojason
    with open('Bexar_County_ZIP_Code_Areas.geojson', 'r') as jsonFile:
        data = json.load(jsonFile)
    tmp = data

    # Matches the zipcodes in the 'zips' column referenced to the zips found in the .geojson,
    # removing the unnecessary ones in the .geojson
    geozips = []
    for i in range (len(tmp['features'])):
        if tmp['features'][i]['properties']['ZIP'] in list(table['zipcode'].unique()):
            geozips.append(tmp['features'][i])
    
    # creating a new json object
    new_json = dict.fromkeys(['type', 'features'])
    new_json['features'] = geozips
    new_json['type'] = 'FeatureCollection'

    # saving new json file as "updated-file"
    open("updated-file.json", "w").write(json.dumps(new_json, sort_keys=True, indent=4, separators=(',', ':')))

    # reading the updated json file
    sa_geo = r'updated-file.json'
    # specifying san antonio latitude and longitude for the folium map
    m = folium.Map(location = [29.4241, -98.4936], zoom_start = 11)

    m.choropleth(
        geo_data = sa_geo,
        fill_opacity = 0.7,
        line_opacity = 0.2,
        data = table,
        key_on = 'feature.properties.ZIP',
        columns = [zips, mapped_feature],
        fill_color = 'BuPu',
        legend_name = (' ').join(mapped_feature.split('_')).title() + ' ' + add_text + ' Across SA')
    folium.LayerControl().add_to(m)
    m.save(outfile = mapped_feature + '_map.html')

def create_zip_df(df):
    '''
    This function takes in the original dataframe and returns
    a dataframe with the quantitative variables averaged by zipcode
    for easier exploration
    '''
    # Creating a dataframe with zipcode and a "days_open" averaged column
    zip_df = pd.DataFrame(df.groupby('zipcode').days_open.mean()).reset_index()
    # Adding a "days_before_or_after_due" averaged column
    zip_df['days_before_or_after_due'] = pd.DataFrame(df.groupby('zipcode').days_before_or_after_due.mean()).reset_index()['days_before_or_after_due']
    # Adding a "resolution_days_due" averaged column
    zip_df['resolution_days_due'] = pd.DataFrame(df.groupby('zipcode').resolution_days_due.mean()).reset_index()['resolution_days_due']
    # Adding a "days_open" median column
    zip_df['days_open_med'] = pd.DataFrame(df.groupby('zipcode').days_open.median()).reset_index()['days_open']
    # Adding a "days_before_or_after_due" median column
    zip_df['days_before_or_after_due_med'] = pd.DataFrame(df.groupby('zipcode').days_before_or_after_due.median()).reset_index()['days_before_or_after_due']
    # Adding a "resolution_days_due" median column
    zip_df['resolution_days_due_med'] = pd.DataFrame(df.groupby('zipcode').resolution_days_due.median()).reset_index()['resolution_days_due']
    return zip_df

def create_district_df(df):
    '''
    This function takes in the original dataframe and returns
    a dataframe with the quantitative variables averaged by district
    for easier exploration
    '''
    # Creating a dataframe with zipcode and a "days_open" averaged column
    district_df = pd.DataFrame(df.groupby('council_district').days_open.mean()).reset_index()
    # Adding a "days_before_or_after_due" averaged column
    district_df['days_before_or_after_due'] = pd.DataFrame(df.groupby('council_district').days_before_or_after_due.mean()).reset_index()['days_before_or_after_due']
    # Adding a "resolution_days_due" averaged column
    district_df['resolution_days_due'] = pd.DataFrame(df.groupby('council_district').resolution_days_due.mean()).reset_index()['resolution_days_due']
    # Adding a "days_open" median column
    district_df['days_open_med'] = pd.DataFrame(df.groupby('council_district').days_open.median()).reset_index()['days_open']
    # Adding a "days_before_or_after_due" median column
    district_df['days_before_or_after_due_med'] = pd.DataFrame(df.groupby('council_district').days_before_or_after_due.median()).reset_index()['days_before_or_after_due']
    # Adding a "resolution_days_due" median column
    district_df['resolution_days_due_med'] = pd.DataFrame(df.groupby('council_district').resolution_days_due.median()).reset_index()['resolution_days_due']
    return district_df

def create_dept_df(df):
    '''
    This function takes in the original dataframe and returns
    a dataframe with the quantitative variables averaged by dept
    for easier exploration
    '''
    # Creating a dataframe with zipcode and a "days_open" averaged column
    dept_df = pd.DataFrame(df.groupby('dept').days_open.mean()).reset_index()
    # Adding a "days_before_or_after_due" averaged column
    dept_df['days_before_or_after_due'] = pd.DataFrame(df.groupby('dept').days_before_or_after_due.mean()).reset_index()['days_before_or_after_due']
    # Adding a "resolution_days_due" averaged column
    dept_df['resolution_days_due'] = pd.DataFrame(df.groupby('dept').resolution_days_due.mean()).reset_index()['resolution_days_due']
    # Adding a "days_open" median column
    dept_df['days_open_med'] = pd.DataFrame(df.groupby('dept').days_open.median()).reset_index()['days_open']
    # Adding a "days_before_or_after_due" median column
    dept_df['days_before_or_after_due_med'] = pd.DataFrame(df.groupby('dept').days_before_or_after_due.median()).reset_index()['days_before_or_after_due']
    # Adding a "resolution_days_due" median column
    dept_df['resolution_days_due_med'] = pd.DataFrame(df.groupby('dept').resolution_days_due.median()).reset_index()['resolution_days_due']
    return dept_df

def create_call_reason_df(df):
    '''
    This function takes in the original dataframe and returns
    a dataframe with the quantitative variables averaged by dept
    for easier exploration
    '''
    # Creating a dataframe with zipcode and a "days_open" averaged column
    call_reason_df = pd.DataFrame(df.groupby('call_reason').days_open.mean()).reset_index()
    # Adding a "days_before_or_after_due" averaged column
    call_reason_df['days_before_or_after_due'] = pd.DataFrame(df.groupby('call_reason').days_before_or_after_due.mean()).reset_index()['days_before_or_after_due']
    # Adding a "resolution_days_due" averaged column
    call_reason_df['resolution_days_due'] = pd.DataFrame(df.groupby('call_reason').resolution_days_due.mean()).reset_index()['resolution_days_due']
     # Adding a "days_open" median column
    call_reason_df['days_open_med'] = pd.DataFrame(df.groupby('call_reason').days_open.median()).reset_index()['days_open']
    # Adding a "days_before_or_after_due" median column
    call_reason_df['days_before_or_after_due_med'] = pd.DataFrame(df.groupby('call_reason').days_before_or_after_due.median()).reset_index()['days_before_or_after_due']
    # Adding a "resolution_days_due" median column
    call_reason_df['resolution_days_due_med'] = pd.DataFrame(df.groupby('call_reason').resolution_days_due.median()).reset_index()['resolution_days_due']
    return call_reason_df