#!/usr/bin/env python
# coding: utf-8

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~imports~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import scipy.stats as stats
get_ipython().run_line_magic('matplotlib', 'inline')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~STATISTICAL TESTING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~mann whitney testing~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def upper_lower_mw_testing(train):
    '''This function takes in the train data set, runs a mann whitney u statistical test on districts that fall below 20,000 per capita income and districts that fall above 20,000 per capita income response time. It then returns the result of the statistical test.'''
    null_hypothesis="there is no difference between districts that fall below 20,000 per capita income and districts that fall above 20,000 per capita income response time."
    district_2 = train[train['council_district'] == 2]
    n = train.shape[0]     # number of observations
    degf = n - 2        # degrees of freedom: the # of values in the final calculation of a statistic that are free to vary.
    conf_interval = .95 # desired confidence interval
    α = 1 - conf_interval
    t, p = stats.ttest_1samp(district_2.days_before_or_after_due, train.days_before_or_after_due.mean())
    districts_lower = train[(train['council_district'] == 2) | (train['council_district'] == 3) | (train['council_district'] == 4) | (train['council_district'] == 5)]
    districts_upper = train[(train['council_district'] == 1) | (train['council_district'] == 6) | (train['council_district'] == 7) | (train['council_district'] == 8) | (train['council_district'] == 9) | (train['council_district'] == 10)]
    districts_lower.days_before_or_after_due.std(), districts_upper.days_before_or_after_due.std()
    stats.mannwhitneyu(districts_lower.days_before_or_after_due, districts_upper.days_before_or_after_due)
    if p > α:
        return print("We fail to reject the null hypothesis. The null hypothesis is that", null_hypothesis)
    else:
        return print("We reject the null hypothesis that", null_hypothesis)

    

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Pearson R Coefficient Testing~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def days_due_response_prc_testing(train):
    '''This function takes in the train data set, runs a pearson r coefficient statistical testing and returns the result of the statistical test.'''
    null_hypothesis = "There is no correlation between days until due date and response time."
    corr, p = stats.pearsonr(train.resolution_days_due, train.days_before_or_after_due)
    if p > α:
        return print("We fail to reject the null hypothesis. The null hypothesis is that", null_hypothesis)
    else:
        return print("We reject the null hypothesis that", null_hypothesis)



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~anova testing~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def dbad_anova_test(train):
    '''This function takes in the train data set, runs ANOVA statistical testing and returns the result of the statistical test.'''
    n = train.shape[0]     # number of observations
    degf = n - 2        # degrees of freedom: the # of values in the final calculation of a statistic that are free to vary.
    conf_interval = .95 # desired confidence interval
    α = 1 - conf_interval
    null_hypothesis = 'there is no difference in days before or after due date between the districts.'
    F, p = stats.f_oneway( 
        train.days_before_or_after_due[train.council_district== 0],
        train.days_before_or_after_due[train.council_district== 1],
        train.days_before_or_after_due[train.council_district== 2],
        train.days_before_or_after_due[train.council_district== 3],
        train.days_before_or_after_due[train.council_district== 4],
        train.days_before_or_after_due[train.council_district== 5],
        train.days_before_or_after_due[train.council_district== 6], 
        train.days_before_or_after_due[train.council_district== 7], 
        train.days_before_or_after_due[train.council_district== 8], 
        train.days_before_or_after_due[train.council_district== 9],
        train.days_before_or_after_due[train.council_district== 10]
        )
    if p > α:
        return print("We fail to reject the null hypothesis. The null hypothesis is that", null_hypothesis)
    else:
        return print("We reject the null hypothesis that", null_hypothesis)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def days_open_by_month_anova_test(train):
    '''This function takes in the train data set, runs ANOVA statistical testing and returns the result of the statistical test.'''
    n = train.shape[0]     # number of observations
    degf = n - 2        # degrees of freedom: the # of values in the final calculation of a statistic that are free to vary.
    conf_interval = .95 # desired confidence interval
    α = 1 - conf_interval
    null_hypothesis = 'there is no difference in days a case is open between the months a case is opened in.'
    F, p = stats.f_oneway( 
        train.days_open[train.open_month== 1],
        train.days_open[train.open_month== 2],
        train.days_open[train.open_month== 3],
        train.days_open[train.open_month== 4],
        train.days_open[train.open_month== 5],
        train.days_open[train.open_month== 6],
        train.days_open[train.open_month== 7], 
        train.days_open[train.open_month== 8], 
        train.days_open[train.open_month== 9], 
        train.days_open[train.open_month== 10],
        train.days_open[train.open_month== 11],
        train.days_open[train.open_month== 12]
        )
    if p > α:
        return print("We fail to reject the null hypothesis. The null hypothesis is that", null_hypothesis)
    else:
        return print("We reject the null hypothesis that", null_hypothesis)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def days_open_by_week_anova_test(train):
    '''This function takes in the train data set, runs ANOVA statistical testing and returns the result of the statistical test.'''
    n = train.shape[0]     # number of observations
    degf = n - 2        # degrees of freedom: the # of values in the final calculation of a statistic that are free to vary.
    conf_interval = .95 # desired confidence interval
    α = 1 - conf_interval
    null_hypothesis = 'there is no difference in days a case is open between the week a case is opened in.'
    F, p = stats.f_oneway( 
        train.days_open[train.open_week== 1],
        train.days_open[train.open_week== 2],
        train.days_open[train.open_week== 3],
        train.days_open[train.open_week== 4],
        train.days_open[train.open_week== 5],
        train.days_open[train.open_week== 6],
        train.days_open[train.open_week== 7], 
        train.days_open[train.open_week== 8], 
        train.days_open[train.open_week== 9], 
        train.days_open[train.open_week== 10],
        train.days_open[train.open_week== 11],
        train.days_open[train.open_week== 12],
        train.days_open[train.open_week== 13],
        train.days_open[train.open_week== 14],
        train.days_open[train.open_week== 15],
        train.days_open[train.open_week== 16],
        train.days_open[train.open_week== 17], 
        train.days_open[train.open_week== 18], 
        train.days_open[train.open_week== 19], 
        train.days_open[train.open_week== 20],
        train.days_open[train.open_week== 21],
        train.days_open[train.open_week== 22],
        train.days_open[train.open_week== 23],
        train.days_open[train.open_week== 24],
        train.days_open[train.open_week== 25],
        train.days_open[train.open_week== 26],
        train.days_open[train.open_week== 27], 
        train.days_open[train.open_week== 28], 
        train.days_open[train.open_week== 29], 
        train.days_open[train.open_week== 30],
        train.days_open[train.open_week== 31],
        train.days_open[train.open_week== 32],
        train.days_open[train.open_week== 33],
        train.days_open[train.open_week== 34],
        train.days_open[train.open_week== 35],
        train.days_open[train.open_week== 36],
        train.days_open[train.open_week== 37], 
        train.days_open[train.open_week== 38], 
        train.days_open[train.open_week== 39], 
        train.days_open[train.open_week== 40],
        train.days_open[train.open_week== 41],
        train.days_open[train.open_week== 42],
        train.days_open[train.open_week== 43],
        train.days_open[train.open_week== 44],
        train.days_open[train.open_week== 45],
        train.days_open[train.open_week== 46],
        train.days_open[train.open_week== 47], 
        train.days_open[train.open_week== 48], 
        train.days_open[train.open_week== 49], 
        train.days_open[train.open_week== 50],
        train.days_open[train.open_week== 51],
        train.days_open[train.open_week== 52],
        train.days_open[train.open_week== 53]
        )
    if p > α:
        return print("We fail to reject the null hypothesis. The null hypothesis is that", null_hypothesis)
    else:
        return print("We reject the null hypothesis that", null_hypothesis)
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def days_open_by_district_anova_test(train):
    '''This function takes in the train data set, runs a ANOVA statistical testing and returns the result of the statistical test.'''
    n = train.shape[0]     # number of observations
    degf = n - 2        # degrees of freedom: the # of values in the final calculation of a statistic that are free to vary.
    conf_interval = .95 # desired confidence interval
    α = 1 - conf_interval
    null_hypothesis = 'there is no difference in days a case is open between the districts.'
    F, p = stats.f_oneway( 
        train.days_open[train.council_district== 0],
        train.days_open[train.council_district== 1],
        train.days_open[train.council_district== 2],
        train.days_open[train.council_district== 3],
        train.days_open[train.council_district== 4],
        train.days_open[train.council_district== 5],
        train.days_open[train.council_district== 6], 
        train.days_open[train.council_district== 7], 
        train.days_open[train.council_district== 8], 
        train.days_open[train.council_district== 9],
        train.days_open[train.council_district== 10]
        )
    if p > α:
        return print("We fail to reject the null hypothesis. The null hypothesis is that", null_hypothesis)
    else:
        return print("We reject the null hypothesis that", null_hypothesis)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Chi^2 Test~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def reason_chi_test(df):
    '''Runs chi square test for level and call reason'''
    # normlaize makes it percentage
    observe = pd.crosstab(train.call_reason, train.level_of_delay, margins = True)
    chi2, p, degf, expected = stats.chi2_contingency(observe)
    # Chi test is for catigorical vs catigorical
    null_hypothesis = "The reason for the call and the level of delay are independent from each other"
    alt_hypothesis = "The reason for calling and the delay are dependent from one another."
    alpha = .05 #my confident if 0.95 therfore my alpha is .05
    if p < alpha:
        print("I reject the hypothesis that: \n", null_hypothesis)
        print(' ')
        print("I move forward with my alternative hypothesis that \n", alt_hypothesis)
        print(' ')
        print(f'The alpha is: \n', alpha)
        print(' ')
        print(f'P Value is: \n', p)
    else:
        print("I fail to reject the null hypothesis")
        print("There is not enough evidence to move forward with the alternative hypothesis")
        print(f'P Value is: \n', p)
        print(' ')
        print(f'P Value is: \n', alpha)
        
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def dept_chi_test(df):
    '''Runs chi square test for level and department'''
    # normlaize makes it percentage
    observe = pd.crosstab(train.dept, train.level_of_delay, margins = True)
    chi2, p, degf, expected = stats.chi2_contingency(observe)
    # Chi test is for catigorical vs catigorical
    null_hypothesis = "The department hadling a call and the level of delay are independent from each other"
    alt_hypothesis = "The department and the delay are dependent from one another."
    alpha = .05 #my confident if 0.95 therfore my alpha is .05
    if p < alpha:
        print("I reject the null hypothesis that: \n", null_hypothesis)
        print(' ')
        print("I move forward with my alternative hypothesis that \n", alt_hypothesis)
        print(' ')
        print(f'The alpha is: \n', alpha)
        print(' ')
        print(f'P Value is: \n', p)
    else:
        print("I fail to reject the null hypothesis")
        print("There is not enough evidence to move forward with the alternative hypothesis")
        print(f'P Value is: \n', p)
        print(' ')
        print(f'P Value is: \n', alpha)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        
def by_dept_chi_square(train):
    '''This function takes in the train data set, performs a chi squared statistical test, and returns the result of the statistical test.'''
    n = train.shape[0]     # number of observations
    degf = n - 2        # degrees of freedom: the # of values in the final calculation of a statistic that are free to vary.
    conf_interval = .95 # desired confidence interval
    α = 1 - conf_interval
    # set the null hypothesis
    null_hypothesis = 'there is no difference in if a case resolution is late between the districts.'
    # make contigency table
    contingency_table = pd.crosstab(train.council_district, train.is_late)
    # run chi squared test
    test_results = stats.chi2_contingency(contingency_table)
    # find p value
    _, p, _, expected = test_results
    #give results of statistical testing
    if p > α:
        return print("We fail to reject the null hypothesis. The null hypothesis is that", null_hypothesis)
    else:
        return print("We reject the null hypothesis that", null_hypothesis)        
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def by_district_chi_square(train):
    '''This function takes in the train data set, runs a chi squared statistical testing and returns the result of the statistical test.'''
    n = train.shape[0]     # number of observations
    degf = n - 2        # degrees of freedom: the # of values in the final calculation of a statistic that are free to vary.
    conf_interval = .95 # desired confidence interval
    α = 1 - conf_interval
    null_hypothesis = 'the number of registered voters in a district does not affect the level of delay.' # set the null hypothesis     # make contigency table
    contingency_table = pd.crosstab(train.council_district, train.level_of_delay)
    # run chi squared test
    test_results = stats.chi2_contingency(contingency_table)
    # find p value
    _, p, _, expected = test_results
    
    #give results of statistical testing
    if p > α:
        print("We fail to reject the null hypothesis. The null hypothesis is that", null_hypothesis)
    else:
        print("We reject the null hypothesis that", null_hypothesis)   
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~       
def chi2(df, variable, target, alpha=0.05):
    '''
    This function takes in 4 arguments:
    1.  df
    2. categorical variable
    3.  target variable
    4. alpha
    This function returns:
    1.  chi squared test and statistical analysis
    '''
    # crosstab
    observed = pd.crosstab(df[variable], df[target])
    # run chi2 test and returns chi2 stats, p-value, degrees of freedom, and explected values.
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    # Print the chi2 value and pvalue
    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}\n')
    # Tests whether the chi_squared test rejects the null hypothesis or not.
    if p < alpha:
        print(f'The p-value: {round(p, 4)} is less than alpha: {alpha}, we can reject the null hypothesis')
    else:
        print('There is insufficient evidence to reject the null hypothesis')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~T- Tests~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def january_t_test(train):
    '''This function takes in the train data set, runs a t test statistical testing and returns the result of the statistical test.'''
    H0 = 'there is no difference in days a case is open between month 1 and the rest of the population'
    H1 = 'there is a significant difference between the days a case is open between month 1 and the general population'
    alpha = .05
    month1 = train[train['open_month'] == 1]
    μ = train.days_open.mean()
    xbar = month1.days_open.mean()
    s = month1.days_open.std()
    n = month1.shape[0]
    degf = n - 1
    standard_error = s / sqrt(n)
    t = (xbar - μ) / (s / sqrt(n))
    p = stats.t(degf).sf(t) * 2 # *2 for two-tailed test
    print(t)
    print(p)
    if p < alpha:
        print(H1)
    else:
        print(H0)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def february_t_test(train):
    '''This function takes in the train data set, runs a t test statistical testing and returns the result of the statistical test.'''
    H0 = 'there is no difference in days a case is open between month 2 and the rest of the population'
    H1 = 'there is a significant difference between the days a case is open between month 2 and the general population'
    alpha = .05
    month2 = train[train['open_month'] == 2]
    μ = train.days_open.mean()
    xbar = month2.days_open.mean()
    s = month2.days_open.std()
    n = month2.shape[0]
    degf = n - 1
    standard_error = s / sqrt(n)
    t = (xbar - μ) / (s / sqrt(n))
    p = stats.t(degf).sf(t) * 2 # *2 for two-tailed test
    print(t)
    print(p)
    if p < alpha:
        print(H1)
    else:
        print(H0)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def march_t_test(train):
    '''This function takes in the train data set, runs a t test statistical testing and returns the result of the statistical test.'''
    H0 = 'there is no difference in days a case is open between month 3 and the rest of the population'
    H1 = 'there is a significant difference between the days a case is open between month 3 and the general population'
    alpha = .05
    month3 = train[train['open_month'] == 3]
    μ = train.days_open.mean()
    xbar = month3.days_open.mean()
    s = month3.days_open.std()
    n = month3.shape[0]
    degf = n - 1
    standard_error = s / sqrt(n)
    t = (xbar - μ) / (s / sqrt(n))
    p = stats.t(degf).sf(t) * 2 # *2 for two-tailed test
    print(t)
    print(p)
    if p < alpha:
        print(H1)
    else:
        print(H0)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def december_t_test(train):
    '''This function takes in the train data set, runs a t test statistical testing and returns the result of the statistical test.'''
    H0 = 'there is no difference in days a case is open between month 12 and the rest of the population'
    H1 = 'there is a significant difference between the days a case is open between month 12 and the general population'
    alpha = .05
    month12 = train[train['open_month'] == 12]
    μ = train.days_open.mean()
    xbar = month12.days_open.mean()
    s = month12.days_open.std()
    n = month12.shape[0]
    degf = n - 1
    standard_error = s / sqrt(n)
    t = (xbar - μ) / (s / sqrt(n))
    p = stats.t(degf).sf(t) * 2 # *2 for two-tailed test
    print(t)
    print(p)
    if p < alpha:
        print(H1)
    else:
        print(H0)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def district_2_days_open_t_test(train):
    '''This function takes in the train data set, runs a t test statistical testing and returns the result of the statistical test.'''
    H0 = 'there is no difference in days a case is open between district 2 and the rest of the population'
    H1 = 'there is a significant difference between the days a case is open between district 2 and the general population'
    alpha = .05
    μ = train.days_open.mean()
    xbar = district2.days_open.mean()
    s = district2.days_open.std()
    n = district2.shape[0]
    degf = n - 1
    standard_error = s / sqrt(n)
    t = (xbar - μ) / (s / sqrt(n))
    p = stats.t(degf).sf(t) * 2 # *2 for two-tailed test
    print(t)
    print(p)
    if p < alpha:
        print(H1)
    else:
        print(H0)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def district_3_days_open(train):
    '''This function takes in the train data set, runs a t test statistical testing and returns the result of the statistical test.'''
    H0 = 'there is no difference in days a case is open between district 3 and the rest of the population'
    H1 = 'there is a significant difference between the days a case is open between district 3 and the general population'
    alpha = .05
    μ = train.days_open.mean()
    xbar = district3.days_open.mean()
    s = district3.days_open.std()
    n = district3.shape[0]
    degf = n - 1
    standard_error = s / sqrt(n)
    t = (xbar - μ) / (s / sqrt(n))
    p = stats.t(degf).sf(t) * 2 # *2 for two-tailed test
    print(t)
    print(p)
    if p < alpha:
        print(H1)
    else:
        print(H0)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def registered_voters_t_test(train):
    '''This function takes in the train data set, runs a t test statistical testing and returns the result of the statistical test.'''
    n = train.shape[0]     # number of observations
    degf = n - 2        # degrees of freedom: the # of values in the final calculation of a statistic that are free to vary.
    conf_interval = .95 # desired confidence interval
    α = 1 - conf_interval
    null_hypothesis = 'the number of registered voters in a district does not affect the number of days open.' # set the null hypothesis

    x1 = train[train.council_district == 5].days_open
    x2 = train[train.council_district != 5].days_open
    t, p= stats.ttest_ind(x1, x2)
    if p > α:
        print("We fail to reject the null hypothesis. The null hypothesis is that", null_hypothesis)
    else:
        print("We reject the null hypothesis that", null_hypothesis)
        
        
        
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~PREP FOR VISUALIZATIONS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~      
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~VISUALIZATION FUNCTIONS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def make_days_bad_relplot(train):
    '''This function creates a relplot from the train data set for days before or after due.'''
    plt.figure(figsize=(20, 10))
    sns.relplot(x='case_id', y='days_before_or_after_due', col= 'source_id', hue='dept', data=train)
    plt.xlabel("Case ID")
    plt.ylabel("Days early or late")
    plt.subplots_adjust(top=0.85)
    plt.suptitle('Evaluating number of days early or late per call type')
    return plt.show()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def make_avg_days_by_dept(train):
    '''This function creates a relplot from the train data set for average days open by department.'''
    plt.figure(figsize=(20, 10))
    sns.relplot(x='case_id', y='days_open', col= 'dept', hue='is_late', data=train)
    plt.xlabel("Case ID")
    plt.ylabel("Days early or late")
    plt.subplots_adjust(top=0.85)
    plt.suptitle('Evaluating average number of days open by dept')
    return plt.show()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def make_isLate(train):
    '''This function makes a stripplot for is late from the train data set.'''
    plt.figure(figsize=(20, 10))
    sns.stripplot(x="dept", y="case_id", hue='is_late', data=train, jitter=0.05)
    plt.xlabel("Department")
    plt.ylabel("Case ID")
    plt.suptitle('Evaluating is late by department')
    return plt.show()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def dummy_dept(df):
    '''This function accepts a dataframe, makes dummy variables of the departments and concatenates them to the dataframe.'''
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
    return df
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def make_source_id_dummies(df):
    '''This function takes in the cleaned dataframe, makes dummy variables of the source id column, readds the names of the
    dummy columns and returns the concatenated dummy dataframe to the original dataframe.'''
    #make dummies
    dummy_df = pd.get_dummies(df['source_id'])
    #add back column names
    dummy_df.columns = ['web_portal', '311_mobile_app', 'constituent_call', 'internal_services_requests']
    # concatenate dummies to the cleaned data frame
    df = pd.concat([df, dummy_df], axis=1)
    return df
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_response_by_district(train):
    '''This visual shows the number of cases in each level of
    response time in each council district'''
    # Set figure size
    plt.figure(figsize=(16,6))
    # set title of plot
    plt.title("Delay Levels Accross Districts", size=20, color='black')
    # rename x-axis so it isnt level_of_delay
    plt.xlabel('Level Of Response Time')
    # rename y-axis so it is not count
    plt.ylabel('Number of Cases')
    # create the visual (palette subject to change)
    sns.countplot(x='level_of_delay', hue='council_district', data=train,
                   palette='viridis')
    # show just the plot
    plt.show()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_response_by_dept(train):
    '''This visual shows the number of cases in each
    level of delay by department'''
    # set figure size
    plt.figure(figsize=(16,6))
    # make the title of the visual
    plt.title("Delay Levels Accross Departments", size=20, color='black')
    # rename x-axis so it is not level_of_delay
    plt.xlabel('Level Of Response Time')
    # rename y-axis so it is not count
    plt.ylabel('Number of Cases')
    # make the visual itself (palette subject to change)
    sns.countplot(x='level_of_delay', hue='dept', data=train,
                   palette='viridis_r')
    # just show the visual
    plt.show()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_response_by_reason(train):
    '''This visual shows the number of cases for each delay level
    by the reason for the case being made'''
    # set figure size
    plt.figure(figsize=(16,6))
    # set the titke
    plt.title("Delay Levels Accross Call Reasons", size=20, color='black')
    # rename x-axis so it is not level_of_delay
    plt.xlabel('Level of Response Time')
    # rename y-axis so it is not just count
    plt.ylabel('Number of Cases')
    # make the visual (palette subject to change)
    sns.countplot(x='level_of_delay', hue='call_reason', data=train,
                   palette='viridis_r')
    # just show the visual
    plt.show()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def dist_council(df):
    '''A distribution of calls by council district. The northern districts seem to make 311 reports less often'''
    plt.subplots(figsize=(22, 6))
    sns.set_theme(style="darkgrid")
    sns.countplot(data = df, x = 'council_district', palette = "magma").set_title('Count of Calls by District')
    plt.show()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def dist_timely(df):
    '''A distribution of responses by how timely they were. Most responses are very early it would seem'''
    plt.subplots(figsize=(22, 6))
    sns.set_theme(style="darkgrid")
    sns.countplot(data = df, x = 'level_of_delay', palette = "magma").set_title('Counts by Level of Delay')
    plt.show()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def dist_calls_month(df):
    '''A distribution of calls by month. Seems as though less calls are being made during the fall and winter months.'''
    plt.subplots(figsize=(22, 6))
    sns.set_theme(style="darkgrid")
    sns.countplot(data = df, x = 'open_month', palette = "magma").set_title('Count of Calls by Month Opened')
    plt.show()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def avg_by_month(train):
    ''' Distribution of average response time by month'''
    plt.subplots(figsize=(22, 6))
    sns.set_theme(style="darkgrid")
    sns.barplot(data = train.groupby('open_month').mean().reset_index(), x = 'open_month', y = 'days_before_or_after_due', palette = "viridis").set_title('Average Days Before or After Due by Month')
    plt.show()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def dbad_dist_avg(train):
    ''' Showing the days_before_or_after_due average by district, district 2 is the latest. Looks like those 
 northern districts are getting better service too. Although with the evening out that happens on median
 perhaps there are some outliers that are dragging the numbers down.'''
    district_df = create_district_df(train)
    plt.subplots(figsize=(22, 6))
    sns.set_theme(style="darkgrid")
    sns.barplot(data = district_df, x = 'council_district', y = 'days_before_or_after_due', palette = "viridis").set_title('Average Days Before or After Due')
    plt.show()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def dbad_dept_avg(train):
    '''Showing the days_before_or_after_due average by dept, buildings obviously taking the longest and cleanup 
is also regularly late. Customer service may appear to perform poorly because their tasks are typically 
given low priority.'''
    call_reason_df = create_call_reason_df(train)
    plt.subplots(figsize=(22, 6))
    sns.set_theme(style="darkgrid")
    sns.barplot(data = call_reason_df, x = 'call_reason', y = 'days_before_or_after_due', palette = "viridis").set_title('Average Days Before or After Due Date by Reason')
    plt.show()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def dept_count_plot(train):
    '''Showing the counts of calls by dept, solid waste management is the most called on by far'''
    plt.subplots(figsize=(22, 6))
    sns.set_theme(style="darkgrid")
    sns.countplot(data = train, x = 'dept', palette = "viridis").set_title('Counts of Calls by Department')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def dbad_avg_plot(train):
    '''Showing the days_before_or_after_due average by dept, with the consistency of district response times in 
 comparison, it would seem department and call_reason are large indicators of how early/late a task will be done'''
    dept_df = create_dept_df(train)
    plt.subplots(figsize=(22, 6))
    sns.set_theme(style="darkgrid")
    sns.barplot(data = dept_df, x = 'dept', y = 'days_before_or_after_due', palette = "viridis").set_title('Average Days Before or After Due Date by Department')
    plt.show()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def resolution_days_due_vs_days_before_or_after_due(train):
    '''
    This function will take in the train dataframe from the City of San Antonio 311 Data
    and return a graph dipicting the linear relationship between how many days a case is 
    given to be resolved and how many days the case is early or late
    '''
    plot = sns.relplot(data = train, x = 'resolution_days_due', y = 'days_before_or_after_due', hue = 'council_district', palette= 'Dark2')
    plot.set(xlim=(0, 100))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def sa_map(train):
    '''
    This function will take in the train dataframe from the City of San Antoniot 311 data
    and return a crude map of the city by making data points using geospatial coordinates 
    of 311 cases provided by the city
    '''
    sns.histplot(data = train, x = 'longitude', y ='latitude', hue = 'council_district', palette= 'Dark2')
    plt.xticks(rotation = 45)
    plt.tight_layout()
    plt.show()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_response_by_district(train):
    '''This visual shows the number of cases in each level of
    response time in each council district'''
    # Set figure size
    plt.figure(figsize=(16,6))
    # set title of plot
    plt.title("Delay Levels Accross Districts", size=20, color='black')
    # rename x-axis so it isnt level_of_delay
    plt.xlabel('Level Of Response Time')
    # rename y-axis so it is not count
    plt.ylabel('Number of Cases')
    # create the visual (palette subject to change)
    sns.countplot(x='level_of_delay', hue='council_district', data=train,
                   palette='viridis')
    # show just the plot
    plt.show()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_response_by_dept(train):
    '''This visual shows the number of cases in each
    level of delay by department'''
    # set figure size
    plt.figure(figsize=(16,6))
    # make the title of the visual
    plt.title("Delay Levels Accross Departments", size=20, color='black')
    # rename x-axis so it is not level_of_delay
    plt.xlabel('Level Of Response Time')
    # rename y-axis so it is not count
    plt.ylabel('Number of Cases')
    # make the visual itself (palette subject to change)
    sns.countplot(x='level_of_delay', hue='dept', data=train,
                   palette='viridis_r')
    # just show the visual
    plt.show()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_response_by_reason(train):
    '''This visual shows the number of cases for each delay level
    by the reason for the case being made'''
    # set figure size
    plt.figure(figsize=(16,6))
    # set the titke
    plt.title("Delay Levels Accross Call Reasons", size=20, color='black')
    # rename x-axis so it is not level_of_delay
    plt.xlabel('Level of Response Time')
    # rename y-axis so it is not just count
    plt.ylabel('Number of Cases')
    # make the visual (palette subject to change)
    sns.countplot(x='level_of_delay', hue='call_reason', data=train,
                   palette='viridis_r')
    
    plt.legend(loc='upper right')
    # just show the visual
    plt.show() 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_open_vs_resolve(train):
    '''creates 2 different plots
    1 plot will reflect level of delay by days open and days till resolution
    other plot will reflect department by days open and days till resolution'''
    # set subplot and its size
    plt.subplots(1, 2, figsize=(40,13), sharey=True)
    # set grid color
    sns.set(style="white")
    # create custom color palette for second plot
    dept_colors = ['sienna', 'darkgray', 'rebeccapurple',
                   'red', 'aqua', 'fuchsia', 'orange', 'green', 
                   'violet']
    # put first plot in its position for subplotting
    plt.subplot(1,2,1)
    # make the title
    plt.title("Levels of Dealy by Numbers of Days Open by Number of Days to Resolution", size=20, color='black')
    # name the x axis
    plt.xlabel('Number of Days the Case was Open')
    # name the y axis
    plt.ylabel('Number of Days the Case was Given to be Resolved')
    # create the plot
    sns.scatterplot(data=train, x='days_open', y='resolution_days_due', palette='nipy_spectral',
                    hue='level_of_delay', edgecolor='black')
    # put second plot in its position within subplot
    plt.subplot(1,2,2)
    # set the title
    plt.title("Departments by Numbers of Days Open by Number of Days to Resolution", size=20, color='black')
    # set x axis name
    plt.xlabel('Number of Days the Case was Open')
    # set y axis name
    plt.ylabel('Number of Days the Case was Given to be Resolved')
    # create plot
    sns.scatterplot(data=train, x='days_open', y='resolution_days_due', palette=dept_colors,
                    hue='dept', edgecolor='black')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_source_resolution_days(train):
    '''plots the level of delays based on the source of the report'''
    # set figure sizing
    plt.figure(figsize=(16,6))
    # make the plot
    sns.barplot(data=train, x="level_of_delay", y="resolution_days_due", hue='source_id', palette='viridis')
    # place the legend in desired location
    plt.legend(loc='upper right')
    # make the title
    plt.title("Delay Level vs. Resolution Days Due Based on Reporting System")
    # make the x label
    plt.xlabel('Level of Delay')
    # make the y label
    plt.ylabel('Number of Days given for a Resolution to be Made')
    # show just the plot
    plt.show()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_dept_resolution_days(train):
    '''plots the level of delays based on the source of the report'''
    # set figure sizing
    plt.figure(figsize=(16,6))
    # make the plot
    sns.barplot(data=train, x="level_of_delay", y="resolution_days_due", hue='dept', palette='viridis')
    # place the legend in desired location
    plt.legend(loc='upper right')
    # make the title
    plt.title("Delay Level vs. Resolution Days Due Based on Department")
    # make the x label
    plt.xlabel('Level of Delay')
    # make the y label
    plt.ylabel('Number of Days given for a Resolution to be Made')
    # show just the plot
    plt.show()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~