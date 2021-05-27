#!/usr/bin/env python
# coding: utf-8

# In[ ]:
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
    null_hypothesis="there is no difference between districts that fall below 20,000 per capita income and districts that fall above 20,000 per capita income response time."
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
    null_hypothesis = "There is no correlation between days until due date and response time."
    corr, p = stats.pearsonr(train.resolution_days_due, train.days_before_or_after_due)
    if p > α:
        return print("We fail to reject the null hypothesis. The null hypothesis is that", null_hypothesis)
    else:
        return print("We reject the null hypothesis that", null_hypothesis)



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~anova testing~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def dbad_anova_test(train):
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
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def days_open_by_district_anova_test(train):
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
        
        
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~T- Tests~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def january_t_test(train):
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
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~VISUALIZATION FUNCTIONS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~