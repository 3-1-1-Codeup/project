#!/usr/bin/env python
# coding: utf-8

# In[1]:
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import scipy.stats as stats
get_ipython().run_line_magic('matplotlib', 'inline')

#~~~~~~~~~~~~~~~~~~~~~
def make_dbad_relplot(train):
    plt.figure(figsize=(20, 10))
    sns.relplot(x='case_id', y='days_before_or_after_due', col= 'source_id', hue='dept', data=train)
    plt.xlabel("Case ID")
    plt.ylabel("Days early or late")
    plt.subplots_adjust(top=0.85)
    plt.suptitle('Evaluating number of days early or late per call type')
    return plt.show()



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def get_chi_square(train):
    '''This function takes in the train data set, performs a chi squared statistical test, and returns the result of the statistical test.'''
    n = train.shape[0]     # number of observations
    degf = n - 2        # degrees of freedom: the # of values in the final calculation of a statistic that are free to vary.
    conf_interval = .95 # desired confidence interval
    α = 1 - conf_interval
    # set the null hypothesis
    null_hypothesis = 'there is no difference in if a case resoution is late between the districts.'
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


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def get_anova_test(train):
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




