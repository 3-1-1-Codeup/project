<a name="top"></a>
![name of photo](https://github.com/3-1-1-Codeup/project/blob/main/workbooks/caitlyn/images/read_me_take3/title.png?raw=true)

***
[[Project Description](#project_description)]
[[Project Planning](#planning)]
[[Key Findings](#findings)]
[[Data Dictionary](#dictionary)]
[[Acquire & Prep](#acquire_and_prep)]
[[Data Exploration](#explore)]
[[Statistical Analysis](#stats)]
[[Modeling](#model)]
[[Conclusion](#conclusion)]
[[Recreate This Project](#recreate)]
[[Meet the Team](#team)]
___


## <a name="project_description"></a>
![desc](https://github.com/3-1-1-Codeup/project/blob/main/workbooks/caitlyn/images/read_me_take3/description.png?raw=true)
[[Back to top](#top)]

<details>
  <summary>Click to expand!</summary>

### Description
Using data acquired from the City of San Antonio, our team aims to create a classification model to predict the level of delay in a call's response time. From this project we want to answer what drives the level of delay and if there is a way to minimize late response times for 3-1-1 calls in our city.

### Goals
- Make a classification model to predict the level of delay in response time for a 311 call.
- See how response time is affected by different key features.
- Find the main drivers of delayed response time.
    
### Where did you get the data?
- Data was gathered from "The City of San Antonio" website
    - https://data.sanantonio.gov/dataset/service-calls/resource/20eb6d22-7eac-425a-85c1-fdb365fd3cd7
- Added data from the following website to create features such as per_capita_income, voter_turnout, etc.
    - https://sa2020.org/city-council-profiles



</details>
    
    
## <a name="planning"></a> 
![plan](https://github.com/3-1-1-Codeup/project/blob/main/workbooks/caitlyn/images/read_me_take3/planning.png?raw=true)
[[Back to top](#top)]

<details>
  <summary>Click to expand!</summary>

### Project Outline:
  
- Acquisition of data:
- Download CSV from the City of San Antonio website.
  - https://data.sanantonio.gov/dataset/service-calls/resource/20eb6d22-7eac-425a-85c1-fdb365fd3cd7
- Bring data into python
  
  
  
  
  Projet Outline:
    
- Acquisiton of data:
    - Download CSV from the City of San Antonio website.
        - https://data.sanantonio.gov/dataset/service-calls/resource/20eb6d22-7eac-425a-85c1-fdb365fd3cd7 
    - Bring data into python
    - Run basic exploration: 
      -.info()
      -.describe()
      -.isnull()
      -.value_counts()
      -basic univariate
      -key takeaways
- Prepare and clean data with python - Jupyter Labs: 
    - Set index
    - Drop features
    - Handle null values
    - Handle outliers
    - Merge some feature values (only the ones that go with each other)
    - Rename
    - Create
    - Bin to create a new categorical feature(s)
  
- Explore data:
    - What are the features?
    - What questions are we aiming to answer?
    - Categorical or continuous values.
    - Make visuals (at least 2 to be used in deliverables)
        - Univariate
        - Bivariate
        - Multivariate
- Run statistical analysis:
    - At least 2.
- Modeling:
    - Make multiple models.
    - Pick best model.
    - Test Data.
    - Conclude results.
        
### Hypothesis/Questions
- Does the type of call in an area affect the level of response?
- Does the specific location affect the response time?
- Do category and department affect response time?
- Is there a link to which form of reporting is responded to quickest and slowest?

### Target variable
- `level_of_delay`
    - Made in the feature engineering step.
        - This feature takes the number of days a case was open (open-closed) and divided it by the number of days the case was given to be resolved and calculates the percent of the allocated resolution time that was used.


</details>

    
## <a name="findings"></a> 
![find](https://github.com/3-1-1-Codeup/project/blob/main/workbooks/caitlyn/images/read_me_take3/findings.png?raw=true)

[[Back to top](#top)]

<details>
  <summary>Click to expand!</summary>

### Explore:
- Department, call reason, and number of days given for a resolution were found to be major drivers of response time.
- District was a driver, but only when paired with department or call reason. 
    
    
### Stats
- Stat Test 1: 
    - Anova
        - Null : "There is no difference in days before or after due date between the districts."
            - Reject the null
            
- Stat Test 2: 
    - Chi$^2$
        - Null: "The call reason of the issue and the level of delay are independent from each other"
        - Reject the null
    
- Stat Test 3: 
    - Mann-Whitney U 
        - Null: "There is no difference between districts that fall below 20,000 per capita income and districts that fall above 20,000 per capita income response time." 
        - Reject the null

### Modeling:
- Baseline:
    -  57.199 %
- Models Made:
    - Logistic Regression
    - KNN
    - Decision Tree
    - Random Forest
    - SGD Classifier
    - Ridge Classifier
    - Ridge CV Classifier
- Best Model:
    - Decision Tree
- Model testing:
    - Train
        - 68 %
    - Validate
        - 68 %
- Performance:
    - Test
        - 68 %

***

    
</details>

## <a name="dictionary"></a>
![dict](https://github.com/3-1-1-Codeup/project/blob/main/workbooks/caitlyn/images/read_me_take3/dict.png?raw=true)

[[Back to top](#top)]

<details>
  <summary>Click to expand!</summary>

### Data Used
    
| Attribute | Definition | Data Type |
| ----- | ----- | ----- | 
| call_reason | The department division within the City deaprtment to whom the case is assigned. | object |
| case_status | The status of a case which is either open or closed. | object |
| case_type | The service request type name for the issue being reported. Examples include stray animals, potholes, overgrown yards, junk vehicles, traffic signal malfunctions, etc. | object |
| closed_date | The date and time that the case/request was was closed. If blank, the request has not been closed as of the Report Ending Date. | object |
| council_district | The Council District number from where the issue was reported. | int64 |
| days_before_or_after_due | How long before or after the due date were the cases closed | float64 |
| days_open | The number of days between a case being opened and closed. | float64 |
| dept | The City department to whom the case is assigned. | object |
| due_date | Every service request type has a due date assigned to the request, based on the request type name. The SLA Date is the due date and time for the request type based on the service level agreement (SLA). Each service request type has a timeframe in which it is scheduled to be addressed. | object |
| is_late | This indicates whether the case has surpassed its Service Level Agreement due date for the specific service request. | object |
| open_date | The date and time that a case was submitted. | object |
| open_month | Month of the year the case was made | int64 | 
| open_week | Week of the year the case was made | int64 | 
| open_year | The year the case was made | int64 | 
| pct_time_of_used | How much of the resolution_days_due was the case open? | float64 | 
| resolution_days_due | The number of days between a case being opened and due. | float64 |
| source_id | The source id is the method of input from which the case was received. | object |
    
\*  Indicates the target feature in this City of San Antonio data.

***
</details>

## <a name="acquire_and_prep"></a> 
![acquire_prep](https://github.com/3-1-1-Codeup/project/blob/main/workbooks/caitlyn/images/read_me_take3/a&p.png?raw=true)
[[Back to top](#top)]

<details>
  <summary>Click to expand!</summary>

### Acquire Data:
- Data was gathered from "The City of San Antonio" website
    - https://data.sanantonio.gov/dataset/service-calls/resource/20eb6d22-7eac-425a-85c1-fdb365fd3cd7
  
- Added data from the following website to create features such as per_capita_income, voter_turnout, etc.
    - https://sa2020.org/city-council-profiles
    
### Prepare Data
*All functions for the following preparation can be found in the wrangle.py file on our github repository.*
- Make case id the index
- Handle null values 
- Remove unneeded features
- Create new features such as:
    - days_open
    - resolution_days_due
    - days_before_or_after_due
    - pct_time_of_used
    - voter_turnout_2019
    - num_of_registered_voters
    - per_capita_income
- Create dummy columns for district
- Rename the features to make them easier to understand and to make them easier for python to call
- Merge some values that go hand in hand from reason for calling 
- Extract zip code from the address

***

</details>



## <a name="explore"></a> 
![dict](https://github.com/3-1-1-Codeup/project/blob/main/workbooks/caitlyn/images/read_me_take3/explore.png?raw=true)
[[Back to top](#top)]

<details>
  <summary>Click to expand!</summary>
    
### Findings:
- Each department has better levels of response in certain areas.
- The departments with the lowest number of calls were more likely to have worse response times
- Internal requests were generally late in comparison to other forms of reporting. While mobile app was generally completed early.
- Customer Service generally got issues resolved late or very late.
- Animal Services usually only gave a day to complete a case and those cases usually took months to close.
- Winter months tend to have the longest average days open time, while Autumn months have the shortest.


***

</details>    

## <a name="stats"></a> 
![stats](https://github.com/3-1-1-Codeup/project/blob/main/workbooks/caitlyn/images/read_me_take3/stats.png?raw=true)
[[Back to top](#top)]
<details>
  <summary>Click to expand!</summary>

### Stats Test 1:
  
#### Confidence level and alpha value:
- 95% confidence
  - alpha = 0.05
  

- What is the test?
    - ANOVA test.
- Why use this test?
    - The ANOVA test tests the means between many groups to determine if there is a difference.
- What is being compared?
    - The mean of days before or after due for each district.
- Question being asked:
    -Is there a significant difference between districts for days before or after due date?
    
#### Hypothesis:

- Null Hypothesis: There is no difference in days before or after due date between the districts.

- Alternative Hypothesis: There is a significant difference in days before or after due date between the districts.

#### Results:
- We reject the null hypothesis.

### Stats Test 2:
    
#### Confidence level and alpha value:
- 95% confidence
  - alpha = 0.05
    
- What
    - Chi$^2$ Test.
- Why use this test?
    - This test was used because it compares two categorical data variables.
- What is being compared?
    -   Call reason and level of delay
- Question being asked:
    - Is there a significant difference between the call reason and level of delay?

#### Hypothesis:
- Null Hypothesis: "The call reason of the issue and the level of delay are independent from each other"
    
- Alternative Hypothesis: "The call reason and the level of delay are dependent from one another."

#### Results:
- We reject the null hypothesis.

### Stats Test 3:
    
#### Confidence level and alpha value:
- 95% confidence
  - alpha = 0.05
    
- What is the test?
    - Mann-Whitney U Test.
- Why use this test?
    - This test was used because it is used to test whether two samples are likely to derive from the same population .
- What is being compared?
    - Response times between districts that fall below 20,000 per capita income and districts that fall above 20,000 per capita income.
- Question being asked:
    - Is there a difference for response time for all districts that fall below 20,000 per capita income and those that are above?
    
#### Hypothesis:
- Null Hypothesis: There is no difference between districts that fall below 20,000 per capita income and districts that fall above 20,000 per capita income response time.
    
- Alternative Hypothesis: There is a difference between districts that fall below 20,000 per capita income and districts that fall above 20,000 per capita income response time.

#### Results:
- We reject the null hypothesis

***

    
</details>    

## <a name="model"></a> 
![model](https://github.com/3-1-1-Codeup/project/blob/main/workbooks/caitlyn/images/read_me_take3/model.png?raw=true)
[[Back to top](#top)]
<details>
  <summary>Click to expand!</summary>

Summary of modeling choices...
        
### Models Made:
- Logistic Regression
- Decision Tree
- Random Forest
- KNN
- Ridge Classifier
- SGD Classifier

### Baseline Accuracy  
- 57.199%
      
| Model | Accuracy with Train | Accuracy with Validate |
| ---- | ----| ---- | ---- |
| Logistic Regression | 61.1% | 61% |
| Decision Tree | 68% | 68% |
| Random Forest | 66.6% | 66.4% |
| KNN | 53.1%  | 53.2 % |
| Ridge Classifier | 59% | 59% |
| SGD Classifier | 56% | 56% |
    
    
## Selecting the Best Model:

- Decision Tree

- Why did we choose this model?
    - This model ran the best accross train and validate.
    
- What does this model do?
    - Decision trees are flexible models that don’t increase their number of parameters as we add more features (if we build them correctly). At each node of a decision tree, one of the features of our data is evaluated in order to make an specific data point follow a certain path when making a prediction.

### Model on All Data Sets

| Best Model | Accuracy with Train | Accuracy with Validate | Accuracy with Test|
| ---- | ----| ---- | ---- |
| Decision Tree | 68% | 68% | 68% |


***

</details>  

## <a name="conclusion"></a> 
![conclusion](https://github.com/3-1-1-Codeup/project/blob/main/workbooks/caitlyn/images/read_me_take3/conclusion.png?raw=true)
[[Back to top](#top)]
<details>
  <summary>Click to expand!</summary>

**We found....**

- Each department is better in certain areas about being on time/early and late in others.
- The more calls a department had the better they were at getting issues resolved on time.
- Internal requests were generally late in comparison to other forms of reporting.
- When an issue was reported via the app, there were no extremely late responses.
- Customer Service generally got issues resolved late or very late. 
- Animal Services usually only gave a day to complete a case and those cases usually took months to close.
- Winter months tend to have the longest average days open time, while Autumn months have the shortest.

**With further time...**

- Overall extremely late responses are spread out throughout the city. There is a significant delay within calls listed as on time. Therefore, we would like to evaluate the amount of time between districts for calls that were considered on time. 
- Analyze the data further through time series analysis. Some questions that we would like to investigate are:
    - Do days of the week effect when the case was done?
    - Are Mondays the slowest days because of the weekend backlog?
    - Do minor holidays affect response time?
- Obtain census data to gain insight more into zip codes, neighborhoods, and demographics beyond just the large districts.
- Determine priority level for each call as a feature based on the number of days given and department to explore if there is a correlation with the level of delay.

**We recommend...**
  
- The City of San Antonio should create standardized timelines for each department to follow when solving cases.
- Animal Care Services and Customer Service should both have a thorough review of their cases and timelines to rectify latency issues.
- Late and extremely late cases should be investigated through all departments.
- The classification in the raw data set for whether a case was completed late or not needs to be re-made. This is due to an issue where this feature classifies cases as being late when they were completed as late. For example if a case was due in fifteen days but was completed a day before its due date, it would be classified as late.


</details>  


## <a name="Recreate This Project"></a> 
![recreate](https://github.com/3-1-1-Codeup/project/blob/main/workbooks/caitlyn/images/read_me_take3/recreate.png?raw=true)
[[Back to top](#top)]

<details>
  <summary>Click to expand!</summary>

### 1. Getting started
- Start by cloning the github repository on your From your terminal command line, type: 
git clone git@github.com:3-1-1-Codeup/project.git

- Download .CSV of Data from the link below and name it as service-calls.csv in your working directory:
https://data.sanantonio.gov/dataset/service-calls/resource/20eb6d22-7eac-425a-85c1-fdb365fd3cd7

- Use the wrangle.py, explore.py, and model.py to follow the processes we used.
    
Good luck I hope you enjoy your project!

</details>
    
## <a name="team"></a>
![meet](https://github.com/3-1-1-Codeup/project/blob/main/workbooks/caitlyn/images/read_me_take3/meet.png?raw=true)

A big thank you to the team that made this all possible:

![team](https://github.com/3-1-1-Codeup/project/blob/main/workbooks/caitlyn/images/read_me_take3/team.png?raw=true)


>>>>>>>>>>>>>>>
.

