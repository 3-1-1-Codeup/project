<a name="top"></a>
![name of photo](https://github.com/3-1-1-Codeup/project/blob/main/workbooks/caitlyn/images/readme_images/title.png?raw=true)

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
___


## <a name="project_description"></a>
![desc](https://github.com/3-1-1-Codeup/project/blob/main/workbooks/caitlyn/images/readme_images/description.png?raw=true)
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
![plan](https://github.com/3-1-1-Codeup/project/blob/main/workbooks/caitlyn/images/readme_images/planning.png?raw=true)
[[Back to top](#top)]

<details>
  <summary>Click to expand!</summary>

### Projet Outline:
    
- Acquisiton of data:
    - Download CSV from the City of San Antonio website.
        - https://data.sanantonio.gov/dataset/service-calls/resource/20eb6d22-7eac-425a-85c1-fdb365fd3cd7 
    - Bring data into python
    - Run basic exploration
        - .info()
        - .describe()
        - .isnull()
        - .value_counts()
        - basic univariate
        - key take aways
- Prepare and clean data with python - Jupyter Labs
    - Set index
    - Drop features
    - Handle null values
    - Handle outliers
    - Merge some feature values together (only the ones that go with each other)
    - Rename
    - Create
    - Bin to create new categorical feature(s)
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
- Does the type of call in an area effect the level of response?
- Does the specific location effect the response time?
- Does category and department affect response time?
- Is there a link to which form of reporting is responded to quickest and slowest?

### Target variable
- `level_of_delay`
    - Made in the feature engineering step.
        - This feature takes the number of days a case was open (open-closed) and divided it by the number of days the case was given to be resolved (open-due) and calculates the percent of the allocated resolution time that was used


</details>

    
## <a name="findings"></a> 
![find](https://github.com/3-1-1-Codeup/project/blob/main/workbooks/caitlyn/images/readme_images/findings.png?raw=true)

[[Back to top](#top)]

<details>
  <summary>Click to expand!</summary>

### Explore:
- Department, call reason, and number of days given for a resolution were found to be major drivers of response time.
- District was a driver, but only when paired with department or call reason. 
    
    
### Stats
- Stat Test 1: 
    - Anova test
        - Null : "There is no difference in days before or after due date between the districts."
            - Reject the null
            
- Stat Test 2: 
    - Anova test
        - reject of accept null
    
- Stat Test 3: 
    - Chi Square
        - "The department hadling a call and the level of delay are independent from each other"
    
- Stat Test 4: 
    - Chi Square
        - "The reason for the call and the level of delay are independent from one another"
    
- Stat Test 5: 
    - Anova test
        - reject of accept null
    
- Stat Test 6: 
    - Anova test
        - reject of accept null
    

### Modeling:
- Baseline:
    - 77.2 %
- Models Made:
    - Logistic Regression
    - KNN
    - Decision Tree
    - Random Forest
    - SGD Classifier
    - Ridge CLassifier
- Best Model:
    - 
- Model testing:
    - 
- Performance:
    - 

***

    
</details>

## <a name="dictionary"></a>
![dict](https://github.com/3-1-1-Codeup/project/blob/main/workbooks/caitlyn/images/for_readme/dict.png?raw=true)

[[Back to top](#top)]

<details>
  <summary>Click to expand!</summary>

### Data Used
    
| Attribute | Definition | Data Type |
| ----- | ----- | ----- | 
| address | The address or intersection for the reported case/service requested. | object |
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
| latitude | The Y coordinate of the case reported. (longitude) | float64 |
| *level_of_delay |Level of delay based on days_before_or_after_due | object |
| longitude | 	The X coordinate of the case reported. (latitude) | float64 |
| num_of_registered_voters | Number of people registered to vote in that district | int64 | 
| open_date | The date and time that a case was submitted. | object |
| open_month | Month of the year the case was made | int64 | 
| open_week | Week of the year the case was made | int64 | 
| open_year | The year the case was made | int64 | 
| pct_time_of_used | How much of the resolution_days_due was the case open? | float64 | 
| per_capita_income | The income per capita in the district | int64 |
| resolution_days_due | The number of days between a case being opened and due. | float64 |
| source_id | The source id is the method of input from which the case was received. | object |
| square_miles | Square miles in the district | float64 |
| voter_turnout_2019 | How Many people showed up to vote in 2019 in that district | float64 | 
    
\*  Indicates the target feature in this Zillow data.

***
</details>

## <a name="acquire_and_prep"></a> 
![acquire_prep](https://github.com/3-1-1-Codeup/project/blob/main/workbooks/caitlyn/images/readme_images/acquire_prepare.png?raw=true)
[[Back to top](#top)]

<details>
  <summary>Click to expand!</summary>

### Acquire Data:
- Data was gathered from "The City of San Antonio" website
    - https://data.sanantonio.gov/dataset/service-calls/resource/20eb6d22-7eac-425a-85c1-fdb365fd3cd7
- Added data from the following website to create features such as per_capita_income, voter_turnout, etc.
    - https://sa2020.org/city-council-profiles
    
### Prepare Data
*All funcitons for the following preparation can be found in the wrangle.py file on our github repository.*
- Make case id the index
- Handle null values 
- Remove features that are not needed
- Create new features such as
    - days_open
    - resolution_days_due
    - days_before_or_after_due
    - pct_time_of_used
    - voter_turnout_2019
    - num_of_registered_voters
    - per_capita_income
- Create dumy columns for district
- Rename the features to make them easier to understand
- Merge some values that go hand in hand from reason for calling 
- Extract zip code from the address

***

</details>



## <a name="explore"></a> 
![dict](https://github.com/3-1-1-Codeup/project/blob/main/workbooks/caitlyn/images/for_readme/explore.png?raw=true)
[[Back to top](#top)]

<details>
  <summary>Click to expand!</summary>
    
### Findings:
- 

***

</details>    

## <a name="stats"></a> 
![stats](https://github.com/3-1-1-Codeup/project/blob/main/workbooks/caitlyn/images/for_readme/stats.png?raw=true)
[[Back to top](#top)]
<details>
  <summary>Click to expand!</summary>


### Stats Test 1:
- What is the test?
    - 
- Why use this test?
    - 
- What is being compared?
    - 

#### Hypothesis:
- The null hypothesis (H<sub>0</sub>) is...
    - 
- The alternate hypothesis (H<sub>1</sub>) is ...
    - 


#### Confidence level and alpha value:
- I established a 95% confidence level
- alpha = 1 - confidence, therefore alpha is 0.05

#### Results:
- Reject the null or fail to reject
- Move forward with Alternative Hypothesis or not 

- Summary:
    - F score of:
        - 
    - P vlaue of:
        - 

### Stats Test 2:
- What is the test?
    - 
- Why use this test?
    - 
- What is being compared?
    - 

#### Hypothesis:
- The null hypothesis (H<sub>0</sub>) is...
    - 
- The alternate hypothesis (H<sub>1</sub>) is ...
    - 


#### Confidence level and alpha value:
- I established a 95% confidence level
- alpha = 1 - confidence, therefore alpha is 0.05

#### Results:
- Reject the null or fail to reject
- Move forward with Alternative Hypothesis or not 

- Summary:
    - F score of:
        - 
    - P vlaue of:
        - 

***

    
</details>    

## <a name="model"></a> 
![model](https://github.com/3-1-1-Codeup/project/blob/main/workbooks/caitlyn/images/for_readme/model.png?raw=true)
[[Back to top](#top)]
<details>
  <summary>Click to expand!</summary>

Summary of modeling choices...
        
### Models Made:
- 

### Baseline Accuracy  
- 
      
| Model | Accuracy with Train | Accuracy with Validate |
| ---- | ----| ---- | ---- |
| Model | Accuracy with Train | Accuracy with Validate |
| Model | Accuracy with Train | Accuracy with Validate |
    
    
## Selecting the Best Model:

- 

- Why did we choose this model?
    - 

### Model on All Data Sets

| Best Model | Accuracy with Train | Accuracy with Validate | Accuracy with Test|
| ---- | ----| ---- | ---- |
| Model | Accuracy with Train | Accuracy with Validate | Accuracy with Test|


***

</details>  

## <a name="conclusion"></a> 
![conclusion](https://github.com/3-1-1-Codeup/project/blob/main/workbooks/caitlyn/images/for_readme/conc.png?raw=true)
[[Back to top](#top)]
<details>
  <summary>Click to expand!</summary>

We found....

With further time...

We recommend...


</details>  


## <a name="Recreate This Project"></a> 
![recreate](https://github.com/3-1-1-Codeup/project/blob/main/workbooks/caitlyn/images/readme_images/recreate.png?raw=true)
[[Back to top](#top)]

<details>
  <summary>Click to expand!</summary>

### 1. Getting started

    
Good luck I hope you enjoy your project!

</details>
    


## 

![Folder Contents](URL to photo)


>>>>>>>>>>>>>>>
.

