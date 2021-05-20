<a name="top"></a>
![name of photo](url_to_photo)

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


## <a name="project_description"></a> Project Description
![desc](URL to photo)
[[Back to top](#top)]

<details>
  <summary>Click to expand!</summary>

### Description
- 

### Goals
- 
    
### Where did you get the data?
- 

Project Name: ___

explain the project for resume
    

</details>
    
    
## <a name="planning"></a> Project Planning
![plan](URl to photo)
[[Back to top](#top)]

<details>
  <summary>Click to expand!</summary>

### Projet Outline:
    
- Acquisiton of data
- Prepare and clean data with python - Jupyter Labs
    - Drop
    - Rename
    - Create
    - Dummies
    - Etc.
- Explore data:
    - What are the features?
    - Null values:
        - Are the fixable or should they just be deleted.
    - Categorical or continuous values.
    - Make graphs that show:
        - At least 2.
- Run statistical analysis:
    - At least 2.
- Modeling:
    - Make multiple models.
    - Pick best model.
    - Test Data.
    - Conclude results.
        
### Hypothesis
- 

### Target variable
- 

</details>

    
## <a name="findings"></a> Key Findings
![find](URL to photo)

[[Back to top](#top)]

<details>
  <summary>Click to expand!</summary>

### Explore:
- 
    
    
### Stats
- Stat Test 1: 
    - which test:
        - reject of accept null

            
- Stat Test 2: 
    - which test:
        - reject of accept null
    

### Modeling:
- Baseline:
    - 
- Models Made:
    - 
- Best Model:
    - 
- Model testing:
    - 
- Performance:
    - 

***

    
</details>

## <a name="dictionary"></a> Data Dictionary
![dict](URL to photo)

[[Back to top](#top)]

<details>
  <summary>Click to expand!</summary>

### Data Used
    
| Attribute | Definition | Data Type |
| ----- | ----- | ----- | 
| Category | This general category was developed to place 311 services in a high level category, different than their respective department. | object |
| CASEID | The unique case reference number is assigned by the 311 Lagan customer relationship management system. | int64 |
| OPENEDDATETIME | The date and time that a case was submitted. | object |
| SLA_Date | Every service request type has a due date assigned to the request, based on the request type name. The SLA Date is the due date and time for the request type based on the service level agreement (SLA). Each service request type has a timeframe in which it is scheduled to be addressed. | object |
| CLOSEDDATETIME | The date and time that the case/request was was closed. If blank, the request has not been closed as of the Report Ending Date. | object |
| Late (Yes/No) | This indicates whether the case has surpassed its Service Level Agreement due date for the specific service request. | object |
| Dept | The City department to whom the case is assigned. | object |
| REASONNAME | The department division within the City deaprtment to whom the case is assigned. | object |
| TYPENAME | The service request type name for the issue being reported. Examples include stray animals, potholes, overgrown yards, junk vehicles, traffic signal malfunctions, etc. | object |
| CaseStatus | The status of a case which is either open or closed. | object |
| SourceID | The source id is the method of input from which the case was received. | object |
| OBJECTDESC | 	The address or intersection for the reported case/service requested. | object |
| Council District | The Council District number from where the issue was reported. | int64 |
| XCOORD | 	The X coordinate of the case reported. (latitude) | float64 |
| YCOORD | The Y coordinate of the case reported. (longitude) | float64 |
| Report Starting Date | The start date range for the case open date for this extract file. | object |
| Report Ending Date | The end date range for the case open date for this extract file. | object |
  
    
\*  Indicates the target feature in this Zillow data.

***
</details>

## <a name="acquire_and_prep"></a> Acquire and Prepare Data
![acquire_prep](URL to photo)
[[Back to top](#top)]

<details>
  <summary>Click to expand!</summary>

### Acquire Data:
- 
    
### Prepare Data
- 

***

</details>



## <a name="explore"></a> Exploration
![dict](URL to photo)
[[Back to top](#top)]

<details>
  <summary>Click to expand!</summary>
    
- wrangle.py 

### Findings:
- 

***

</details>    

## <a name="stats"></a> Statistical Testing
![stats](URL to photo)
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
​
    
</details>    

## <a name="model"></a> Modeling
![model](URL to photo)
[[Back to top](#top)]
<details>
  <summary>Click to expand!</summary>

Summary of modeling choices...
        
### Models and R<sup>2</sup> Values:
- 

### Baseline Accuracy  
- 
    
### Model
Model Accuracy:  
    
### Model
Model Accuracy:  


## Selecting the Best Model:

- 
    
### Use Table below as a template for all Modeling results for easy comparison:

| Model | Accuracy with Train | Accuracy with Validate | Accuracy with Test|
| ---- | ----| ---- | ---- |
| Model | Accuracy with Train | Accuracy with Validate | Accuracy with Test|
| Model | Accuracy with Train | Accuracy with Validate | Accuracy with Test|


- Why did I choose this model?
    - 

## Testing the Model

- Model Testing Results
     - 


***

</details>  

## <a name="conclusion"></a> Conclusion
![conclusion](URL to photo)
[[Back to top](#top)]
<details>
  <summary>Click to expand!</summary>

I found....

With further time...

I recommend...


</details>  


## <a name="Recreate This Project"></a> Recreate the Project
![recreate](URL to Photo)
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

