Thank you Caitlin.


We began with acquire:

    - We obtained our main data from the City of San Antonio open data sets.
    - We chose the 3-1-1 department logs because we wanted to ensure that everyone throughout San Antonio has an equal opportunity for solving the case in a timely manner.
    -  We also got supplemental data regarding the demographics of each district from the San Antonio 2020 website. 
    - The reason we chose this website is because the Department of Planning and Community Development published the site as a summary report of the demographic distributions of San Antonio using the US Census Bureau results.

To prepare our data we:
    
    	- merged some data variables from reasons to calling to prevent misclassification and made case status a boolean variable. 
	- We also made new features, created the target variable level_of_delay, handled the null values and dropped any duplicated or unnecessary features.
	- Finally, we changed feature names to be easier to understand.
    
    - The target variable which is level of delay measures if a case is resolved very early, early, on time, late or very late. 
    - This was done through binning the comparison of the amount of days used to the amount of days allotted for a case to be completed.
    
    - The last step of our prepare was to split the data using SKLearn into train, validate and test data sets. In order to be prepared for modeling, we dropped our object columns, scaled the numeric data using the SKLearn Preprocessing MinMax scaler, and split the data into X/Y data sets. 
    
    
The next step in the pipeline we used was explore:


    We explored the train data through making univariate plots of zip code, days open, resolution days dues, days before and after due,  city council district and the target variable of level of delay. 
    Key findings from our univariate testing were:
      - Districts 1, 2, 3, & 5 have higher reports
      - District 0, 8, and 9 have far less early responses
      - but this is because the have called far less calls compared to all other districts
      - Early responses are far more common than any other level of delay
    We also created bivariate plots to visualize how the data variables are connected to each other. 
    Key findings from out bivariate testing were:
	    - each department has certain cases and areas where they perform really well but then their timeliness drops in other areas.
	    - the department with the most issues with level of delay were the customer service department. 

    We also ran statistical tests on our hypothesis questions. We ran three different hypothesis tests which were the ANOVA test, Chi2 test and Mann-Whitney U test. In all three tests, we were able to reject our null hypotheses and able to move forward in our investigation of our alternative hypotheses.

I am now going to pass the next stage of explore to Sam.
