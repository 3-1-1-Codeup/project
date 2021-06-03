Thank you, Caitlin.
### Acquire
We began with acquiring:

- We obtained our main data from the City of San Antonio open data sets.
- We chose the 3-1-1 department logs because we wanted to ensure that everyone throughout San Antonio has an equal opportunity for having their case solved on time.
- We also got supplemental data regarding the demographics of each district from the San Antonio 2020 website.
- The reason we chose this website as a data source is that the Department of Planning and Community Development published the site as a summary report of the demographic distributions of San Antonio using the US Census Bureau results.
- WHY DO WE CARE?
- Imagine there is an aggressive loose dog on your street and you are unable to leave your house without being attacked.
- You call 3-1-1 and they say they will come pick up the dog.
- But it's been days and the dog is still there.
- So you start to wonder, is it an issue on just my side of town or is just a department response time issue?

# CLICK 
### Prepare
- To prepare our data we:
    
    - merged some data variables from reasons to calling to prevent misclassification and made case status a boolean variable. 
    - We also made new features, created the target variable level_of_delay
    	- The target variable is the level of delay measures if a case is resolved very early, early, on time, late, or very late. 
    	- This was done through binning the comparison of the number of days used to the number of days allotted for a case to be completed.
    
    - We handled the null values, and dropped any duplicated or unnecessary features.
    - Finally, we changed feature names to be easier to understand.
  

    - The last step of our preparation was to split the data using SKLearn into the train, validate and test data sets. 
    - To be prepared for modeling, we dropped our object columns, scaled the numeric data using the SKLearn Preprocessing MinMax scaler, and split the data into X/Y data sets. 
    
 # Click
 ### Explore #1
- The next step in the pipeline we used was to explore:

    - This photo taken from https://sa2020.org/city-council-profiles
    - There are 10 districts in San Antonio
    - Districts 1 and 5 along with part of 7 make up urban core.
    - District 2 makes up the east side and district 6 makes up the west side.
    - Districts 3 and 4 make up the South Side of San Antonio
    - Part of district 7, all of districts 8-10 make up the North side.
    - Some key findings about our districts are:
      - Districts 1, 2, 3, & 5 have a  higher amount of reports
      - District 0, 8, and 9 have far fewer early responses
        - but this is because they have called far fewer calls compared to all other districts
# Click 
### Explore # 2
   - The departments for the City of San Antonio are Solid Waste Management, Development Services, Animal Care Services, 
Trans & Capital improvements, customer service, metro health, code enforcement services and parks/rec.
   - Overall the average amount of calls is 39,645 for all departments combined
   - The department with the most calls is the Solid Waste Management which handles trash related calls.
   - The department with the least amount of calls is parks and rec which maintains the city parks.
   - Key findings from exploring departments were:
        -  each department has certain cases and areas where they perform well but then their timeliness drops in other areas.
        - the department with the most issues with the level of delay was the customer service department. 

I am now going to pass the next stage of exploration to Sam.

