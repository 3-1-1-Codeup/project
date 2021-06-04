Thank you, Caitlin.
### Acquire
- We began with acquiring:
- We obtained our main data from the City of San Antonio open data sets.
- We chose the 3-1-1 department logs because we wanted to ensure that everyone throughout San Antonio has an equitable opportunity for having their case solved on time.
- We also got supplemental data regarding the demographics of each district from the San Antonio 2020 website.
- We chose this website as a data source because the Department of Planning and Community Development published the site as a summary report of the demographic distributions of San Antonio using the US Census Bureau results.


# CLICK 
- You may be wondering why did we choose this topic?
- Imagine there is an aggressive loose dog on your street and you are unable to leave your house without being attacked.
- You call 3-1-1 and they say they will come pick up the dog.
- But hours turn into days and the dog is still there.
- So you start to wonder, is timeliness an issue on a departmental level or is this a larger issue only affecting my side of town?

# CLICK 
### Prepare
- To prepare our data we:
    
    - merged some data variables prevent misclassification and made case status a boolean variable. 
    - We also made new features and created the target variable which is level_of_delay
    	- Level of delay, measures if a case is resolved very early, early, on time, late, or very late. 
    	- This was done through binning the comparison of the number of days used to the number of days allotted for a case to be completed.
    
    - We handled the null values, and dropped any duplicated or unnecessary features.
    - We then changed feature names to be easier to understand.
  

    - The last step of our preparation was to split the data using SKLearn into the train, validate and test data sets. 
    - Then for modeling, we dropped our object columns, scaled the numeric data using the SKLearn Preprocessing MinMax scaler, and split the data into X/Y data sets. 
    
 # Click
 ### Explore #1
- The next step in the pipeline we used was explore:

    - There are 10 districts in San Antonio
    - Districts 1 and 5 along with a small portion of 7 make up the urban core.
    - District 2 and a portion of 10 makes up the east side and district 6 and a portion of 7 makes up the west side.
    - Districts 3 and 4 make up the South Side of San Antonio
    - Part of district 7 and 10 all of districts 8 and 9 make up the North side.
    - Some key findings about our districts are:
      - Districts 1, 2, 3, & 5 have a  higher amount of reports
      - District 8, and 9 have far fewer early responses
        - We attribute this to them having fewer calls compared to all other districts
# Click 
### Explore # 2
   - The departments for the City of San Antonio are Solid Waste Management, Development Services, Animal Care Services, 
Trans & Capital improvements, customer service, metro health, code enforcement services and parks/rec.
   - The department with the most calls is the Solid Waste Management which handles trash related calls.
   - The departments with the least amount of calls are parks and rec, customer service, and metro health.
   - Key findings from exploring departments were:
        -  each department has certain cases and areas where they perform well but then their timeliness drops in other areas.
        - the department with the most issues with the level of delay is the customer service department. 

I am now going to pass the next stage of exploration to Sam.

