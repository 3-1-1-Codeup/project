Thank you, Caitlin.
### Acquire
We began with acquiring:

We obtained our main data from the City of San Antonio open data sets.
We chose the 3-1-1 department logs because we wanted to ensure that everyone throughout San Antonio has an equal opportunity for having their case solved on time.
We also got supplemental data regarding the demographics of each district from the San Antonio 2020 website.
The reason we chose this website as a data source is that the Department of Planning and Community Development published the site as a summary report of the demographic distributions of San Antonio using the US Census Bureau results.

# CLICK 
### Prepare
- To prepare our data we:
    
    - merged some data variables from reasons to calling to prevent misclassification and made case status a boolean variable. 
    - We also made new features, created the target variable level_of_delay
    	- The target variable is the level of delay measures if a case is resolved very early, early, on time, late, or very late. 
    - , handled the null values, and dropped any duplicated or unnecessary features.
    - Finally, we changed feature names to be easier to understand.
  
    - This was done through binning the comparison of the number of days used to the number of days allotted for a case to be completed.
    
    - The last step of our preparation was to split the data using SKLearn into the train, validate and test data sets. 
    - To be prepared for modeling, we dropped our object columns, scaled the numeric data using the SKLearn Preprocessing MinMax scaler, and split the data into X/Y data sets. 
    
 # Click
 ### Explore #1
- The next step in the pipeline we used was to explore:


    - Introduce districts
    - This photo taken frmo https://sa2020.org/city-council-profiles
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
    - Introduce departments
    

I am now going to pass the next stage of exploration to Sam.



review panel #1 feedback:
- why this matters
- tools used on slide
- bool to boolean 
- adding level of delay rather than target variable
- adding we drop unnecessary columns
- too much for a presentation
- 1. very early response is most prevalent bar chart for one type of response highlighted
- title to be what we get out of it
- 2. only shows very late response by district with each color with average line 
- Title: Does the “very-late response” vary by the district?
- 3. only very early with mean by district to compare to very late
