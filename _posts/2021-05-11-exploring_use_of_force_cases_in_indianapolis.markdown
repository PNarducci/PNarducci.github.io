---
layout: post
title:      "Exploring Use of Force Cases in Indianapolis"
date:       2021-05-11 17:53:15 +0000
permalink:  exploring_use_of_force_cases_in_indianapolis
---

## Introduction
On May 25, 2020, George Floyd, and African-American male was murdered by Indianapolis Metropolitan Police Officer Derek Chauvin. This incident sparked world-wide protests for police reform, especially in regards to policing tactics and how they are deeply connencted to systemic racism in many countries. Even though this became a global movement, my project will strictly focus on the Indianapolis Metropolitan Police Department (IMPD) and their incidents of use of force. This project's intent is not to put blame on the officers themselves, but to create a model that can help the city of Indianapolis to better train and serve their communities without causing them severe harm. 

The tragic events that led to the death of Geroge Floyd should be used as a lesson for all of us moving forward on how the narrative on policing should change in America. These are not isolated, random, or one-in-a-million events - many of the incidents are pre-meditated due to the racially biased, and combative training that recruits go through in order to become police officers. Examples of such incidents of racial profiling can be seen in the killing of Daunte Wright, who was pulled over for an air freshener on his review mirror - a tactic normally used to pull of people of color. The hope is, that by modeling these scenarios, cities throughout America and the world can prevent these incidents from occuring by removing the factors that allow these events to occur. 

## Cleaning a Realworld Dataset
The most rewarding part of this capstone project was getting the chance to pick and clean a dataset that I could potentionally encounter in an actual work setting. The IMPD Use of Force (UOF) dataset was extremely dirty and had many issues. Honestly, before deciding on this dataset, most policing datasets focused on UOF were extremely disorganized and not very well put together. I chose this dataset because it was far more transparent and contained more interesting and useful columns than it's UOF counterparts. Overall, this dataset was dirty, but contained established features that would be important when it came to inbestigating these situations.

In order to clean this data I had to first remove the duplicates in the dataset. A glaring issue with this dataset was that many events were listed multiple times and contained the same information. In order to create a more fluid dataset I had to remove these listings by performing the following:
```
uof_df = uof_df.drop_duplicates(subset=['incnum', 'citcharge_type', 'offnum'])
```
What this code does is remove duplicates by looking at incident number first, then charge type, followed by officer number. 

Next, I ended up dropping the following columns for the following reasons:
* objectid - a generic value used to label the entire row entry
* cit_weapon_type - contains all NaN values. Has no useful information
* incnum - a generic value used for the incident in question. May repeat due to duplication
* offnum - a generic value used to represent the officer involved in incident
* citnum - a generic value used to represent the civilian involved in the incident

At this point, my dataset now contains 30 columns that still need to be cleaned and processed so that my models can better process the data. However, I'm not going to go through each column in detail. The info regarding the columns can be found in this projects repo [repo](https://github.com/PNarducci1690/Capstone-Project-Use-of-Force). For now, I will focus on discussing the following:
* Creating the target variable
* Pulling latitude and longitude information from addresses
* Creating officer designation

### Use of Force Target Variable Creation
In order to run my model I hade to re-create the Use of Force target variable. This column originally contained the type of use of force that officers used, but due to the fact that would create issues if left unaltered, I decided to take the information in the dataset and place them into categories that the IMPD uses when a UOF incident occurs. These categories are physical, less lethal and lethal use of force. 

In order to replace these values, I looked at the value counts and then used the replace method in order to replace those terms based upon how they would be categorized by the IMPD. For example:
```
uof_df['uof_force_type'] = uof_df['uof_force_type'].replace('lethal-handgun', 'lethal').replace('pointing a firearm', 'lethal').replace('lethal-rifle', 'lethal').replace('lethal-vehicle', 'lethal').replace('lethal-shotgun', 'lethal')
```
Once this was done, I replaced the NaN values as physical since physical made up majority of the dataset. The method you see above is the method I took for most of the data in this dataset. There were numerous misspellings and categorizing the data made things easier, as well as reduced the dimensionality for when I would run the model. 

### Getting Coordinates from Addresses
This by far was the most time consuming and tedious task for the entire dataset. But, I'm glad I did it in order to create a latitude and longitude feature, because I believe it helped me in creating a more viable model. So how did I go about this? Well, first I had to create an address category. I did this by joining multiple columns that containded various pieces of address information
```
uof_df['address'] = uof_df["street_n"].map(str) + " " + \
uof_df["street_g"].map(str) + " " + \
uof_df["street"].map(str) + " " + \
uof_df["street_t"].map(str) + " " + \
uof_df["city"] + " " + 'indiana'
```
After removing any NaN values, I imported the geopandas library in order to get the latitude and longitude from the addresses. However, in order to run this effectively I had to chunck the addresses by 1000 listings per chunk. I did this because it would have taken forever to complete and many addresses would have been misread. Afterwards I had to go through each chunk and use the .replace() method in order to edit all the addresses that were mislabeled. Afterwards I took the latitude and longitude information and placed them in separate columns.

### Officer Designation
Officer designation was the hardest coulmn to figure out because it's column title did not tell me what this data was about. I can't stress how important it is for us as a society that works and manipulates data to label things not just for ourselves, but for those who will also be working with the data. After a lot of research I realised that the UDTEXT24a - UDTEXT24d columns contained information that told me what the officers position was in the IMPD. In order to create a new feature from this data I had to first join these columns together 
```
uof_df["uo_designation"] = uof_df.filter(like='udtext24').astype(str).apply(' '.join, 1)
```
After combining them, I then removed any NaN values and any reperitive words from the designation
```
uof_df["uo_designation"] = (uof_df["uo_designation"] .str.split().apply(lambda x: OrderedDict.fromkeys(x).keys()).str.join(' '))
```

## EDA
Exploring the data before running the model was very enlightening. Through exploration I was able to confirme certain things that I expected, like black men expriencing higher rates of force by white police officers than other races

![Use of Force by race](https://github.com/PNarducci1690/Capstone-Project-Use-of-Force/blob/main/capstone%20images/Civilian_force_type.PNG)

It was also interesting to see that from October to January, Civilians are more likely to experience lethal force at the hands of police officers

![Use of Force by Month](https://github.com/PNarducci1690/Capstone-Project-Use-of-Force/blob/main/capstone%20images/month_and_uof.PNG)

and that geographical location did not determine where you would expect certain types of use of force to occur. However, majority of them were clustered in the center of the city.

![Use of Force by Month](https://github.com/PNarducci1690/Capstone-Project-Use-of-Force/blob/main/capstone%20images/lat_long_uof.PNG)

## Results
Since this was a multi-class classification problem I decided to use the following models - k-Nearest Neighbors, Naive Bayes, Decision Trees, Random Forest, and XGBoost. I chose these models because they are better algorithms when working with non- binary data. After running all the models, XGBoost ended up performing the best. 
![XGBoost Results](https://github.com/PNarducci1690/Capstone-Project-Use-of-Force/blob/main/capstone%20images/xgboost_model_results.PNG)
XGBoost ended up with an f1 score of 93.52, which means that my model was able to explain 93% of what was being predicted - which was use of force type. I ran a confusion matrix in order to see how my model was classifying the use of force
![Matrix](https://github.com/PNarducci1690/Capstone-Project-Use-of-Force/blob/main/capstone%20images/xgboost_conf_matrix.PNG)
The model ran pretty well and classified things well, but the model may need to be adjusted slightly because it was mislabeling 506 less lethal incidents as physical. This is a concern because miss labelin more aggresive use of force tactics may detract from how vital it is that police reform occurs since the public may believe that these incidents are happening less frequently.

Overall, the model ran well and brought even more questions up, such as why did the model weigh canine bites, park rangers, and recruiting units more than say race or gender? These were unexpected, but in the future, I would like to explore this features in much greater detail because targeting this features may help greatly in reducing the amount of use of force the IMPD is performing on a daily basis. 

## Conclusion
Systemic racism and over policing are an issue in our society. Everyday, people - specifically black males - encounter police while going about their daily routines and experience situations in which officers escalate the scenario do to racial profiling. I hope this model can help the city of Indianapolis find better ways to train their officers so that these incidents can decrease within there city. There is no reason a person should fear someone whose position is deamed as one as protection and safety. 
