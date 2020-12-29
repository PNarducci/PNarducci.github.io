---
layout: post
title:      "Pump It Up! Using Machine Learning to Determine Well Functionality"
date:       2020-12-29 00:02:57 +0000
permalink:  pump_it_up_using_machine_learning_to_determine_well_functionality
---

## Introduction

Tanzania is a country located in East Africa in a region known as the African Great Lakes region. Currently, Tanzania is facing a [water crisis](https://www.google.com/search?q=tanzania+water+crisis&oq=Tanzania+water+crisis&aqs=chrome.0.35i39i457j69i60.4710j0j4&sourceid=chrome&ie=UTF-8) in which 30 million people, in a country of 57 million people, go without clean water. Due to this, a competition is being held on [Driven Data ](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/) in order to determine functionality of the water wells throughout Tanzania. The purpose of this competition is to use data from Taarifa in order to improve maintenance operations of these wells within the country.

However, in order to determine the functionality of the wells, I had to work with various machine learning models that I was not entirely familiar with. This blogpost will explore the three machine learning models I decided to use - k-Nearest Neighbors, Random Forest, and XGBoost, discuss the results of the one that produced the best score on Driven Data, and discuss future methodologies that I could use in the future to improve this project. This blog post is created by a new coder, for new coders, who get

## Data Wrangling

In order to begin working with this dataset, I had to head over to Driven Data's [Data Download](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/data/) section and download the data files there. The data was then imported and broken into training and testing data frames:

```
tww_training_target = pd.read_csv('Data/Training_Set_Labels.csv')
tww_training_target = tww_training_target.drop('id', axis=1)

tww_training_feature = pd.read_csv('Data/Training_Set_Values.csv')
tww_training_feature = tww_training_feature.drop('id', axis=1)

tww_testing_feature = pd.read_csv('Data/Testing_Set_Values.csv')
tww_testing_feature = tww_testing_feature.drop('id', axis=1)

tww_testing_submission = pd.read_csv('Data/Testing_Submission_Labels/SubmissionFormat.csv')
tww_testing_submission = tww_testing_submission.drop('status_group', axis=1)
```

I then began cleaning the traning data features in order to prepare them for analysis with the various models I would be using. The trickiest part about this data set was fixing the dementionality issues that the categorical features would create once they were one-hot-encoded. In order to do this, I had to look at each repetitive categorical features value counts and determine which feature summarized the data the best. The features that were not used were dropped, along with the num_private feature because it contained mostly zeroes and there was no description on what data this feature was supposed to describe.
```
tww_training_feature = tww_training_feature.drop(['num_private', 
                                                  'region_code', 
                                                  'subvillage', 
                                                  'lga', 
                                                  'ward', 
                                                  'recorded_by', 
                                                  'scheme_name', 
                                                  'waterpoint_type', 
                                                  'source', 
                                                  'quantity_group', 
                                                  'water_quality', 
                                                  'payment', 
                                                  'extraction_type_group', 
                                                  'extraction_type', 
                                                  'management', 
                                                  'scheme_management', 
                                                  'scheme_name', 
                                                  'source_type', 
                                                  'funder'], axis=1)
```

After reducing the amount of features significantly, I then had to convert data types, remove NAN and zero values, and create new features. After exploring the data, It made sense to replace NAN types with the datasets mode. For example, if we take a look at the installer feature:
```
tww_training_feature['installer'] = tww_training_feature['installer'].fillna(tww_training_feature['installer'].value_counts().idxmax())
```
I ended up replacing the NAN values with the most common value count from the feature. Other features that had to be adjusted were permit and public meeting, which were bool data types that had their NAN values filled with True values, and then they were converted to integers so that my models could easily work with them. I also ended up creating a new feature called age_of_well by performing the following:
```
tww_training_feature['year_recorded'] = pd.to_datetime(tww_training_feature['date_recorded']).dt.year

tww_training_feature['construction_year'] = tww_training_feature['construction_year'].replace({0:1990})

tww_training_feature['age_of_well'] = tww_training_feature['year_recorded'] - tww_training_feature['construction_year']

tww_training_feature = tww_training_feature.drop(['date_recorded', 'construction_year', 'year_recorded'], axis=1)
```
In order to create this new feature, I had to convert date_recorded to datetime that only held the year, and then subtract the construction_year from the newly created year_recorded feature in order to feature engineer age_of_well. In order to ensure that this dataset contained consistency, I ended up assigning data with a year of 0 to an arbitrary year that did not exist within the dataset, which, in this case, was 1990 (my birth year!). The features construction_year, date_recorded, and year_recorded were then dropped from the dataset because I felt that age_of_well summarized these features intended purposes in a new and improved feature.

After cleaning the features that needed to be cleaned, I ensured that certain string features had their data set to lowercase values in order to create consistency amoung elements within the features. Another issue that was still present was the dimentionality of the categorical features. In order to do this, I took the elements within the feature that made up less than 1% of the feature and put them in a new category called other. In doing so, I greatly reduced the amount of dummy variables that my models would have to work with. How I performed this task can be seen in the example code below:
```
series_two = pd.value_counts(tww_training_feature['installer'])
mask = (series_two/series_two.sum() * 100).lt(1)
tww_training_feature['installer'] = np.where(tww_training_feature['installer'].isin(series_two[mask].index),'other',tww_training_feature['installer'])
```

Once all issues and cocnerns for this dataset were handled, I then one-hot-encoded my categorical values, and created a heatmap in order to look at colinearity among my continuos data sets. 
![Heatmap](https://raw.githubusercontent.com/PNarducci1690/Proj_3_Tanzanian_Water_Wells/main/images/Heatmap.png)

As can be seen, latitude and longitude had the highest correaltion value with the target value. However, no values had high colinearity with each other, therefore no categorical features were dropped from the dataset. Once the data cleaning was finished, I performed the same techniques on the testing features.
## The Winning Model
Now that the data had been thoroughly cleaned, it was time to see how it performed on the three models I wished to use: k-Nearest Neighbors, Random Forest, and XGBoost. I chose these machine learning methods because:
* I wanted to practice with them in order to gain a better understanding of them.
* Determined that they were the best models to use with a multiclassification data set.
* I also wanted to establish a learning process trail from what I felt was the least valuable model to the most valuable model in determing the best score for this data competition.

I'm not going to go into full detail here about each model type (that can be found [here](https://github.com/PNarducci1690/Proj_3_Tanzanian_Water_Wells/blob/main/Tanzanian_Water_Well_Project-Copy1.ipynb), but I will talk about some key methods I used in order to determine how I went about creating the best models I possibly could

### k-Nearest Neighbors

k-Nearest Neighbors does exactly what it sounds like it does - It classifies the unknown data point based upon the number of it's closest neighbors. The best score I was able to produce with this model was a score of .7822, but after playing around with the data so much, the final model score based on the current data set was .7587. In the future, I will probably run these models in different notebooks, since different features are weighed differently by each model. Some key things that I did in order to strengthen this model:
* used MinMaxScaler on the categorical features, and then ran PCA on them in order to reduce dimentionality.
* Used RandomizedSearchCV in order to determine the best hyperparameters for tuning the model.

![k-NN Classification Report and Confusion Matrix](https://raw.githubusercontent.com/PNarducci1690/Proj_3_Tanzanian_Water_Wells/main/images/kNN_classification_and_CF_matrix.PNG)

### Random Forest

Random Forest is a type of supervised learning that uses many decision trees in order to determine the best classifier for the model. It's a very popular and powerful machine learning technique that is very easy to manipulate and use due to scikit learns wonderful documentation on the method. The final model produced a score of .8081! A significant increase from the k-NN model I used. Some key things that I did in order to strengthen this model:
* Used RandomizedSearchCV in order to determine the best hyperparameters for tuning the model.

![RF Classification Report and Confusion Matrix](https://raw.githubusercontent.com/PNarducci1690/Proj_3_Tanzanian_Water_Wells/main/images/rf_classification_and_CF_matrix.PNG)

### XGBoost
XGBoost is another supervised learning model that uses and ensemble of decision trees, like Random Forest. However, it uses gradient boosting in order to strengthen weak classifiers, and produces very powerful results. My XGBoost model ended up producing a score of .8069. This was a success for me since I had difficulty fully grasping the power of XGBoost at first. Like Random Forest and XGBoost, I used RandomizedSearchCV in order to determine the best hyperparameters. If you plan on using XGBoost, the max_depth parameter is a very important parameter, so I suggest playing around with it.

![RF Classification Report and Confusion Matrix](https://raw.githubusercontent.com/PNarducci1690/Proj_3_Tanzanian_Water_Wells/main/images/XGBoost_classification_and_CF_matrix.PNG)

So, which model performed the best? Overall, Random Forest produced the best score on Driven Data, with a score of .8081, and will be the model that I discuss my results on. Which was an incredible score to achieve on my first ever data science competition since the leading score is .8294. However, XGBoost and Random Forest performed very similarly, but it's important to note that both models relied heavily on different features. Random Forest relied heavily on Latitude and Longitude, while XGBoost relied heavily on waterpoint_type_group_other and quantity_enough. It's interesting that one model relied heavily on geographical importance, while the other relied heavily on the water source itself. I would also like to point out that accuracy score in both of these models should be taken with a grain of salt since the target values contain multiclass imbalances. When looking at these models, the f1 score is a better estimate than the the accuracy score since it takes into account these imbalances during the scoring process. 

## Random Forest Results
The results of the Random Forest Model show that latitude and longitude were the two most important features when determining if a water well in Tanzania was functional, non-functional and functional, but needs repairs. 
![RF Important Features](https://raw.githubusercontent.com/PNarducci1690/Proj_3_Tanzanian_Water_Wells/main/images/Feature%20Importance.PNG)

In order to visualize this I ended up creating a folium map, however my first couple of maps didn't come out the way I wanted them to, so I ended up borowing a map constructed by fellow Flatironers ([click here](https://mmsubra1.medium.com/data-mining-the-water-table-with-folium-7db354d97154)) in order to produce the result I wanted.

![Well Functionality](https://raw.githubusercontent.com/PNarducci1690/Proj_3_Tanzanian_Water_Wells/main/images/LatLong_wells.PNG)

The map shows the functional mas as green, the functional, but need repairs wells as yellow, and the nonfunctional wells as red. What is interesting to note is that all of these wells are found in similar locations, and majority of the wells are found in the north half of the country. One issue with this map is that all it shows is their location, not wht they don't work. In order to determine why else these wells may be malfunctioning is by looking at the next most important features: the age of the well, the gps_heigh of the well, and population near the well.

### Age of Well
![Age and Well Functionality](https://raw.githubusercontent.com/PNarducci1690/Proj_3_Tanzanian_Water_Wells/main/images/well_age.PNG)

According to my results, the younger the well, the more functional it is. Non-funtional are the oldest wells, while wells that need repair are the second oldest. This makes sense, but it is also a point of concern since the functional wells are roughly 4 years younger than the wells that need repairs. Also, within 5 years, these functional wells will no longer be functional. It also brings up interesting questions such as:
* Does the year in which the well was constructed matter?
* Are the nonfunctional wells not funtioning because of their type of extraction

These are questions that I don't have an answer to, but will consider when I revisit this dataset.

### GPS Height of Well
![GPS Height and Well Functionality](https://raw.githubusercontent.com/PNarducci1690/Proj_3_Tanzanian_Water_Wells/main/images/GPS_height.PNG)

What is interesting to note here is that as a well increases in height, it is more functional. I don't have an answer for the reasoning as to why this is, but I wonder if these pumps are higher in elevation due to changes in precipitation or method of extraction.

### Population Near Well
![Population and Well Functionality](https://raw.githubusercontent.com/PNarducci1690/Proj_3_Tanzanian_Water_Wells/main/images/pop_well.PNG)

Population also has an affect on well functionality. The higher the local population the more functional a well is. This is either due to the fact that newer wells have ben built near these locations, or that more people have moved to where wells are functional.

Overall, my Randomforest results show that geography, population, and the amount of water within a well are determining factors as to why a well is functional. Since my Random Forest Model was able to account for 80% of the data, I believe it is a good model to use when determining well malfunction in Tanzania.
## Looking Ahead
Since I was under time restraints to finish this project, there were many methods that I wanted to try, but didn't have enough time to fully explore or manipulate to the best of my ability. I will discuss the following so that others, if they decide to use this blogpost, may try out these methods on the models that they are trying to produce.
* Using Catboost to mean encode my categorical features instead of one-hot-encoding them.
* Using SMOTE or cost-sensative learning to deal with multiclass imbalaces.
* Using permutation importance and/or Drop-Column importance in order to determine what features will help in producing a better model.
* Using pipelines in order to streamline my models.


