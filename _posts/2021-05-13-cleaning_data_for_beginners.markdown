---
layout: post
title:      "Cleaning Data for Beginners"
date:       2021-05-13 14:58:05 +0000
permalink:  cleaning_data_for_beginners
---


The hardest thing for any new Data Scientists to comprehend is cleaning their data. For me, it was something that I couldn't really grasp the power of until after I ran my first model and it looked absolutely terrible. I had thought I had did everything right, but I realized that I skipped over so many important cleaning and processing techniques such as feature creation. I learned the hard way that the most important part of creating a good model is to first ensure that the data you are feeding it is processed.

In this tutorial I will go through some data cleaning processes that will help those who are new to working with datasets using python. I will be working with Kaggle's Titanic dataset and only going through the cleaning process - not the modeling process. Kaggle's Titanic dataset is designed for those who are new to coding in python.

## Investigating the Dataset
It's important to first import the datasets you are going to use for data processing. In my case, I'm only importing libraries that will alow to manipulate the data better during the cleaning process, as well as create graphics in order to better visualize the data.

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```

Next, import the CSV file and then use the .info() method to get a snapshot of the data.
```
titanic_df = pd.read_csv('train.csv')
```
![Data snapshot](https://github.com/PNarducci1690/Flatiron-Blogpost-Cleaning-Data/blob/main/titanic_data.PNG)

Using the .info() method allows me to get a quick snapshot of the data. I can see that it contains 12 columns with two categorized as a float data type, 5 as an int data type, and 5 as an object data type. I can see that each column is supposed to contain 891 entries, however I can see that some columns are missing values. These columns are Cabin, Age, and Embarked. Now let's print the first the few rows of each column in order to see what information is contained there.

![Table](https://github.com/PNarducci1690/Flatiron-Blogpost-Cleaning-Data/blob/main/titanic_table.PNG)

In order for these columns to make sense I should explain them.

PassengerID is a generic number that provides a value for each row and should be dropped from the dataframe because it has no imortance for our model
Survival is our target variable in this dataset, One means the passenger survived and zero means that they did not.
Pclass means what class the passenger was in during the trip. These classes range from 1st clas to 3rd class.
The Name column represents the name of the passenger.
Sex column is passengers sex designation.
Age column provides the passengers age.
SibSP contains the number of sibings or spouses the passenger had aboard the titanic
Parch contains the number of parents or children the passenger had aboard the titanic
Ticket is the passengers ticket number.
Fare is how much it cost the passenger for passage on the Titanic.
Cabin is the passengers cabin number.
Embarked is the port at which the passenger boarded the Titanic.
Now that we have an understanding of each column, let's move onto the cleaning process

## Cleaning the Data
First, let's ensure consistency in the data set by lowercasing the column titles and string data
```
titanic_df.columns = map(str.lower, titanic_df.columns)

titanic_df = titanic_df.apply(lambda x: x.str.lower() if(x.dtype == 'object') else x)
```

Now, I will remove the columns that hold no significance. These columns are:

passengerid since it is just a generic number used to label the row entry
ticket since a randomly generated ticket number should not determine whether a passenger survived the voyage
cabin because there are way too many missing values.
For now, I will leave it only at these columns. Do you have to remove these columns? no. Is it always good to remove columns? Not necessarily. You should always look at your data and come to an understanding about it. Based upon my analysis I feel that these columns will not have a significant impact on survivability. However, others may feel differently and may have even come up with a way to utilize these columns in order to fine new information. It's important to figure out what you are trying to determine before you start dropping data. In my case, I am solely focused on survivability of the passenger.

```
titanic_df = titanic_df.drop(titanic_df[['passengerid', 'ticket', 'cabin']], axis=1)
```
Great now we have 2 columns that have missing values that I need to work with. Let's take a look at these columns

### Age
![Age histogram](https://github.com/PNarducci1690/Flatiron-Blogpost-Cleaning-Data/blob/main/age_histogram.PNG)
Clearly, most passengers ages fall between late teens and early 30s. We can also see that the number of passengers decreases as the age increases, with very few elderly people on the Titanic. However, there seems to be a pretty substanital population of children on the Titanic suggesting that most passengers were families. Now what are the best ways to deal with the missing age values?

#### Using mean or median
One method that is a common practice when replacing missing values is to use the mean or median values. The mean refers to the average age within the dataset, while the median is the more centralized value of the dataset. Which one should be used? That depends on preference as well as outliers. If there are outliers that can skew the mean value, then sometimes its better to go with the median. However, I like to look at both values first in order to see how close in proximity they are. If there is not a substantial difference between the mean and median then picking either is okay.

![mean and median](https://github.com/PNarducci1690/Flatiron-Blogpost-Cleaning-Data/blob/main/age_median_mode.PNG)

We can see that the mean value is 30 (when we round up) and the median value is 28. The mean value is slightly higher because of the outlier ages in the 60 to 80 range. However there is no significant difference between these values and either can be picked. For my purposes I'll stick with the mean value. Now, I will replace the NaN values with 30.
```
titanic_df['age'].replace(np.NaN, 30.0)
```

#### Using other methods to determine age

However, mean and median is not the only way to replace NaN values. Using mean or median generalizes the ages and may have you mislabeling ages. Maybe that passenger was a child, or an elderly individual. These are the risks we take when using a generalized value to describe a column. Is it a bad thing to do? No, not at all. Sometimes it's the only choice we have. In this scenario thought we could use the passengers name titles, such as Master and Miss, to group passengers into categories and use that average age in order to get a more precise age for each passenger.
```
titanic_df[titanic_df.name.str.contains('master')]['age'].mean()
```
Looking at this subset I can see that the average age is 5. Let's replace these ages with the mean value.
```
master = master.replace(np.NaN, 5)
```
Great! The passengers designated with Master had their ages replaced with a mean of 5, which is far more accurate than a mean of 5. Let's add these updated ages back to the dataframe and then replace the rest of the values with the mean from earlier. Remember, it is possible to explore these ages deeper and detrmine more accurate ages for each title, but the purpose of this blog post is to just show some examples of what can be done during the cleaning process.

### Embarked
Embarked refers to the location where the passenger departed from on the Titanic. Let's look at the value counts and see how we can clean this column
```
titanic_df['embarked'].value_counts()
```
I can see that majority of the passengers left from Southampton. Since it is only two missing values we can assume that those passengers left from Southampton. Let's replace those two missing values with s.
```
titanic_df['embarked'] = titanic_df['embarked'].replace(np.NaN, 's')
```

![cleaned dataset](https://github.com/PNarducci1690/Flatiron-Blogpost-Cleaning-Data/blob/main/cleaned_dataset.PNG)
Great! All our columns have been cleaned

## Final Thoughts 
As you can see, determining how to clean data takes time. Figure out what you want from this column and then determine the best way to fill those missing values. This dataset was very simple to clean, but with bigger datasets you may be able to use other columns in order to create new features. For example, in a dataset about homes you may have the year the home was built and the year that the information in the dataset was collected. You could subtract those two pieces of information in order to create a completely new feature called age. The things you can make with data are only limited by your creativity. 

