---
layout: post
title:      "Is King County, Washington Affordable for Home Buyers?"
date:       2020-10-30 23:35:23 +0000
permalink:  is_king_county_washington_affordable_for_home_buyers
---


## Introduction

King County, Washington is considered one of the most expensive places to live in the United States when it comes to cost of living. A single adult needs to make roughly [$44,000](https://www.washington.edu/news/2020/10/15/for-single-adults-and-families-alike-higher-cost-of-living-in-all-washington-counties/) dollars a year in order to get by. Using a King County dataset from [Kaggle](https://www.kaggle.com/harlfoxem/housesalesprediction), I wanted to find out if the situation was as dire as what was being reported. I wanted to know if lower to middle class families could afford a home in this ever growing county in Washington state.

So, how did I go about solving this problem? Well, I decided to run a [Linear Regression](https://realpython.com/linear-regression-in-python/#simple-linear-regression) model using python on the King County (KC) data set. Linear Regression is being used because it allows us to take one or more features (our independent variable) and see if these features have a direct relationship on the target (our dependent variable). In our scenario, we will be using linear regression in order to determine if features, such as the year a home was built or how many bedrooms it contains, has an affect on the homes price (which is our target variable in this problem). Linear regression is also considered one of the easiest models to use. It works by weighing the best predicted weights between the target variable and features against their actual response with the target variable. In genral, we want our predicted estimate to be as close to the actual value between feature and target in order to reduce error and ensure that they do share a relationship.

## Exploring the Data
Before running the linear regression, I had to first clean my data. I did this by first importing all the libraries I would use throughout this project and then importing King County housing data into a Jupyter Notebook:
```
kc_data = pd.read_csv('kc_house_data.csv')
```
Once imported, I proceeded to clean the data. First thing I did was convert the date feature to a datetime data type and sqft_basement feature to a float data type. Once theses features were no longer object types, I could now work with the data in order to deal with the NaN values. First, the "?" values were converted into NaN values, and then the NaN values were changed accordingly in the following features:
```
#For sqft_basement, waterfront and view, replace missing values with 0
kc_data['sqft_basement'].fillna(0, inplace=True)
kc_data['waterfront'].fillna(0, inplace=True)
kc_data['view'].fillna(0, inplace=True)
#For yr_renovated, set yr_renovated to yr_built
kc_data['yr_renovated'].fillna(kc_data[kc_data['yr_renovated'].isna()]['yr_built'], inplace=True)
```
Once the data was cleaned, and the appropriate assumptions were made, I could now run a pairwise correlation in order to look for multicollinearity and drop features accordingly. I ended up dropping the following features:
```
new_kc_data = kc_data.drop(['sqft_above', 'id', 'date'], axis=1)
```
Then proceeded to place the continuous features and the categorical features into two separate features in order to work with them.

```
cont_data = new_kc_data[['price','bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'view', 'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long', 'sqft_living15', 'sqft_lot15']]
cat_data = new_kc_data[['waterfront', 'condition', 'grade', 'zipcode']]
```

## Working with Continuous Features

When looking at the continuous features, I noticed many outliers in many of my features. In order to fix this, I used the Z-score of my data, up to three standard deviations from the mean, in order to remove significant outliers from the dataset. I did the following:
```
z_cont = np.abs(stats.zscore(cont_data))
cont_data_out = cont_data[(z_cont < 3).all(axis=1)]
```

As you can see below, this is what the bedrooms data set looked like before using the Z-score:
![Bedrooms before Z-Score](https://raw.githubusercontent.com/PNarducci1690/project_2_Images/main/pre_Z_score_bedrooms.PNG)

and here is what it looked like after using the Z-score:
![Bedrooms after Z-Score](https://raw.githubusercontent.com/PNarducci1690/project_2_Images/main/post_Z_score_bedroom.PNG)

As you can see, the data has been altered to better reflect this feature.


I also checked for multicollinearity among my continuous features and ended up dropping sqft_lot15 since it would become problematic for my linear regression model.

## Working with Categorical Features

When working with the categorical features of my data set, I turned them into dummy variables. I did this because in order to properly run my linear regression model I needed these features to actually have a value that could be properly weighed by my model. In order to do this I did the following:

```
cond_dummies = pd.get_dummies(cat_data['condition'], prefix='cond', drop_first=True)
grade_dummies = pd.get_dummies(cat_data['grade'], prefix='grade', drop_first=True)
zc_dummies = pd.get_dummies(cat_data['zipcode'], prefix='zc', drop_first=True)
```

Then combined all the categorical features into one dataframe.

## Running a Linear Regression Model
In order to run my linear regression, I had to combine the categorical, continuous and target features into one feature. After doing this, I had to remove the remaining NaN values that were now present in this new dataframe due to the outliers that I removed when working with my categorical variables. Once the data was fixed, I ran an OLS model in order to check my r-squared value and to remove any features that had a p-value greater than .05. Once I remvoed those values I ended up with the following results:

![OLS Model](https://raw.githubusercontent.com/PNarducci1690/project_2_Images/main/OLS_results.PNG)

Our OLS Model shows a r-squared value of .813, also we have no features that have a p-value greater than .05. Now that the OLS assumptions have been made, we can now run our linear regression in order to see how well our model tests against data that is has not encountered before. We do this by creating a training set an a testing set and running a linear regression on both:

```
#First, we create our X and Y variables.
y = kc_data_fin['price']
X = kc_data_fin.drop('price', axis=1)

#We are going to use train_test_split from scikit in order to test our models performance.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = None, random_state=42)

#Then fit our training sets using LinearRegression() and make predictions for both our training and #testing models
linreg = LinearRegression()
model = linreg.fit(X_train, y_train)
y_hat_train = linreg.predict(X_train)
y_hat_test = linreg.predict(X_test)

#Then we take the MSE of our training and testing values and compare the results
train_mse = mean_squared_error(y_train, y_hat_train)
test_mse = mean_squared_error(y_test, y_hat_test)
```

After running the above code we get the following results:

```
Train Root Mean Squarred Error: 102636.12912855412
Test Root Mean Squarred Error: 103249.18688608424
R-squared: 0.8175346457180718
```

We notice that our testing set does very well against our training set, confirming that our model works when determining home price values in King County, Washington.

## Checking our Linear Regression Model Using Cross Validation

Next, in order to ensure that my linear regression model checks out ok, I ran a cross validation on the data. A cross validation, in essence, allows us to partition our data into multiple testing sets, a set amount of times, and then average out those predictions after running them against the training sets, in order to produce a more accurate value. In order to do this, I ran the following code:
```
mse = make_scorer(mean_squared_error)
cross_val= cross_val_score(LinearRegression(), X, y, cv=10, scoring=mse)

#Results
RMSE: 103570.01001904806
R-squared: 0.8175346457180718
```
As can be seen, our cross vaidation model works very well with our linear regression model, thus confirming that our model is very accurate. 

## Confirming Normality
Since OLS makes assumptions and assumes are model is correct, I had to test and see if my model was normally distributed. In order to do this, I ran a check to see how my risiduals were distributed in my model. I got the following results:

![Linear Assumption of Model](https://raw.githubusercontent.com/PNarducci1690/project_2_Images/main/Linear_Assumption.PNG)

![Distributions of Residuals](https://raw.githubusercontent.com/PNarducci1690/project_2_Images/main/Residual_Distribution.PNG)

![Checking for Homoscedasticity](https://raw.githubusercontent.com/PNarducci1690/project_2_Images/main/Homoscedasity.PNG)

As can be seen from these results, we see that, when looking at our models residiuals, there is normality. 

## So, Can a Family Afford to Live in King County, Washington?

According to our model, King County, Washington home prices are directly affected by the features in our model. But, is the county affordable for home buyers? Well, there are affordable homes as the following map shows: 

![Zipcode Affect on House Price](https://raw.githubusercontent.com/PNarducci1690/Project_2_KC_Housing_Data/master/King_County_Graphs/Zipcode_Price_Affect_On_Home_Price.PNG)

but, these homes are very far from the city, where most families moving to this area are hoping to work. And, as families move closer to the city, housing prices increase. Families are also further hurt by home size restrictions. According to the model, as the square footage of the home increases so does the price of the home. When looking at the following jointplot, most homes in King County range from 1000 to 2000 sqft. in size and can cost as $400,000 (or more) in value. That seems like a pretty expensive home for a very limited amount of space, especially families seeking to have children and raise a family in this area. 

![Sqft. Affect on Home Price](https://raw.githubusercontent.com/PNarducci1690/Project_2_KC_Housing_Data/master/King_County_Graphs/Sqft_living%20Affect%20on%20Home%20Price.PNG)

Overall, according to the model I produced, I would say the King County is a seller's market. Looking at the data, I noticed that if a home is renovated and receives a grade of 8 or higher from the hosing comittee of King County, the individual can look at a profit of about $500,000 or more. Which is great for sellers, but problematic for families seeking to purchase an affordable home in King County. Based upon my results, families moving to King County should look for a home that has no more than two floors, has about 1,500 sqft. to work with, and is not newely built, but a fixer upper. However, a fixer upper will lead to even more financial woes.

So, is King County affordable? Yes, and no. It depends on the families needs and wants when it comes to a home. A middle class family could make it, but will find that most of their money will be going towards their home and the family they are raising. As for a lower class family, this area will be very hard for them to remain finacially stable in. Families may have to work more than one job in order to make ends meet, which is problematic when it comes to raising a family. I would suggest that families take their time when looking in this region for a home, and weigh their finances. Purchasing a home is a great experience, but turning the home into a burden will create undo stress that working families in today's age don't need more of. Be patient, and be sure to find the home that best suits you right now - not the home that is just out of reach.
