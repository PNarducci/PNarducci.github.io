---
layout: post
title:      "Tweet Product Sentiment Using NLP"
date:       2021-02-13 03:30:41 +0000
permalink:  tweet_product_sentiment_using_nlp
---


![Negative Sentiment Word Cloud](https://raw.githubusercontent.com/PNarducci1690/Proj_4_Twitter_Sentiment/main/images/neg_word_cloud.PNG)

## Introduction
Over the past decade the impact of social media has expanded from its beginnings as a means to interact with friends and family. Websites like Facebook, YouTube, Instagram, and Twitter are now a means by which consumers can interact directly with their favorite brands, and vice versa. As a business in the 21st century using social media is a must due to ability to not only build your brand quickly, but creating consumer retention by communicating with your audience daily with just a click of a button. According to [Forbes](https://www.forbes.com/sites/forbescommunicationscouncil/2018/05/11/how-social-media-can-move-your-business-forward/?sh=569360cd4cf2), roughly 53% of the world's population is connected to the internet, with roughly 3.2 billion of the population active on social media.Forbes also goes on to mention that a 2017 survey conducted on 5,7000 marketers revealed that 69% had developed loyal fans for their brands.

This loyalty, or consumer retention is very important for a business - especially when using social media. As social organisms, we as a species build bonds or connections with others through interaction. A businesses goal, especially younger businesses seeking to grow, is to retain consumers well into the foreseable future. This is further refleted in the same Forbes article in which the author's company conducted a survey in which 60% of the respondents stated that they kept up to date with the businesses they followed on social media and 54% also believed that they believed that businesses that engaged with their clients and followers were more focused on providing a better service for the consumer.

Now, what does all of this have to do with my project? My project required me to use Natural Language Processing (NLP) on a dataset containing Tweets directed at the tech brands of Apple, Google, and Android. The tweets were categorized into sentiments, as well as brand and product types by humans. As I explored the dataset, I focused on two main ideas:

* How can I use tweets in order to determine the sentiment directed at a brand?
* How can I determine the true negative tweets within the data set in order to retain customer support?

However, before I could even begin to answer these questions I had to first process my data

## Data Cleaning

This data set came with many challenges that were very new and challenging for me as someone new to NLP. String datatypes are not easy for a computer to work with since their value is not computationally friendly. In order for the computer to process all of the tweets that my data set contained I had to convert these strings into a numerical value. However, in order to do that I had to first clean the text in order. For my purposes I'm going to discuss how to clean a dataframe containing a series of tweets in detail, but here is a list of some of the other processes I had to take in order to fully clean the data:

* Changing column titles to make them less wordy
* Dropping or replacing NaN values
* Creating functions that would read cleaned text in order to replace missing brand information

My data set was made up of three columns - tweets, brand, and sentiment. In order to properly determine sentiment I had to clean the tweets. I did this by importing the following packages:

```
import pandas as pd
import re
import string
import html
```

re stands for regular expression and is a python library that allows us to define a pattern of characters within text. Once identified you can easily remove or replace these patterns throughout the entire text with ease. In order to clean all the texts in one go within the tweet column I created a function called tweet_cleaner: 

```
def tweet_text_cleaner(tweet):
    tweet = html.unescape(tweet) #unescapes html code
    tweet = tweet.replace(u'\ufffd', '?') #removes unicode
    tweet = tweet.lower() #make string values lowercase
    tweet = re.sub('\d[a-z0-9]+', '', tweet) #remove numeric values and accompanying text
    tweet = re.sub('\d+', '', tweet) #remove remaining numeric text
    tweet = re.sub('@[a-z0–9]+', '', tweet) # remove @ and accompying text
    tweet = re.sub('#[a-z0–9]+', '', tweet) # Remove # and accompying text
    tweet = re.sub('rt[\s]+', '', tweet) # Removes rt
    tweet = re.sub('https?://[A-Za-z0-9./]+', '', tweet) # removes hyperlinks
    tweet = re.sub('https?://[A-Za-z0-9./]+', '', tweet) # removes hyperlinks
    tweet = re.sub('\{[a-z0–9]+}', '', tweet) #will remove {} and ac
    for key in replacers.keys():
        tweet = tweet.replace(key, replacers[key])
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    tweet = re.sub('\s+', ' ', tweet)

    return tweet
```

What tweet cleaner does is remove all string values that are unneeded in order for me to run my model. As can be seen, various punctuation types, @, #, numerical values, unicode, and certain twitter slang like rt (retweet) were removed because they hold no value for the model that will be produced. You can also see that some common  apostrophe words and slang words were replaced with there full meaning in order to produce better results. Overall, the cleaning of the tweets took a lot of time and research to do since re was very overwhelming to look at and use at first because of its tendency to look like gibberish. However there are some wonderful re cheat sheets that can be found on the internet. [Here](https://www.rexegg.com/regex-quickstart.html#anchors) is the one I used.
## Analysis

![Brand Sentiment](https://raw.githubusercontent.com/PNarducci1690/Proj_4_Twitter_Sentiment/main/images/product_sentiment.PNG)

After cleaning the data I had to figure out a way to visualize the sentiment in order to bettwe understand my dataset. The word cloud at the begginning of my blog post was one of many different visualisations I created in order to deeply explore my data. The word cloud depicted above is a representation of the words or phrases that can be found within the tweets with a negative sentiment. Within the word cloud I can see some key negative words such as "hate" or "design headaches" which add reason as to why a tweet may be negative. However, a word cloud is not accurate and some of the words present here could also be shared with the positive and neutral sentiment labeled tweets. After trying various methods I ended up using a scattertext. What scattertext essentially does is looks at words within the corpus and gives them a label from 1 to -1 based upon the sentiment of that word.

![Word Sentiment](https://raw.githubusercontent.com/PNarducci1690/Proj_4_Twitter_Sentiment/main/images/word_sentiment.PNG)

In my case, a word like cool is considered a highly positive word because of how it is situated on the chart (far upper left) and a word like anyone is considered negative because of its location to the far lower right. Overall, I was really happy with this because of how interactive this chart is and how useful it can be when giving value to string data.

## Model

After exploring the data it was time to create my models. In order to do this I had to vectorize the data. I tried three vectorizing methods - Count vectorizer, TF-IDF vectorizer, and Doc2vec. Once the words were vectorized I proceeded to try three different model types - Random Forest, Multinomial Naieve Bayes, and XGBoost while also trying SMOTE-NC (an oversampling method) and Randomized Undersampling (an undersampling technique) in order to fix the imbalances in the data set.
After running all of the models, the model that produced the best score (I used the f1 score due to the imbalances) was a Random Forest model that used count vectorizer tweets and SMOTE-NC to handle the imbalances

![Model](https://raw.githubusercontent.com/PNarducci1690/Proj_4_Twitter_Sentiment/main/images/model_classification_report.PNG)

However, despite the models f1 score of 71.34 it failed to represent the true negatives in my dataset. The reason I was focused on the tru negatives was the belief that a successful business will take into account the opinions of those consumers that they fail to meet the needs of. For a business to sustain itself it must retain consumers. This misrepresentation can also be seen in this data sets confusion matrix

![CF Matrix](https://raw.githubusercontent.com/PNarducci1690/Proj_4_Twitter_Sentiment/main/images/model_cf.PNG)


## Conclusion

Overall, NLP is a very powerful tool when done correctly. For me, it was a difficult task because of the learning curb that came with it. I feel that my model performed well to an extent, but could perform even better with more time and deeper insight into all the various topics and ideas I explored while working through this process.

In the future I would try the following:
* Create a neural network
* Remove more common words from the text
* Handle dimensionality
* Create more noise by translating the text to another language and then back to english

As I explore and enhance my knowledge of these methods I hope to one day build stronger sentiment models. NLP is powerful because in a tech heavy society ruled by social media a business can easily lose or gain traction based upon the click of a button. 
