# Sentiment_Analysis_LR - Sentiment Analysis with Logistic Regression.

Data: NLTK tweeter corpus. A set of tweets labeled as being positive(1) or negative(0). Balanced data set with 50%/50% of positive and negative labels.

1. Text Pre-processing part.
Every tweet is pre-processed using 'prepro_tweet' function, so that it is being tokenized, stop words, handles and URL eliminated; stemmed and lower cased. Later on they are 
stored in the lists of pre-processed tokens.

2. The dictionary of positive and negative frequencies for each token has been created using the 'count_dict' function. In the dictionary the keys are the tuples of the token
and the label(sentiment) and the values are the counts of the times the token appears whithin the label. {('token', label): frequency}.

3. Feature Extraction part.
Using the 'extract_features' function every string in the tweet is being converted into a vector of dimention 3. Where the first value is 1 for the bias unit; the second is the
sum of the positive frequencies for every word (repeated included) in the string; the third is sum of the negative frequencies for every word (repeated included) in the string.
string => [1, positive_frequencies_sum, negative_frequencies_sum].

4. Logistic Regression uses a sigmoid function that converts an output of x0*theta0 + x1*theta1 + ... xn*thetan into a probability between 0 and 1. In the utils file there is
function called 'sigmoid' that will be applied in the training part.

5. Training Part.
To train the Logistic Regression using the 'gradientDescent' function we will iterate over the entire train data to find the optimum [theta0, theta1, theta2] that minimizes 
the cost function. The main steps are:
- initialize the parameter theta as [0, 0, 0];
- calculate the logits x0*theta0 + x1*theta1 + x2*theta2;
- using the 'sigmoid' function, calculate the probabilities;
- calculate the cost function;
- update the thetas in the gradient direction;
- repeat; after many iterations we arrive at the optimum cost and save the thetas parameter as the final model.

6. Test.
Applying the thetas on the unseen data evaluate how well the model generalizes using the 'accuracy_score' function.


# Sentiment_Analysis_LR - Sentiment Analysis with Naive Bias.

Data: NLTK tweeter corpus. A set of tweets labeled as being positive(1) or negative(0). Balanced data set with 50%/50% of positive and negative labels.

1. Text Pre-processing part.
Every tweet is pre-processed using 'prepro_tweet' function, so that it is being tokenized, stop words, handles and URL eliminated; stemmed and lower cased. Later on they are 
stored in the lists of pre-processed tokens.

2. The dictionary of positive and negative frequencies for each token has been created using the 'count_dict' function. In the dictionary the keys are the tuples of the token
and the label(sentiment) and the values are the counts of the times the token appears whithin the label. {('token', label): frequency}.

3. Train Naive Bias

Given a freqs dictionary, train_x (a list of tweets) and a train_y (a list of labels for each tweet), we are going to implement a naive bayes classifier using 'train_naive_bayes' which returns the logprior and log likelihood dictionary of words and their likelihoods as its keys and values respectively.

To calculate logprior we need to calculate the number of documents (tweets) 𝐷 , as well as the number of positive documents (tweets) 𝐷𝑝𝑜𝑠 and number of negative documents (tweets) 𝐷𝑛𝑒𝑔 . Calculate the probability that a document (tweet) is positive 𝑃(𝐷𝑝𝑜𝑠) , and the probability that a document (tweet) is negative 𝑃(𝐷𝑛𝑒𝑔). The logprior is 𝑙𝑜𝑔(𝐷𝑝𝑜𝑠)−𝑙𝑜𝑔(𝐷𝑛𝑒𝑔)

To compute compute the loglikelihood for each word we use the formula 𝑙𝑜𝑔(𝑃(𝑊𝑝𝑜𝑠)/𝑃(𝑊𝑛𝑒𝑔)) Where 𝑃(𝑊𝑝𝑜𝑠) and 𝑃(𝑊𝑛𝑒𝑔) are positive and negative probability of each word that could be calculated using the equations:

𝑃(𝑊𝑝𝑜𝑠)=(𝑓𝑟𝑒𝑞𝑝𝑜𝑠+1)/(𝑁𝑝𝑜𝑠+𝑉) 𝑃(𝑊𝑛𝑒𝑔)=(𝑓𝑟𝑒𝑞𝑛𝑒𝑔+1)(𝑁𝑛𝑒𝑔+𝑉)

Where 𝑉 is the number of unique words that appear in the freqs dictionary regardless of the label. 𝑁𝑝𝑜𝑠 and 𝑁𝑛𝑒𝑔 are the total number of positive words and total number of negative words; 𝑓𝑟𝑒𝑞𝑝𝑜𝑠 and 𝑓𝑟𝑒𝑞𝑛𝑒𝑔 are the positive and negative frequency of each word 𝑓𝑟𝑒𝑞𝑝𝑜𝑠 and 𝑓𝑟𝑒𝑞𝑛𝑒𝑔.

4. Test the naive bayes model
Implement naive_bayes_predict function that takes in the tweet, logprior, loglikelihood. and returns the likelihood that the tweet belongs to the positive(>0) or negative class(<=0). For each tweet it sums up loglikelihoods of each word in the tweet and then adds the logprior to this sum to get the predicted sentiment of that tweet.

𝑝=𝑙𝑜𝑔𝑝𝑟𝑖𝑜𝑟+∑𝑖𝑁(𝑙𝑜𝑔𝑙𝑖𝑘𝑒𝑙𝑖ℎ𝑜𝑜𝑑𝑖)

To test the accuracy of our algorythm we will implement test_naive_bayes function. The function takes in your test_x, test_y, log_prior, and loglikelihood and returns the accuracy of our model.

5. Error Analysis
This part shows some tweets that the model missclassified. Where some epty tweets were found.

6. Filter words by Ratio of positive to negative counts.
Some words have more positive counts than others, and can be considered "more positive" others more negative. To define the level of positiveness or negativeness we are going to compare the positive to negative frequency of the word. And with the get_words_by_threshold function we can calculate the ratio of positive to negative frequencies of a word using get_ratio function and filter a subset of words that have a minimum or maximum ratio of positivity / negativity or higher/lower. For example the most negative ratios are in the tokens like '':(', ''♛' and ':-('; and the most positive are ':)', ':D' and ':-)', meaning we should be careful removing punctuation characters and try to preserve the 
ones that represent emptions.

7. Visualizing Naive Bayes
For each tweet, we have calculated the likelihood of the tweet to be positive and the likelihood to be negative. We have calculated in different columns the numerator and denominator of the likelihood ratio introduced previously. Then we plot the numerator and denominator of the likelihood ratio for each tweet. Here we can clearly see that the absolute majority of negative tweets have greater absolute value of denominator and the positive tweets have greater absolute value of numerator.

8. Use the confidence ellipse to understand the Naïve Bayes model even better.

A confidence ellipse is a way to visualize a 2D random variable. It is a better way than plotting the points over a cartesian plane because, with big datasets, the points can overlap badly and hide the real distribution of the data. Confidence ellipses summarize the information of the dataset with only four parameters:
Center: It is the numerical mean of the attributes
Height and width: Related with the variance of each attribute. The user must specify the desired amount of standard deviations used to plot the ellipse.
Angle: Related with the covariance among attributes.
The parameter n_std stands for the number of standard deviations bounded by the ellipse. In this case there is 2 and 3 standard deviations (95% and 99.7%)

# Identifying_hate_tweets(Unbalanced_Data)

The data was downloaded from the Kaggle competition web page: https://www.kaggle.com/imene0swaaaan/twitter-sentiment-analysis containinf hate and racist tweets to be identified.

Every tweet is pre-processed using 'prepro_tweet' function, so that it is being tokenized, stop words, handles and URL eliminated; stemmed and lower cased. Later on they are 
stored in the lists of pre-processed tokens.

Naive Bias and Logistic Regression models were applied to classify tweets as hate and normal ones. 

As the data set is extremely imbalanced with only 7% of hate tweets, the oversampling method was performed and the minority class was  increased up to 33%.

Finally both models were applied on the original/imbalanced data set and modified data. Using F1 scores and data visualization some conclusions on how the imbalanced data can affect the performance of the model were made.

# Comments cosine similarity TF-IDF from scratch.

Comments people left at the restaurant in corpus data set are vectorized using TF-IDF technique. Using cosine similarity for the given comment n number most similar comments are found using vector cosine similarity technique.
