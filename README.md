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



