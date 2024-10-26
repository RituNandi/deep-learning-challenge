# deep-learning-challenge

# Alphabet Soup Funding Final Report

# Overview
This project aims to develop a binary classifier that can predict the likelihood of applicants achieving success in their ventures if they receive funding from Alphabet Soup. The project will utilize the features present in the given dataset and employ diverse machine learning methods to train and assess the model's performance. The objective is to optimize the model in order to attain an accuracy score surpassing 75%.


# Results

### Data Preprocessing

The model aims to predict the success of applicants if they receive funding. This is indicated by the IS_SUCCESSFUL column in the dataset which is the target variable of the model. The feature variables are every column other than the target variable and the non-relevant variables such as EIN and NAME. The features capture relevant information about the data and can be used in predicting the target variables, the non-relevant variables that are neither targets nor features will be dropped from the dataset to avoid potential noise that might confuse the model.
During preprocessing, I implemented binning/bucketing for rare occurrences in the APPLICATION_TYPE and CLASSIFICATION columns. Subsequently, I transformed categorical data into numeric data using `pd.get_dummies`. I split the data into separate sets for features and targets, as well as for training and testing. Lastly, I scaled the data to ensure uniformity in the data distribution.

### Compiling, Training, and Evaluating the Model

Initial Model: For my initial model, I decided to include 3 layers: an input layer with 80 neurons, a second layer with 30 neurons, and an output layer with 1 neuron. I made this choice to ensure that the total number of neurons in the model was between 2-3 times the number of input features. In this case, there were 43 input features remaining after removing 2 irrelevant ones. I selected the relu activation function for the first and second layers, and the sigmoid activation function for the output layer since the goal was binary classification. To start, I trained the model for 100 epochs and achieved an accuracy score of approximately 73%. There was no apparent indication of overfitting or underfitting.

### Optimization attempts

1. For my first optimization attempt, I used hyperparameter tuning. During this process, Keras identified the optimal hyperparameters, which include using the relu activation function, setting 9 neurons for the first layer, and assigning 9, 9, 9, 5, 1 and 9 units for the subsequent layers. As a result, the model achieved an accuracy score of 73.12%.
2. For my second optimization attempt, I removed the STATUS column and applied binning to the ASK_AMT column in the preprocessing phase. I created two bins: one for amounts equal to $5,000 and another for all other amounts in the ASK_AMT column. This decision was based on the observation of an imbalance in the dataset, which was identified through a histogram analysis. I kept  3 layers: I selected the tanh activation function for the first and second layers, and the sigmoid activation function for the output layer. By implementing these adjustments, I achieved an accuracy score of approximately 73.0%.
3. For my third optimization attempt, I used hyperparameter tuning again. I changed the min_value to 10, and max_value to 100 in hp.Int. I tried with higher ranges to see if larger neural networks perform better. Keras identified the optimal hyperparameters, which include using the relu activation function, setting 80 neurons for the first layer, and assigning 20, 90, 40, 80, 10 and 100 units for the subsequent layers. I achieved an accuracy score of approximately 73.6%.
4. In my final optimization attempt, I increased number of hidden layers to 4: I selected the sigmoid activation function for the three hidden layers, as well as for the output layer. I set 80 neurons for the first layer, and assigning 30 and 10 units for the subsequent layers. With this final optimization attempt, I was able to achive an accuracy score of almost 73.0%. 

After four attempts to optimize my model, I was not able to achieve a goal of 75% accuracy score.

# Summary
Given that I couldn't attain the target accuracy of 75%, I wouldn't suggest any of the models above. However, with additional time, I would explore alternative approaches like incorporating the Random Forest Classifier and experimenting with different preprocessing modifications and different optimizres. I believe that making changes to the dropout layers, trying out various activation functions, and adjusting the number of layers and neurons could also contribute to optimizing the model and achieving the desired goal of 75% accuracy.