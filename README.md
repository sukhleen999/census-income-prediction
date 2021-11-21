# Census Income Prediction
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

- Author: Affrin Sultana, Navya Dahiya, Philson Chan, Sukhleen Kaur  

Data analysis project for Group 1 of DSCI 522 (Data Science Workflows), A course in the 2021 Master of Data Science program at the University of British Columbia.
## Project Description
In this project, we would focus on the [Census Income Dataset from UCI Machine Learning Repository](https://archive-beta.ics.uci.edu/ml/datasets/census+income). The project aims to carry out a binary classification to predict whether the annual income of the person is above USD 50K or not, by analysing their demographic information.

In the dataset, various demographic features such as age, education level and marital status are considered. The training dataset consists of 32561 examples, while the testing set has 16281 rows, each consists of 14 features and 1 target column. In order to produce an unbiased predictor, we plan to carry out exploratory data analysis (EDA) with the training dataset. We would assess the distribution of various features and the target column, as well as the distribution of different features against the target column.  

After the EDA, we would create a pipeline to preprocess the dataset, as well as selecting suitable classification model to predict the target. We are planning to explore various types of classification model, including Random Forest classifier, K-nearest neighbor (KNN) classifier and logistic regression. To select the model with best performance, we would fine tune the hyperparameter of the models, and conduct cross validation to evaluate the models with various metrics, such as precision, recall, F1 Score and overall accuracy.

Finally, we would re-fit the best model with the entire training dataset and evaluate the model with the testing data. The table of evaluation metrics would be reported to reflect the performance of the predictive model on unseen data. A few follow-up discussions would also be carried out to analyse the importance of certain features as well as the social implication of the prediction model.

## Usage
### Download File
Training data:  
`python3 src/download_data.py https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data --out_dir=data/raw --file_name=train.csv`  
Testing data:  
`python3 src/download_data.py https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test --out_dir=data/raw --file_name=test.csv`

## Reference
Census Income. (1996). UCI Machine Learning Repository.