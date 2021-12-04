# Census Income Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

-Author: Affrin Sultana, Navya Dahiya, Philson Chan, Sukhleen Kaur

Data analysis project for Group 1 of DSCI 522 (Data Science Workflows), A course in the 2021 Master of Data Science program at the University of British Columbia.

## Project Description

In this project, we would focus on the [Census Income Dataset from UCI Machine Learning Repository](https://archive-beta.ics.uci.edu/ml/datasets/census+income). The project aims to carry out a binary classification to predict whether the annual income of the person is above USD 50K or not, by analysing their demographic information.

In the dataset, various demographic features such as age, education level and marital status are considered. The training dataset consists of 32561 examples, while the testing set has 16281 rows, each consists of 14 features and 1 target column. In order to produce an unbiased predictor, we plan to carry out exploratory data analysis (EDA) with the training dataset. In case, we find missing values, we will work on wrangling the data to take care of them either during the EDA or while carrying out the pre-processing activity using Simple Impute function. We would assess the distribution of various features and the target column, as well as the distribution of different features against the target column.

We also plan on exploring the distribution of data across both the classes to check if the data is balanced or not. In case there is an imbalance, we may try to deal with it using various methods such as Oversampling or by hyper-parameter tuning of the `class_weight` parameter of the model that we are using.

After the EDA, we would create a pipeline to preprocess the dataset, as well as tuning a suitable classification model to predict the target. In particular we will be developing the pipeline mainly with the toolset provided by `scikit-learn`, while `altair` would be used for data visualization and different graphs on model evaluation.

In this project, we are planning use Random Forest classifier as our classification model. To yield the optimum model, we are going to select certain important features with Recursive feature elimination (RFE) algorithm, and fit these features in the model. To select the model with best performance, we would fine tune the hyperparameter of the models, and conduct cross validation to evaluate the models with various metrics, such as roc-auc score, precision, recall, F1 Score, overall accuracy.

Finally, we would re-fit the best model with the entire training dataset and evaluate the model with the testing data. We may evaluate and refit the best model using roc_auc score as the major metrics to optimise. Based on the results obtained, the table of evaluation metrics would be reported to reflect the performance of the predictive model on unseen data. A few follow-up discussions would also be carried out to analyse the importance of certain features as well as the social implication of the prediction model.

## Usage

### Download File

Training [data:\\](data:\){.uri} `python3 src/download_data.py https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data --out_dir=data/raw --file_name=train.csv`\
Testing [data:\\](data:\){.uri} `python3 src/download_data.py https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test --out_dir=data/raw --file_name=test.csv`

## Dependencies

The dependencies for this project are mentioned in the `census-income.yaml` environment file in the directory of this project

## License

The Census Income Prediction materials here are licensed under the MIT License. If re-using/re-mixing please provide attribution and link to this webpage.

## Reference

Census Income. (1996). UCI Machine Learning Repository. Click [here](https://archive-beta.ics.uci.edu/ml/datasets/census+income) to access it
