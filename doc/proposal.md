# Census Income Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

-Author: Affrin Sultana, Navya Dahiya, Philson Chan, Sukhleen Kaur

Data Science project for Group 1 of DSCI 522 (Data Science Workflows), A course in the 2021 Master of Data Science program at the University of British Columbia.

## Project Description

In this project, we would focus on the [Census Income Dataset from UCI Machine Learning Repository](https://archive-beta.ics.uci.edu/ml/datasets/census+income). The project aims to carry out a binary classification to predict whether the annual income of the person is going to be above USD 50K or not, by analyzing their demographic information.

In the dataset, various demographic features such as age, education level and marital status are considered. The training dataset consists of 32561 examples, while the testing set has 16281 examples, each consists of 14 features and 1 target column. In order to produce an unbiased classifier, we plan to carry out exploratory data analysis (EDA) with the training dataset. In case, we find missing values, we will work on wrangling the data to take care of them either during the EDA or while carrying out the pre-processing activity using the `SimpleImputer` transformer. Furthermore, we would visualize and assess the distribution of various features and the target column, as well as the distribution of different features against the target column. We will also look at the class imbalance (if any) by visualizing the target column and determine how it can be handled for better results. We may try to deal with it using various methods such as oversampling/undersampling or by hyper-parameter tuning of the `class_weight` parameter of the model that we are going to use. Lastly, we will explore the correlation between numeric features and how their association impacts the target feature.

After performing EDA, we would create a pipeline to preprocess the dataset, as well as tune a suitable classification model to predict the target. In particular we will be developing the pipeline mainly with the toolset provided by `scikit-learn` and `altair` would be used for visualising different graphs while evaluating the model.

In this project, we are planning to use `Random Forest Classifier` as our classification model. To obtain the optimum model, we are going to select certain important features with `Recursive feature elimination (RFE)` algorithm, and fit the model with these. To select the model with best performance, we would fine tune the hyperparameters of the model, and conduct cross validation to evaluate model's performance with various metrics, such as `roc-auc` score, `precision`, `recall`, `F1` score, overall `accuracy`.

Finally, we would re-fit the best model with the entire training dataset and evaluate the model with the testing data. We may evaluate and refit the best model using `roc_auc` score as the major metric to optimise our model and to measure the ability of our classifier to distinguish between classes. Based on the results obtained, the table of evaluation metrics would be reported to reflect the performance of the model on unseen data. A few follow-up discussions would also be carried out to analyse the importance of certain features as well as the social implication of the prediction model.

## Usage

### Download File

Training data
```python3 src/download_data.py https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data --out_dir=data/raw --file_name=train.csv```

Testing data
```python3 src/download_data.py https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test --out_dir=data/raw --file_name=test.csv```

## Dependencies

The dependencies for this project are mentioned in the [environment file](https://github.com/UBC-MDS/census-income-prediction/blob/main/census-income.yaml) in the directory of this project

The steps to using the environment are given below:

Creating an environment  ```conda env create --file census-income.yaml```

Activate the environment  ```conda activate census-income```

Deactivate the environment  ```conda deactivate```


## License

The Census Income Prediction materials here are licensed under the MIT License. If re-using/re-mixing please provide attribution and link to this webpage.

## Reference

Census Income. (1996). UCI Machine Learning Repository. Click [here](https://archive-beta.ics.uci.edu/ml/datasets/census+income) to access it
