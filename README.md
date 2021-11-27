# Census Income Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

-   Author: Affrin Sultana, Navya Dahiya, Philson Chan, Sukhleen Kaur

Data analysis project for Group 1 of DSCI 522 (Data Science Workflows), A course in the 2021 Master of Data Science program at the University of British Columbia.

## Project Description

In this project, we would focus on the [Census Income Dataset from UCI Machine Learning Repository](https://archive-beta.ics.uci.edu/ml/datasets/census+income). The project aims to carry out a binary classification to predict whether the annual income of the person is above USD 50K or not, by analysing their demographic information.

In the dataset, various demographic features such as age, education level and marital status are considered. The training dataset consists of 32561 examples, while the testing set has 16281 rows, each consists of 14 features and 1 target column. In order to produce an unbiased predictor, we plan to carry out exploratory data analysis (EDA) with the training dataset. We would assess the distribution of various features and the target column, as well as the distribution of different features against the target column.

After the EDA, we would create a pipeline to preprocess the dataset, as well as selecting suitable classification model to predict the target. We are planning to explore various types of classification model, including Random Forest classifier, K-nearest neighbor (KNN) classifier and logistic regression. To select the model with best performance, we would fine tune the hyperparameter of the models, and conduct cross validation to evaluate the models with various metrics, such as precision, recall, F1 Score and overall accuracy.

Finally, we would re-fit the best model with the entire training dataset and evaluate the model with the testing data. The table of evaluation metrics would be reported to reflect the performance of the predictive model on unseen data. A few follow-up discussions would also be carried out to analyse the importance of certain features as well as the social implication of the prediction model.

## Usage

All the following command shall be executed at the root directory of this repository.

### Download File

Training data:  
`python3 src/download_data.py https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data --out_dir=data/raw --file_name=train.csv`  
Testing data:  
`python3 src/download_data.py https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test --out_dir=data/raw --file_name=test.csv`

### Data Cleaning

`python3 src/clean_data.py data/raw/train.csv data/raw/test.csv --out_dir=data/clean --train_filename=clean_train.csv --test_filename=clean_test.csv`

### Model Building

`python3 src/model_building.py data/preprocessed/clean_train.csv --out_dir=artifacts/model/ --output_model=model.pickle`

### Model Evaluation

`python3 src/model_evaluation.py data/preprocessed/clean_train.csv data/preprocessed/clean_test.csv artifacts/model/model.pickle --out_dir=artifacts/eval/`

## Dependencies

The dependencies for this project are mentioned in the `census-income.yaml` environment file in the directory of this project

-   Python 3.9.7 and Python packages:

    -   Python, version 3.7.0
    -   numpy==1.21.4
    -   seaborn, version 0.9.0
    -   pandas==0.24.2
    -   scikit-learn>=1.0
    -   altair==4.1.0
    -   altair_saver
    -   seaborn==0.8.1
    -   docopt==0.6.2
    -   matplotlib==3.5.0

-   R version 4.1.1 and R packages:

    -   knitr==1.26

    -   tidyverse==1.2.1

        ## License

The Census Income Prediction materials here are licensed under the MIT License. If re-using/re-mixing please provide attribution and link to this webpage.

## Reference

Census Income. (1996). UCI Machine Learning Repository. Click [here](https://archive-beta.ics.uci.edu/ml/datasets/census+income) to access it
