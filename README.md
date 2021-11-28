# Census Income Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

-   Author: Affrin Sultana, Navya Dahiya, Philson Chan, Sukhleen Kaur

Data analysis project for Group 1 of DSCI 522 (Data Science Workflows), A course in the 2021 Master of Data Science program at the University of British Columbia.

## Census Income Predictor

Here we attempt to build a classification model using the Random Forest Classifier algorithm which can use the census income data with demographic features such as level of education, age, hours dedicated to work, etc to predict whether a person’s annual income will be greater than 50K or not. Our classifier was able to correctly predict 13524 examples out of 16281 test examples. Our classifier performed fairly on unseen test data with an ROC AUC score of 0.89, indicating that it is able to distinguish the positive class (income > 50k) with 0.89 probability. Among the people whose income is actually >50K, we are able to predict 70% of them correctly and among all the people who earned more than 50K, we were able to predict 71% of them correctly. However, it incorrectly predicted 1042 examples as false positives. These kinds of incorrect predictions could lead people into believing that they can earn more than 50K by following some other career path which might not be favourable for them, thus we recommend continuing the study to improve this prediction model before it is put into production.

In the dataset we have used consists of  various demographic features such as age, education level and marital status are considered. The training dataset consists of 32561 examples, while the testing set has 16281 rows, each consists of 14 features and 1 target column. The data set used in this project is the Census Income Dataset, which is also known as the Adult dataset [@misc_census_income_20], and was created in 1996. It was sourced from the UCI Machine Learning Repository and the data was extracted by Barry Becker using the 1994 Census database and details of which could be found [here](https://archive-beta.ics.uci.edu/ml/datasets/census+income).



## Usage

All the following command shall be executed at the root directory of this repository.

### Download File

Training data:  
```python3 src/download_data.py https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data --out_dir=data/raw --file_name=train.csv``` 

Testing data:  
```python3 src/download_data.py https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test --out_dir=data/raw --file_name=test.csv```

### Data Cleaning

```python3 src/clean_data.py data/raw/train.csv data/raw/test.csv --out_dir=data/clean --train_filename=clean_train.csv --test_filename=clean_test.csv```

### EDA Script
```python3 src/eda_script.py data/clean/clean_train.csv --out_dir=results/eda/```

### Model Building
```python3 src/model_building.py data/clean/clean_train.csv --out_dir=results/model/ --output_model=model.pickle```

### Model Evaluation
```python3 src/model_evaluation.py data/clean/clean_train.csv data/clean/clean_test.csv results/model/model.pickle --out_dir=results/eval/```

## Dependencies

The dependencies for this project are mentioned in the `census-income.yaml` environment file in the directory of this project

-   Python 3.9.7 and Python packages:

      - ipykernel
      - matplotlib>=3.2.2
      - scikit-learn>=1.0
      - pandas>=1.3.*
      - requests>=2.24.0
      - graphviz
      - python-graphviz
      - pip
      - altair>=4.1.0
      - altair_data_server
      - altair_saver
      - docopt==0.6.2

-   R version 4.1.1 and R packages:

    -   knitr==1.26
    -   tidyverse==1.2.1

## License

The Census Income Prediction materials here are licensed under the MIT License. If re-using/re-mixing please provide attribution and link to this webpage.

## Reference

Census Income. (1996). UCI Machine Learning Repository. Click [here](https://archive-beta.ics.uci.edu/ml/datasets/census+income) to access it
A. Liaw and M. Wiener (2002). Classification and Regression by randomForest. R News 2(3), 18--22.
