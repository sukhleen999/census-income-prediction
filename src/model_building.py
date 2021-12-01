# author: Philson Chan
# date: 2021-11-25

'''
This script aims to transform the cleaned data set to a ready-to-use data to be fed to the machine learning model, and save it to the file system

Usage: model_building.py <input_train_file> --out_dir=<out_dir> [--output_model=<model_filename>]

Options:
<input_train_file>                                   Path of cleaned train data file
--out_dir=<out_dir>                                  Path to the output folder
[--output_model=<model_filename>]                    Output model file name, default "model.pickle"
'''

from docopt import docopt
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.compose import make_column_transformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import (
    cross_validate, RandomizedSearchCV
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder
)
import sys

opt = docopt(__doc__)

def main():
    try:
        if opt['<input_train_file>'] is None:
            raise TypeError("Argument input_train_file can't be None")
        if opt['--out_dir'] is None:
            raise TypeError("Argument out_dir can't be None")
        if opt['--output_model'] is None:
            raise TypeError("Argument output_model can't be None")
    except Exception as e:
        print(e)
        sys.exit(1)

    # Create output directory if not exist
    if not os.path.exists(opt['--out_dir']):
        os.makedirs(opt['--out_dir'])
    
    # Actual path to save the model
    output_filename = os.path.join(opt['--out_dir'], opt['--output_model'])

    # Read data and make some feature transformation
    train_df = pd.read_csv(opt["<input_train_file>"])
    # Transform native_country into binary feature, indicating whether the sample comes from US or not
    train_df['native_country'] = train_df['native_country'] == 'United-States'

    # Set positive label to ">50K", which is the class with smaller proportion
    train_df['income'] = train_df['income'] == '>50K'

    # Split data into features & target
    X_train = train_df.drop("income", axis=1)
    y_train = train_df['income']

    # Preprocessing
    numeric_feats = ['age', 'fnlwgt', 'hours_per_week']
    categorical_null_feats = ['workclass', "occupation"]
    categorical_nonull_feats = ["marital_status", "relationship"]

    binary_feats = ['sex', 'native_country']
    passthrough_feats = ['education_num']
    drop_feats = ['education', 'race', 'capital_gain', 'capital_loss']

    col_trans = make_column_transformer(
        (StandardScaler(), numeric_feats),
        (OneHotEncoder(sparse=False, handle_unknown='ignore', drop=[np.nan] * 2), categorical_null_feats),
        (OneHotEncoder(sparse=False, handle_unknown='ignore'), categorical_nonull_feats),
        (OneHotEncoder(drop='if_binary'), binary_feats),
        ('passthrough', passthrough_feats),
        ('drop', drop_feats)
    )

    # Calculate Baseline Performances
    scoring = ['accuracy', 'precision', 'recall', 'f1']
    baseline_results = {}
    pipe_dummy = make_pipeline(
        col_trans,
        DummyClassifier()
    )
    baseline_results['DummyClassifier'] = pd.DataFrame(cross_validate(pipe_dummy, X_train, y_train, scoring=scoring)).mean()

    # Calculate Baseline Perfromance for Random Forest Classifier
    pipe_forest = make_pipeline(
        col_trans,
        RandomForestClassifier(random_state=522)
    )
    baseline_results['RandomForest_default'] = pd.DataFrame(cross_validate(pipe_forest, X_train, y_train, scoring=scoring)).mean()

    # Export baseline_results
    baseline_results = pd.DataFrame(baseline_results)
    baseline_results_path = os.path.join(opt['--out_dir'], "baseline_result.csv")
    baseline_results.to_csv(baseline_results_path)
    print(f"Baseline Result saved to {baseline_results_path}")

    # Hyperparameter Tuning
    param_dist = {
        "randomforestclassifier__class_weight": [None, "balanced"],
        "randomforestclassifier__n_estimators": [10, 20, 50, 100, 200, 500],
        "randomforestclassifier__max_depth": np.arange(10, 20, 2)
    }
    rand_search_rf = RandomizedSearchCV(pipe_forest, param_dist, n_iter=20, 
                                        random_state=952, scoring=scoring, refit="f1")

    print("Model Training In Progess...")
    rand_search_rf.fit(X_train, y_train)
    print("Model Training Done!")

    hyperparam_result = pd.DataFrame(
        rand_search_rf.cv_results_
    ).sort_values("rank_test_f1")[['param_randomforestclassifier__n_estimators',
                                        'param_randomforestclassifier__max_depth',
                                        'param_randomforestclassifier__class_weight',
                                        'mean_test_accuracy',
                                        'mean_test_precision',
                                        'mean_test_recall',
                                        'mean_test_f1',
                                        ]]
    # TODO: Export hyperparam_result
    hyperparam_result_path = os.path.join(opt['--out_dir'], "hyperparam_result.csv")
    hyperparam_result.to_csv(hyperparam_result_path)
    print(f"Hyperparameter Tuning Result saved to {hyperparam_result_path}")

    # TODO: Export the model
    fp = open(output_filename, "wb")
    pickle.dump(rand_search_rf, fp)
    print(f"Model saved to {output_filename}")

if __name__ == "__main__":
    main()
