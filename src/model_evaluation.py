'''
This script aims to transform the cleaned data set to a ready-to-use data to be fed to the machine learning model, and save it to the file system

Usage: model_evaluation.py <input_train_file> <input_test_file> <input_model_file> --out_dir=<out_dir>

Options:
<input_train_file>                                   Path of cleaned train data file
<input_test_file>                                    Path of cleaned test data file
<input_model_file>                                   Path of trained model file
--out_dir=<out_dir>                                  Path to the output folder
'''

from docopt import docopt
import pandas as pd
import pickle
import os
import sys

opt = docopt(__doc__)

def load_model(filename):
    fp = open(filename, "rb")
    obj = pickle.load(fp)
    return obj

def upfront_transform(df):
    # Transform native_country into binary feature, indicating whether the sample comes from US or not
    df['native_country'] = df['native_country'] == 'United-States'

    # Set positive label to ">50K", which is the class with smaller proportion
    df['income'] = df['income'] == '>50K'
    return df

def main():
    try:
        if opt['<input_train_file>'] is None:
            raise TypeError("Argument input_train_file can't be None")
        if opt['<input_test_file>'] is None:
            raise TypeError("Argument input_test_file can't be None")
        if opt['<input_model_file>'] is None:
            raise TypeError("Argument input_model_file can't be None")
        if opt['--out_dir'] is None:
            raise TypeError("Argument out_dir can't be None")
    except Exception as e:
        print(e)
        sys.exit(1)
    
    # Create output directory if not exist
    if not os.path.exists(opt['--out_dir']):
        os.makedirs(opt['--out_dir'])

    # Read data and make some feature transformation
    train_df = pd.read_csv(opt["<input_train_file>"])
    train_df = upfront_transform(train_df)
    test_df = pd.read_csv(opt["<input_test_file>"])
    test_df = upfront_transform(test_df)

    # Split data into features & target
    X_train = train_df.drop("income", axis=1)
    y_train = train_df['income']
    X_test = test_df.drop("income", axis=1)
    y_test = test_df['income']

    rand_search_rf = load_model(opt["<input_model_file>"])

    print(rand_search_rf)

    # Evaluate Model with test data set
    y_pred_train = rand_search_rf.predict(X_train)
    y_pred = rand_search_rf.predict(X_test)

    # Table of Metrics for positive class for train and test set
    model_perf_df = pd.DataFrame(
        {
            "Accuracy": [accuracy_score(y_train, y_pred_train), accuracy_score(y_test, y_pred)],
            "Precision": [precision_score(y_train, y_pred_train), precision_score(y_test, y_pred)],
            "Recall": [recall_score(y_train, y_pred_train), recall_score(y_test, y_pred)],
            "F1 Score": [f1_score(y_train, y_pred_train), f1_score(y_test, y_pred)]
        },
        index=["Train Data", "Test Data"])

    # Export model performance dataframe
    model_perf_df_path = os.path.join(opt['--out_dir'], "model_performance.csv")
    model_perf_df.to_csv(model_perf_df_path)

    # Confusion Matrix for the test set

    test_confusion_matrix = pd.DataFrame(confusion_matrix(y_test, cross_val_predict(rand_search_rf, X_test, y_test)),
                columns = ['Predicted negative (0)', 'Predicted positive (1)'],
                index = ['True negative (0)', 'True positive (1)'])

    # Export confusion matrix
    confusion_matrix_path = os.path.join(opt['--out_dir'], "confusion_matrix.csv")
    test_confusion_matrix.to_csv(confusion_matrix_path)

    # Classification report for test set
    rand_search_rf.fit(X_train, y_train)

    y_pred = rand_search_rf.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["negative (0)", "positive (1)"], output_dict=True)
    classification_report = pd.DataFrame(report).transpose()

    # Export classification report
    classification_report_path = os.path.join(opt['--out_dir'], "classification_report.csv")
    classification_report.to_csv(classification_report_path)

    # Table of Metrics for train set
    PR_curve_df = pd.DataFrame(precision_recall_curve(y_train, rand_search_rf.predict_proba(X_train)[:,1],), index=["precision","recall","threshold"]).T
    PR_curve_df['F1 Score'] =  2 * (PR_curve_df['precision'] * PR_curve_df['recall'])/(PR_curve_df['precision'] + PR_curve_df['recall'])

    # Export PR curve dataframe
    PR_curve_df_path = os.path.join(opt['--out_dir'], "PR_curve_df.csv")
    PR_curve_df.to_csv(PR_curve_df_path)
    
if __name__ == "__main__":
    main()