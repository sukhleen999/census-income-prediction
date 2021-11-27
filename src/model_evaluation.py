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
import altair as alt
import pickle
import os
import sys
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix,
    PrecisionRecallDisplay, roc_curve, roc_auc_score
)
from sklearn.model_selection import cross_val_predict
alt.data_transformers.enable('data_server')
alt.renderers.enable('png')
import altair_saver

opt = docopt(__doc__)

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
    test_df = pd.read_csv(opt["<input_test_file>"])

    # Transform native_country into binary feature, indicating whether the sample comes from US or not
    train_df['native_country'] = train_df['native_country'] == 'United-States'
    test_df['native_country'] = test_df['native_country'] == 'United-States'

    # Set positive label to ">50K", which is the class with smaller proportion
    train_df['income'] = test_df['income'] == '>50K'
    test_df['income'] = test_df['income'] == '>50K'

    # Split data into features & target
    X_train = train_df.drop("income", axis=1)
    y_train = train_df['income']
    X_test = test_df.drop("income", axis=1)
    y_test = test_df['income']

    # Load model
    fp = open(opt["<input_model_file>"], "rb")
    rand_search_rf = pickle.load(fp)

    # Evaluate Model with test data set
    y_pred_train = rand_search_rf.predict(X_train)
    y_pred = rand_search_rf.predict(X_test)

    # Get output of pred_proba of train and test data respectively
    y_pred_train_prob = rand_search_rf.predict_proba(X_train)[:,1]
    y_pred_test_prob = rand_search_rf.predict_proba(X_test)[:,1]
    ap_forest_train = average_precision_score(y_train, y_pred_train_prob)
    roc_forest_train = roc_auc_score(y_train, y_pred_train_prob)

    ap_forest_test = average_precision_score(y_test, y_pred_test_prob)
    roc_forest_test = roc_auc_score(y_test, y_pred_test_prob)


    # Table of Metrics for positive class for train and test set
    model_perf_df = pd.DataFrame(
        {
            "Accuracy": [accuracy_score(y_train, y_pred_train), accuracy_score(y_test, y_pred)],
            "Precision": [precision_score(y_train, y_pred_train), precision_score(y_test, y_pred)],
            "Recall": [recall_score(y_train, y_pred_train), recall_score(y_test, y_pred)],
            "F1 Score": [f1_score(y_train, y_pred_train), f1_score(y_test, y_pred)],
            "AP Score": [ap_forest_train, ap_forest_test],
            "ROC AUC Score": [roc_forest_train, roc_forest_test]
        },
        index=["Train Data", "Test Data"])

    # Confusion Matrix for the test set
    test_confusion_matrix = pd.DataFrame(confusion_matrix(y_test, y_pred),
                columns = ['Predicted negative (0)', 'Predicted positive (1)'],
                index = ['True negative (0)', 'True positive (1)'])

    # Export confusion matrix
    confusion_matrix_path = os.path.join(opt['--out_dir'], "confusion_matrix.csv")
    test_confusion_matrix.to_csv(confusion_matrix_path)
    print(f"Confusion Matrix saved to {confusion_matrix_path}")

    report = classification_report(y_test, y_pred, target_names=["negative (0)", "positive (1)"], output_dict=True)
    clf_report = pd.DataFrame(report).transpose()

    # Export classification report
    clf_report_path = os.path.join(opt['--out_dir'], "classification_report.csv")
    clf_report.to_csv(clf_report_path)
    print(f"Classification report saved to {clf_report_path}")

    # Table of Metrics for train set
    PR_curve_df = pd.DataFrame(precision_recall_curve(y_train, y_pred_train_prob), index=["precision","recall","threshold"]).T
    PR_curve_df['F1 Score'] =  2 * (PR_curve_df['precision'] * PR_curve_df['recall'])/(PR_curve_df['precision'] + PR_curve_df['recall'])
    
    # Threshold to get best F1 score
    max_f1_df = PR_curve_df.iloc[PR_curve_df["F1 Score"].idxmax()].to_frame().T
    best_thres = max_f1_df['threshold'].iloc[0]
    max_f1_df

    # PR curve with best threshold
    PR_curve = alt.Chart(PR_curve_df).mark_circle().encode(
        x="recall",
        y="precision",
        color="F1 Score"
    )

    max_f1_point = alt.Chart(max_f1_df, 
                            title = f'PR curve with best threshold (AP score = {ap_forest_train:.3f})',).mark_circle(
        color="red", size=100, opacity=1).encode(
        x="recall",
        y="precision"
    )

    text = max_f1_point.mark_text(
        align='left',
        baseline='middle',
        dx=15).encode(text= alt.Text("threshold", format = ".2f"))

    PR_curve_plot = PR_curve + max_f1_point + text
    

    # Export PR curve
    PR_curve_plot_path = os.path.join(opt['--out_dir'], "PR_curve.png")
    PR_curve_plot.save(PR_curve_plot_path, scale_factor=3)
    print(f"PR curve saved to {PR_curve_plot_path}")

    # Evaluate Model with test data set with best_threshold
    y_pred_train_thres = y_pred_train_prob > best_thres
    y_pred_thres = y_pred_test_prob > best_thres

    # Table of Metrics for positive class with best_thres
    model_perf_thres_df = pd.DataFrame(
        {
            "Accuracy": [accuracy_score(y_train, y_pred_train_thres), accuracy_score(y_test, y_pred_thres)],
            "Precision": [precision_score(y_train, y_pred_train_thres), precision_score(y_test, y_pred_thres)],
            "Recall": [recall_score(y_train, y_pred_train_thres), recall_score(y_test, y_pred_thres)],
            "F1 Score": [f1_score(y_train, y_pred_train_thres), f1_score(y_test, y_pred_thres)]
        },
        index=["Train Data w/ best threshold", "Test Data w/ best threshold"])

    model_perf_best_thres_df = pd.concat([model_perf_df, model_perf_thres_df])

    # Export model performance with best threshold
    model_perf_best_thres_df_path = os.path.join(opt['--out_dir'], "model_performance.csv")
    model_perf_best_thres_df.to_csv(model_perf_best_thres_df_path)
    print(f"Model performance results saved to {model_perf_best_thres_df_path}") 


    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_test_prob)

    roc_df = pd.DataFrame()
    roc_df['fpr'] = fpr
    roc_df['tpr'] = tpr
    roc_df['thresholds'] = thresholds

    pt_roc_idx = (roc_df['thresholds'] - best_thres).abs().argmin()

    roc_curves = alt.Chart(roc_df, title = f"ROC Curve (AUC score = {roc_forest_test:.3f})").mark_line().encode(
            alt.X('fpr', title="False Positive rate"),
            alt.Y('tpr', title="True Positive rate"))

    roc_max_f1_point = alt.Chart(pd.DataFrame(roc_df.iloc[pt_roc_idx]).T, 
                            ).mark_circle(
        color="red", size=100, opacity=1).encode(
        x="fpr",
        y="tpr"
    )

    roc_text = roc_max_f1_point.mark_text(
        align='left',
        baseline='middle',
        dx=15).encode(text= alt.Text("thresholds:Q", format = ".2f"))

    roc_curve_plot = roc_curves + roc_max_f1_point + roc_text

    # Export ROC curve
    roc_curve_plot_path = os.path.join(opt['--out_dir'], "ROC_curve.png")
    roc_curve_plot.save(roc_curve_plot_path, scale_factor=3)
    print(f"ROC curve saved to {roc_curve_plot_path}")

    # Scoring metrics for test data

    test_model_perf_df = pd.DataFrame({
        "Accuracy" : model_perf_thres_df.loc["Test Data w/ best threshold"]["Accuracy"],
        "Precision" : model_perf_thres_df.loc["Test Data w/ best threshold"]["Precision"],
        "Recall" : model_perf_thres_df.loc["Test Data w/ best threshold"]["Recall"],
        "F1 Score" : model_perf_thres_df.loc["Test Data w/ best threshold"]["F1 Score"],
        "Average Precision Score" : ap_forest_test,
        "AUC Score" : roc_forest_test},
        index = ["Test Data Metrics"]).T

    # Export test data metrics dataframe
    test_model_perf_df_path = os.path.join(opt['--out_dir'], "model_performance_test.csv")
    test_model_perf_df.to_csv(test_model_perf_df_path)
    print(f"Model performance results on test data saved to {test_model_perf_df_path}") 

if __name__ == "__main__":
    main()
