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

    # TODO: Export Evaluation Tables and Graphs

if __name__ == "__main__":
    main()