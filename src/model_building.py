'''
This script aims to transform the cleaned data set to a ready-to-use data to be fed to the machine learning model, and save it to the file system

Usage: model_building.py <input_train_file> --out_dir=<out_dir> [--output_model=<model_filename>]

Options:
<input_train_file>                                   Path of raw train data file
--out_dir=<out_dir>                                  Path to the output folder
[--output_model=<model_filename>]                    Output model file name, default "model.pickle"
'''

from docopt import docopt
import os
import pandas as pd
import pickle
import sys

opt = docopt(__doc__)

def save_model(obj, filename):
    fp = open(filename, "wb")
    pickle.dump(obj, fp)
    
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
    print(opt)
    try:
        if opt['<input_train_file>'] is None:
            raise TypeError("Argument input_train_file can't be None")
        if opt['--out_dir'] is None:
            raise TypeError("Argument out_dir can't be None")
        if opt['--output_model'] is None:
            raise TypeError("Argument train_filename can't be None")
    except Exception as e:
        print(e)
        sys.exit(1)

    # Create output directory if not exist
    if not os.path.exists(opt['--out_dir']):
        os.makedirs(opt['--out_dir'])
    
    # Actual path to save the model
    output_filename = os.path.join(opt['--out_dir'], opt['--output_model'])

    train_df = pd.read_csv(opt["<input_train_file>"])
    train_df = upfront_transform(train_df)

    print(train_df, output_filename)




if __name__ == "__main__":
    main()
