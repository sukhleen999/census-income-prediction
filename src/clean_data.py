'''
This script aims to read the downloaded dataset from the script 
This script reads the dataset, cleans, preprocesses and outputs the clean processed data directory name.

Usage: clean_data.py <input_train_file> <input_test_file> --out_dir=<out_dir> [--train_filename=<train_filename>] [--test_filename=<test_filename>]

Options:
<input_train_file>                                   Path of raw train data file
<input_test_file>                                    Path of raw test data file
--out_dir=<out_dir>                                  Path to the data folder
{--train_filename=<train_filename>]                Output train file name
[--test_filename=<test_filename>]                  Output test file name
'''
from docopt import docopt
import os
from collections import defaultdict

import pandas as pd
import numpy as np


opt = docopt(__doc__)

def main():

    # Create output directory if not exist
    if not os.path.exists(opt['--out_dir']):
        os.makedirs(opt['--out_dir'])

    #columns in the dataset
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']

    #Read training data from directory
    train_file_path = opt['<input_train_file>']
    train_df = pd.read_csv(train_file_path, header=None)
  
    #Assign the columns to the train dataframe
    train_df.columns = columns

    #Strip extra whitespace from strings in the 'object' dtype data 
    train_df_obj = train_df.select_dtypes(['object'])
    train_df[train_df_obj.columns] = train_df_obj.apply(lambda x: x.str.strip())
  
    #Replace '?' with 'Nan'
    train_df = train_df.replace("?", np.nan)

    #Save the cleaned train data into the directory
    train_df.to_csv(f"{opt['--out_dir']}/{opt['--train_filename']}", encoding='utf-8',index=False)

    test_file_path = opt['<input_test_file>']
    test_df = pd.read_csv(test_file_path, header = 1)


    #Read the test data from the directory
    test_df = test_df.T.reset_index().T.reset_index().drop(columns=['index'])

    #Assign the columns to the test dataframe
    test_df.columns = columns

    '''
    As all columns are of 'object' type in test data, 
    convert the ones that should be numeric to numeric data types
    '''
    numeric_cols = train_df.select_dtypes(['number']).columns.tolist()
    convert_dict = defaultdict()
    for col in numeric_cols:
        convert_dict[col]=float
    test_df = test_df.astype(convert_dict)

    #Strip extra whitespace from strings in the 'object' dtype data 
    test_df_obj = test_df.select_dtypes(['object'])
    test_df[test_df_obj.columns] = test_df_obj.apply(lambda x: x.str.strip())

    #Replace '?' with 'Nan'
    test_df = test_df.replace("?", np.nan)

    #Save the cleaned test data into the directory
    test_df.to_csv(f"{opt['--out_dir']}/{opt['--test_filename']}", encoding='utf-8', index=False)

if __name__ == "__main__":
    main()


