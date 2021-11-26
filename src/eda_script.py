'''
This script aims to read the data from clean dataset from the directory and
write a script to perform exploratory data analysis of the train dataset
and show the results(figures and tables)

Usage: eda_script.py <clean_train_file>

Options:
<clean_train_file>                                   Path of clean train data file
'''
from docopt import docopt
import os
from collections import defaultdict
import sys

import pandas as pd
import numpy as np
import altair as alt

from collections import defaultdict
from altair_saver import save

alt.data_transformers.enable('data_server')
alt.renderers.enable("mimetype")


opt = docopt(__doc__)

def main():
    try:
        if opt['<clean_train_file>'] is None:
            raise TypeError("Argument clean_train_file can't be None")
    except Exception as e:
        print(e)
        sys.exit(1)
    try:

        #Read clean training data from directory
        train_file_path = opt['<clean_train_file>']
        train_df = pd.read_csv(train_file_path, header=None)
    
        #Basic sanity check of train dataset
        train_df.shape
        train_df.head(5)
        
        #Check to find any missing value and datatype of the columns
        train_df.info()
        train_df.isnull().sum()
        
        #Check the summary of all numeric columns
        train_df.describe()

        # checking for class imbalance
        
        train_df['income'].value_counts()
        
        # Figure to represent class imbalance
        class_imbalance = alt.Chart(train_df).mark_bar(opacity=0.5).encode(
                          alt.X("income", title = "Income"),
                          alt.Y("count()"),
                          color = "income").properties(
                        width=200,
                        height=200)
        
        #Save the class imbalance figure into the results/eda path
        class_imbalance.save('results/eda/class_imbalance.png',scale_factor=5)
        
        # Visualizing numerical columns
        
        feature_plot = alt.Chart(train_df).mark_bar(opacity=0.5).encode(
                         alt.X(alt.repeat(), type="quantitative", bin=alt.Bin(maxbins=20)),
                         alt.Y("count()", stack = False),
                         color = "income").properties(
                         width=300,
                         height=200).repeat(numeric_cols, columns=2)
        
    #Save the feature plot into the results/eda path
    feature_plot.save('results/eda/feature_plot.png', scale_factor=5)
    
     # Visualizing categorical columns:
        categorical_cols = list(set(train_df.columns) - set(numeric_cols))

# Removing native_country column due to high class imbalance
categorical_cols.remove('native_country')
categorical_cols.remove('income')
# Exploring categorical features

categorical_dist = alt.Chart(train_df).mark_bar(opacity=0.5).encode(
     alt.X("count()", stack=False),
     alt.Y(alt.repeat(), type="ordinal", sort='x'),
     color = "income").properties(
    width=200,
    height=200).repeat(categorical_cols, columns=3)

#Save the categorical distribution figure into the results/eda path
categorical_dist.save('results/eda/categorical_dist.png', scale_factor=5)



        #Save the cleaned train data into the directory
        train_df.to_csv(f"{opt['--out_dir']}/{opt['--train_filename']}", encoding='utf-8',index=False)
        print(f"-----Cleaned and saved the train data file in {opt['--out_dir']}/{opt['--train_filename']}-----")
    
    except Exception as e:
        print(e)