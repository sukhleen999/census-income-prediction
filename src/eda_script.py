'''
This script aims to read the data from clean dataset from the directory and
write a script to perform exploratory data analysis of the train dataset
and show the results(figures and tables)

Usage: eda_script.py <clean_train_file> --out_dir=<out_dir>

Options:
<clean_train_file>                                   Path of clean train data file
--out_dir=<out_dir>                                  Path to the results folder
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
        if opt['--out_dir'] is None:
            raise TypeError("Argument out_dir can't be None")
    except Exception as e:
        print(e)
        sys.exit(1)

     # Create output directory if not exist
        if not os.path.exists(opt['--out_dir']):
            os.makedirs(opt['--out_dir'])

    try:

        # Read clean training data from directory
        train_file_path = opt['<clean_train_file>']
        train_df = pd.read_csv(train_file_path, header=None)


        # First five rows of train dataset shape and export to table train_head.csv

        train_df.head().to_csv(f"{opt['--out_dir']}/train_head.csv")


        # Check to find any missing value and datatype of the columns and export to table train_info.csv

        train_df.info().to_csv(f"{opt['--out_dir']}/train_info.csv")


        # Check the statistical summary of all numeric columns and export to table train_describe.csv

        train_df.describe().to_csv(f"{opt['--out_dir']}/train_describe.csv")

        # checking for class imbalance and export to table class_imbalance.csv

        train_df['income'].value_counts().to_csv(f"{opt['--out_dir']}/class_imbalance.csv")


        # Figure to represent class imbalance
        class_imbalance = alt.Chart(train_df).mark_bar(opacity=0.5).encode(
                          alt.X("income", title="Income"),
                          alt.Y("count()"),
                          color="income").properties(
                          width=200,
                          height=200)

        # Save the class imbalance figure into the results/eda path
        # class_imbalance.save('results/eda/class_imbalance.png',scale_factor=5)


        class_imbalance.save(f"{opt['--out_dir']}/class_imbalance.png", scale_factor=3)

        # Visualizing numerical columns
        feature_plot = alt.Chart(train_df).mark_bar(opacity=0.5).encode(
                         alt.X(alt.repeat(), type="quantitative",
                               bin=alt.Bin(maxbins=20)),
                         alt.Y("count()", stack=False),
                         color="income").properties(
                         width=300,
                         height=200).repeat(numeric_cols, columns=2)

        # Save the feature plot into the results/eda path
        # feature_plot.save('results/eda/feature_plot.png', scale_factor=5)

        feature_plot.save(f"{opt['--out_dir']}/feature_plot.png", scale_factor=3)

        # Visualizing categorical columns:
        categorical_cols = list(set(train_df.columns) - set(numeric_cols))

        # Removing native_country column due to high class imbalance
        categorical_cols.remove('native_country')
        categorical_cols.remove('income')
        # Exploring categorical features

        categorical_dist = alt.Chart(train_df).mark_bar(opacity=0.5).encode(
                           alt.X("count()", stack=False),
                           alt.Y(alt.repeat(), type="ordinal", sort='x'),
                           color="income").properties(
                           width=200,
                           height=200).repeat(categorical_cols, columns=3)

        # Save the categorical distribution figure into the results/eda path
        # categorical_dist.save('results/eda/categorical_dist.png', scale_factor=5)

        categorical_dist.save(f"{opt['--out_dir']}/categorical_dist.png", scale_factor=3)

        print(f"-----Saving the tables and figures in {opt['--out_dir']}-----")

    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()