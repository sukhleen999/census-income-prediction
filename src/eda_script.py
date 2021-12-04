# author: Affrin Sultana and Navya Dahiya
# date: 2021-11-25

'''
This script aims to read the data from clean dataset from the directory and
perform exploratory data analysis on the train dataset
and show the results in the form of figures and tables

Usage: eda_script.py <clean_train_file> --out_dir=<out_dir>

Options:
<clean_train_file>                      Path of clean train data file
--out_dir=<out_dir>                     Path to the results folder
'''
from docopt import docopt
import os
import sys
import io
import pandas as pd
import numpy as np
import altair as alt
import shutil
from altair_saver import save

alt.data_transformers.enable('data_server')
alt.renderers.enable("mimetype")


buffer = io.StringIO()
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

    
    #Create output directory if not exist
    if not os.path.exists(opt['--out_dir']):
        os.makedirs(opt['--out_dir'])
    else:
    # In case you create a file wrongly, delete the whole dir and create an empty one with new files
        shutil.rmtree(opt['--out_dir'])
        os.makedirs(opt['--out_dir'])
    try:

        # Read clean training data from directory
        train_file_path = opt['<clean_train_file>']
        train_df = pd.read_csv(train_file_path)

        numeric_cols=train_df.select_dtypes(['number']).columns.tolist()

        # First five rows of train dataset shape and export to table train_head.csv
        train_df.head().to_csv(f"{opt['--out_dir']}/data_head.csv")
        print(f"----- Saved table for data_head in {opt['--out_dir']}/data_head.csv -----")

        # Check to find any missing value and datatype of the columns and export to table train_info.csv
        train_df.info(buf=buffer)
        s = buffer.getvalue()
        with open("train_df_info.csv", "w", encoding="utf-8") as f:
            f.write(s.split(" ----- \n ")[1])
        train_df_info = pd.read_csv('train_df_info.csv', sep="\s+", header=None)
        train_df_info=train_df_info.iloc[:, 1:]
        train_df_info.columns = ['Column', 'Non-Null', 'Count', 'DType']
        train_df_info=train_df_info[:-2]
        train_df_info.to_csv(f"{opt['--out_dir']}/data_info.csv", index=False)
        print(f"----- Saved plot for data_info in {opt['--out_dir']}/data_info.csv -----")
        os.remove("train_df_info.csv")
        
        # Figure to represent statistical summary of the numerical data
        train_df_desc= alt.Chart(train_df).mark_boxplot().encode(
                        alt.X('income', type='ordinal'),
                        alt.Y(alt.repeat("column"), type='quantitative'),
                        color=alt.Color('income:N', legend=None)
                                        ).properties(width = 100).repeat(
                        column=numeric_cols).properties(title = 'Boxplot for numeric features').configure_title(dx = 400, fontSize = 15)
        train_df_desc.save(f"{opt['--out_dir']}/stat_summary_plot.png", scale_factor=3)
        print(f"----- Saved plot for stat summary in {opt['--out_dir']}/stat_summary_plot.png -----")

        # Figure to represent class imbalance
        class_imbalance = alt.Chart(train_df,
                                     title = "Distribution of target column (Income)").mark_bar(opacity=0.5).encode(
                          alt.X("income", title="Income"),
                          alt.Y("count()"),
                          color="income").properties(
                          width=200,
                          height=200)

        # Save the class imbalance figure into the results/eda path
        class_imbalance.save(f"{opt['--out_dir']}/class_imbalance.png", scale_factor=3)
        print(f"----- Saved plot for class imbalance in {opt['--out_dir']}/class_imbalance.png -----")

        # Visualizing numerical columns
        numeric_feature_plot =  alt.Chart(train_df).mark_bar(opacity=0.5).encode(
                                alt.X(alt.repeat(), type="quantitative", bin=alt.Bin(maxbins=20)),
                                alt.Y("count()", stack = False),
                                color = "income").properties(
                                width=300,
                                height=200).repeat(numeric_cols, columns=2).properties(
        title = "Distribution of numeric features").configure_title(align = 'center', fontSize = 15, dx = 300)

        # Save the feature plot into the results/eda path
        numeric_feature_plot.save(f"{opt['--out_dir']}/numeric_feature_plot.png")
        print(f"----- Saved plot for numeric features distribution in {opt['--out_dir']}/numeric_feature_plot.png -----")

        # Finding the correlation

        corrMatrix = train_df.corr()

        # plotting the correlations

        corr_df = train_df.select_dtypes('number').corr('spearman').stack().reset_index(name='corr')
        corr_df.loc[corr_df['corr'] == 1, 'corr'] = 0  # Remove diagonal
        corr_df['abs'] = corr_df['corr'].abs()

        corr_plot = alt.Chart(corr_df).mark_rect().encode(
            x=alt.X('level_0', title=''),
            y=alt.Y('level_1', title=''),
            size=alt.Size('abs', legend=None),
            color=alt.Color('corr', scale=alt.Scale(domain=(-1, 1), domainMid=0), title='Correlation')).properties(
            title = "Correlation plot for numeric features").configure_title(fontSize =15)

        corr_plot.save(f"{opt['--out_dir']}/corr_plot.png")
        print(f"----- Saved plot for correlation in {opt['--out_dir']}/corr_plot.png -----")

        # Visualizing categorical columns:
        categorical_cols = list(set(train_df.columns) - set(numeric_cols))
        categorical_cols.remove('native_country')
        categorical_cols.remove('income')

        # Removing native_country column due to high class imbalance
        train_df.loc[train_df['native_country'] != 'United-States', 'native_country'] = 'Non-United-States'
        native_country= alt.Chart(train_df, title = "Distribution of Native Country").mark_bar(opacity=0.5).encode(
                        y=alt.Y("native_country", type="ordinal"),
                        x=alt.X('count()',  stack = False),
                        color = "income").properties(
                        width=200,
                        height=100)

        native_country.save(f"{opt['--out_dir']}/native_country_plot.png")
        print(f"----- Saved plot for native country feature in {opt['--out_dir']}/native_country_plot.png -----")


        # Exploring categorical features
        categorical_feat_plot = alt.Chart(train_df).mark_bar(opacity=0.5).encode(
                           alt.X("count()", stack=False),
                           alt.Y(alt.repeat(), type="ordinal", sort='x'),
                           color="income").properties(
                           width=200,
                           height=200).repeat(categorical_cols, columns=3).properties(
        title = "Distribution of categorical features").configure_title(align = 'center', fontSize = 15, dx = 400)

        # Save the categorical distribution figure into the results/eda path
        categorical_feat_plot.save(f"{opt['--out_dir']}/categorical_feat_plot.png", scale_factor=3)
        print(f"----- Saved plot for categorical feature in {opt['--out_dir']}/categorical_feat_plot.png -----")

        print(f"-----Saved all tables and figures in {opt['--out_dir']}-----")

    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()