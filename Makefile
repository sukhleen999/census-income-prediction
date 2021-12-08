# Makefile
# Philson Chan, Nov 2021
# This driver script builds the prediction model to predict 
# the income level of people from their demographic features. 
# Upon running the script, a prediction model would be trained 
# and evaluated, and a report would be generated. This script
# takes no arguments.

# example usage:
# make all
ifeq ($(OS),Windows_NT) 
    py_exe := python
else
    py_exe := python3
endif

eda_artifacts = categorical_feat_plot.png class_imbalance.png data_head.csv data_info.csv native_country_plot.png numeric_feature_plot.png stat_summary_plot.png corr_plot.png
eda_files = $(addprefix results/eda/, $(eda_artifacts))

model_artifact = model.pickle baseline_result.csv hyperparam_result.csv
model_files = $(addprefix results/model/, $(model_artifact))

eval_artifact = PR_curve.png ROC_curve.png classification_report.csv confusion_matrix.csv model_performance.csv model_performance_test.csv
eval_files = $(addprefix results/eval/, $(eval_artifact))

all: doc/report.html

# Download Data
data/raw/train.csv : src/download_data.py
	$(py_exe) src/download_data.py https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data --out_dir=data/raw --file_name=train.csv

data/raw/test.csv : src/download_data.py
	$(py_exe) src/download_data.py https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test --out_dir=data/raw --file_name=test.csv

# Data Cleaning
data/clean/clean_train.csv data/clean/clean_test.csv: src/clean_data.py data/raw/train.csv data/raw/test.csv
	$(py_exe) src/clean_data.py data/raw/train.csv data/raw/test.csv --out_dir=data/clean --train_filename=clean_train.csv --test_filename=clean_test.csv

# EDA Graphs
$(eda_files) : src/eda_script.py data/clean/clean_train.csv
	$(py_exe) src/eda_script.py data/clean/clean_train.csv --out_dir=results/eda

# Model Building
$(model_files): src/model_building.py data/clean/clean_train.csv
	$(py_exe) src/model_building.py data/clean/clean_train.csv --out_dir=results/model/ --output_model=model.pickle

# Model Evaluation
$(eval_files): src/model_evaluation.py data/clean/clean_train.csv data/clean/clean_test.csv results/model/model.pickle
	$(py_exe) src/model_evaluation.py data/clean/clean_train.csv data/clean/clean_test.csv results/model/model.pickle --out_dir=results/eval

# Rendering the Report
doc/report.html: doc/report.Rmd doc/child/*.Rmd doc/census_income_refs.bib $(eda_files) $(model_files) $(eval_files)
	Rscript -e "rmarkdown::render('doc/report.Rmd', output_format = 'html_document')"

clean:
	rm -rf results/
	rm -rf data/
	rm -f doc/report.html
