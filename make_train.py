import os
from pathlib import Path
import re

import numpy as np
import pandas as pd


def get_sample_path(experiment_name, sample_name):
	"""Return the path to a sample's data based on that sample's name."""
	sample_paths = [Path(experiment_name, sample_name) for sample_name in os.listdir(experiment_name)
			if sample_name[-2:] == ".D"]
	for sample_path in sample_paths:
		if re.search(sample_name, str(sample_path)) != None:
			return sample_path
	return None


def drop_invalid_rows(df, max_na=2):
	'''Return a copy of df where rows with more than max_na missing
	values have been dropped.

	max_na -- Maximum number of missing values that allows a row to
		  not be dropped.
	'''
	# Get the number of nan values per row
	na_per_row = list(df.isna().sum(axis=1))
	# Get the indices of the rows that have 2 or more nans
	drop_is = [i for i, count in enumerate(na_per_row) if count > max_na]
	# Return a copy of df where rows with more than max_na nan values
	# have been dropped.
	return df.drop(labels=drop_is, axis=0)


def impute_missing(df):
	'''Replace missing values (np.NaN) in each numerical column in df.
	Do this imputation directly on the DataFrame passed to it, not on
	a copy.
	'''
	for column_name in df.columns:
    		if df[column_name].dtype != "object":
        		df[column_name] = df[column_name].replace(np.NaN, df[column_name].median())
	return


# Add columns for the wavelength areas
train_df = pd.read_csv("train_frame.csv")
empty_column = [np.NaN] * len(train_df.index)
train_df["218nm Area"] = empty_column.copy()  # Make copies so they don't refer to the same list
train_df["250nm Area"] = empty_column.copy()
train_df["260nm Area"] = empty_column.copy()
train_df["330nm Area"] = empty_column.copy()
train_df["350nm Area"] = empty_column.copy()


# Could generate these automatically from report00.csv but too lazy rn
report_header = ["Peak Number", "Retention Time", "Peak Type", "Peak Width",
                 "Area", "Height", "Percent Area"]

# Get the report data for each sample
for i, data in enumerate(train_df.values):
    experiment_name, sample_name, ret_time, chem_type, *nm_areas = data
    # Get the path to the sample's data folder
    sample_path = get_sample_path(experiment_name, sample_name)
    
    # Skip samples that don't have paths
    if sample_path == None:
        # Print error message for samples not found
        print(f"Found no sample path for {experiment_name} {sample_name}")
        continue

    # Load the five wavelength files individually
    report01_df = pd.read_csv(Path(sample_path, "REPORT01.CSV"), names=report_header, encoding="UTF-16")
    report02_df = pd.read_csv(Path(sample_path, "REPORT02.CSV"), names=report_header, encoding="UTF-16")
    report03_df = pd.read_csv(Path(sample_path, "REPORT03.CSV"), names=report_header, encoding="UTF-16")
    report04_df = pd.read_csv(Path(sample_path, "REPORT04.CSV"), names=report_header, encoding="UTF-16")
    report05_df = pd.read_csv(Path(sample_path, "REPORT05.CSV"), names=report_header, encoding="UTF-16")

    # Round all retention times to one decimal place
    report01_df = report01_df.round({"Retention Time": 1})
    report02_df = report02_df.round({"Retention Time": 1})
    report03_df = report03_df.round({"Retention Time": 1})
    report04_df = report04_df.round({"Retention Time": 1})
    report05_df = report05_df.round({"Retention Time": 1})

    # Don't forget to round ret_time too!!
    ret_time = round(ret_time, 1)

    # If there is more than 1 retention time that matches (due to
    # rounding), then keep that area at -1
    try:
        area1 = int(report01_df[report01_df["Retention Time"] == ret_time]["Area"].iloc[0])
    except IndexError:
        area1 = np.NaN
    try:
        area2 = int(report02_df[report02_df["Retention Time"] == ret_time]["Area"].iloc[0])
    except IndexError:
        area2 = np.NaN
    try:
        area3 = int(report03_df[report03_df["Retention Time"] == ret_time]["Area"].iloc[0])
    except IndexError:
        area3 = np.NaN
    try:
        area4 = int(report04_df[report04_df["Retention Time"] == ret_time]["Area"].iloc[0])
    except IndexError:
        area4 = np.NaN
    try:
        area5 = int(report05_df[report05_df["Retention Time"] == ret_time]["Area"].iloc[0])
    except IndexError:
        area5 = np.NaN

    train_df.iloc[i] = [experiment_name, sample_name, ret_time, chem_type, area1, area2, area3, area4, area5]


# Drop rows with more than 1 missing value.
train_df = drop_invalid_rows(train_df, max_na=1)

# Impute any remaining missing values
impute_missing(train_df)
# NOTE: Consider replacing with chem type specific medians

# Save the training set to a csv
train_df.to_csv("train.csv", index=False)
