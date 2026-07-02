# Predicting Student Health Risk

This repository showcases a machine learning workflow created for the Kaggle Playground Series Season 6 Episode 7 competition. The goal is to accurately classify the health status of students into three categories: at-risk, unhealthy, or fit. To handle the notable class imbalance present in the training set, the project focuses on balanced accuracy as the primary evaluation metric, ensuring fair recall across all target groups.

## Dataset Information

The data needed for this project is hosted on Kaggle. You will need to download the files and place them into the designated subdirectory within this repository to run the code successfully.

### Expected Directory Structure

```
student-health-risk/
├── README.md
├── 5_sratfold_cb_xbg_lbgm_ensemble.ipynb
└── playground-series-s6e7/
    ├── train.csv
    ├── test.csv
    └── sample_submission.csv
```

* Create a directory named `playground-series-s6e7` within the project root.
* Obtain `train.csv`, `test.csv`, and `sample_submission.csv` from the competition page on Kaggle.
* Store these files inside the new directory.

## Methodology

The approach relies on combining the strengths of three gradient boosted decision tree models to produce robust predictions. The final output is a weighted ensemble of probability estimates derived from these individual models.

* LightGBM: A framework for gradient boosting known for efficiency and scalability.
* CatBoost: A gradient boosting implementation that manages categorical features with ease.
* XGBoost: A powerful library offering flexibility for gradient boosting tasks.

To validate the model, the project employs stratified cross validation using five separate folds. This technique ensures that each training subset preserves the original class distribution. Additionally, each model is configured with balanced class weights to directly address the imbalance in the target class distribution.

## Technical Details

The notebook provides custom functions to manage data handling and training.

* find_data_dir: Locates the required dataset by checking both local folders and typical Kaggle paths.
* prepare_features: Handles data cleaning by identifying missing values in categorical data, replacing them with a missing label, and ensuring correct data types.
* make_lightgbm, make_catboost, make_xgboost: These functions configure the respective models with parameters specifically tuned to improve balanced accuracy.
* align_proba: Ensures that probability outputs from different models are correctly mapped to the target classes.
* balanced_score_from_proba: Computes the balanced accuracy score, ensuring the evaluation reflects equal importance for all classes.

## Dependencies

This project relies on the following Python packages:

* numpy
* pandas
* scikit-learn
* lightgbm
* xgboost
* catboost

## Installation

It is recommended to use a Python virtual environment to manage project dependencies.

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Linux or macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install the required packages
pip install -r requirements.txt
```
