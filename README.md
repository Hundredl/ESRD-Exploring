# Exploring the Relationship Between Dietary Intake and Clinical Outcomes in Peritoneal Dialysis Patients

## Data processing and statistics
- install the env named esrd to run the python code
```
conda create -n esrd python=3.7.13
conda activate esrd
pip install -r requirement.txt
```

- process the raw data
```
bash data_processing.sh
```
- merge the data
```
data_merge.ipynb
```
- perform statistical and basic hypothesis tests on data

```
data_statistic.ipynb
```
## Two-stage Model for Mortality Risk Evaluation and Modeling Non-Linear Nutrient-Risk Correlations

- install the env named for esrd_r to tun the R code.
```
conda env create -f environment.yml
```
- run the two-stage model and get the results.
```
model_run.ipynb
```
- formate the results
```
result_format.ipynb (use the environment "esrd")
```
## Folders

```
project_folder/  # Main project folder
├── esrd/         # Data
│   ├── origin/      # Stores original data
│   ├── processed/   # Stores processed data
│   ├── result/      # Stores some data statistical graphs
│   └── src/         # Stores data processing code
└── result/       # Stores statistical results
    └── statistics_r/   # Statistical results using R
        └── res/        # Result folder
            └── time    # Final runtime result
```