# Team 34 - Accident Probability and Severity Estimation

## Overview 
This project analyzes driving characteristics and conditions extracted from national data Fatality Analysis Reporting System (FARS) and Crash Reporting Sampling System (CRSS) from 2016 - 2023. Our goal is to extend past work on crash severity and pre-crash movement analysis to support routing algorithms for safer driving planning.

The repository includes:

- data cleaning, feature selection, and merging pipeline for FARS and CRSS from 2016 - 2023
- Exploratory Data Analysis and Visualization (EDAV) for both truck-specific crashes and all vehicle types crashes 
- Road-level and Intersection-level matching using OpenStreetMap API and scraped from Georgia AADT traffic count data 
- Output and Models:
    1. Likelihood Ratio - used when exposure(traffic volume) is unknown 
    2. Turning Ratio - estimates left/through/right exposures using Georgia data 
    3. Crash Probabilities

## Repository Structure
```
project/
├── .gitignore 
├── data_process/                   # Jupyter notebooks for analysis & preprocessing
│   ├── fars_clean/                 # produces cleaned csv file using accident, vehicle and person data
│   │   ├── fars_person_clean.ipynb
│   │   ├── fars_accident_clean.ipynb
│   │   └── fars_vehicle_clean.ipynb
│   ├── crss_clean/
│   │   ├── crss_person_clean.ipynb
│   │   ├── crss_accident_clean.ipynb
│   │   └── crss_vehicle_clean.ipynb
│   ├── combined_clean/
│   │   └── combined_cleaned_fars_crss.ipynb
│   ├── edav/
│   │   ├── truck_eda_model.ipynb
│   │   └── all_eda_model.ipynb
│   ├── georgia/
│   │   ├── osm_mapping.py
│   │   ├── signal_aggregation.ipynb
│   │   ├── gdot_signal_scrape.ipynb
│   │   └── match_signal_traffic.ipynb
├── data/                           
│   ├── FARS/                       # Download FARS 2016 - 2023 data
│   ├── CRSS/                       # Download CRSS 2016 - 2023 data
│   ├── signal/                     # Raw signal traffic data scraped from GDOT (xlsx)
│   ├── signal_aggregation/         # created when aggregation.ipynb notebook runs
│   └── outputs/
├── models/
│   ├── likelihood.py               # modeling likelihood ratios for all crashes & trucks
│   ├── turning_ratio.ipynb  
│   └── acc_prob_model.ipynb        # model crash probabilities using linear and logistic regression
├── ml_pipeline/                        # Two-stage ML pipeline for Georgia Fatal Accidents Estimation
│   ├── data/                           # Minimal datasets for running the ml pipeline
│   │   ├── Export.csv                  # Georgia latitude / longitude data for SignalID
│   │   ├── fars_combined.csv           # Harmonized FARS 2016 - 2023 data
│   │   └── signal_hourly_combined.csv  # Sample data from Georgia traffic volume May 1st - Nov 1st 2025
│   ├── models/                         # Folder to save information from trained models
│   └── src/
│   │   ├── data_loader.py              # Script to process data and load information
│   │   ├── features.py                 # Script to calculate feautures based on a dataframe
│   │   ├── inference.py                # Script to predict fatal accident probabilities based on trained models
│   │   ├── train_risk.py               # Script to train fatal accident probabilities for inference
│   │   └── train_volume.py             # Script to train traffic volume for inference
│   ├── requirements.txt                # Python libraries required for ml pipeline to run 
│   └── experiments.ipynb               # Notebook to run ml pipeline and test some examples
└── README.md
```

## Getting Started 

### Data Required 
Please download FARS and CRSS data into the data folder from the NHTSA Website:
FARS - https://www.nhtsa.gov/research-data/fatality-analysis-reporting-system-fars
CRSS - https://www.nhtsa.gov/crash-data-systems/crash-report-sampling-system

1. **Data Cleaning** - Run folders fars_clean and crss_clean
2. **Merging FARS and CRSS**  - Run folder combined_clean 
3. **Road-level Matching and Signal-ID Matching (Georgia)** - Run folder georgia_aadt_match 
4. **EDAV** - Run folder edav
5. **Models** - Run folder models

Each stage saves intermediate output needed by the next step. All data should be saved in the data folder.

## Methodology 
1. **Data Sources**
- FARS 2016 - 2023: accident, vehicle, and person tables 
    - (FARS data contains fatal crashes nationally)
- CRSS 2016 - 2023: accident, vehicle, and person tables 
    - CRSS data contains a sample of both fatal and non-fatal crashes nationally. 
- Georgia AADT by Signal ID

Accident, vehicle, and person provides a comprehensive description about the crash containing information on location, people and vehicle involved, environmental conditions, and important pre-crash movements. 

2. **Data Cleaning, Standardization, and Merging** 

We handled mainly missing values, filter out unrealistic values, and recoded columns and entries so that it is consistent across all years. We then merged across tables (accident, people, and vehicles) and years ensure all entries are unique. A SQLite pipeline was built when completing this. Finally, we combined the CRSS and FARS data together to create a master table that will become the input of our model. 

3. **Road/Intersection Matching**

As FARS provides longtitude and latitude data for each crash, we matched the location to actual roads on OpenStreetMap to gain context. After we found Georgia's AADT data by signals online, we were able to scrape that information and match each crash in Georgia against specific intersections traffic count to get its exposure. We were also able to aggregate by days and day of week.

4. **Modeling Approach**

(1) Likelihood Ratio Modeling 
We applied this model on our cleaned CRSS data to calculate the relative crash risk. We computed the likelihood ratio for left/right/straight movements under different conditions such as weather, visibility, time of day, and etc. and were able to use turning ratio from Georgia as exposure adjustment for this model. This model was applied on data with all vehicles and on trucks specifically. 

(2) Turning Ratio
We used Georgia AADT data to calculate exposure estimates (left/right/through proportions) aggregated by intersection type (3-way and 4-way) and days/day of week. The turning ratios were then used to adjust the likelihood ratio model to obtain crash risk.

## ML pipeline
The folder ml_pipeline implements a machine learning framework to estimate the probability of fatal traffic accidents at intersections in Georgia. By integrating high-fidelity traffic volume data from the Georgia Department of Transportation (GDOT) with fatal accident records from the Fatality Analysis Reporting System (FARS), this model calculates risk profiles for specific vehicle maneuvers (Left Turn, Right Turn, Straight) based on time of day, seasonality, and intersection geometry.

### Model Architecture
This project solves the challenge where Traffic Volume (the primary driver of accidents) is known during training but unavailable during real-time inference.

Stage 1: Volume Imputation (XGBoost Regressor)
Input: Hour, Day of Week, Month, Intersection Type, Maneuver Type.
Output: Predicted Traffic Volume.
Goal: Learn the temporal and geometric patterns of traffic flow.

Stage 2: Risk Classification (XGBoost Classifier)
Input: Hour, Day of Week, Month, Intersection Type, Maneuver Type, Predicted Volume (from Stage 1).
Output: Probability of Fatal Accident.
Technique: Weighted training (Safe Passage vs. Accident) + Isotonic Calibration.

### Installation & Usage

1. Prerequisites

Ensure you have Python 3.8+ installed.

```
git clone https://github.com/drivepoints/dsi-2025-fall-team-34-accident-probability-and-severity-estimation.git
cd ml_pipeline
pip install -r requirements.txt
```

2. Running the Pipeline

To train the models and test inference you should run experiments.ipynb

3. Using for Inference

To use the trained model in your own application:
```
from src.inference import AccidentPredictor

predictor = AccidentPredictor()

# Predict risk for a Left Turn at a 4-way intersection
# on October on Monday at 20:00 p.m. 
vol, risk = predictor.predict(
    hour=20, 
    day_of_week=1, 
    month=8, 
    intersection_type='4-way', 
    maneuver='left'
)

print(f"Risk Probability: {risk:.8f}")
```

## Authors and Contacts
**Brigid Christine Meisenbacher** - bcm2167@columbia.edu
**Jose Murguia Fuentes** - jm5890@columbia.edu
**Renee Hsiao** - jh4947@columbia.edu
**Ziran Lin** - zl3509@columbia.edu

## Acknowledgement 
- **TA:** Adam S. Kelleher
- **Mentors:** Adam Litke, Zsolt Lattmann
- **Data Providers:** NHTSA, OpenStreetMap, GDOT