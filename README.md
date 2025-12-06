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
- CRSS 2016 - 2023: accident, vehicle, and person tables 
- Georgia AADT by Signal ID
FARS data contains fatal crashes nationally. 
CRSS data contains a sample of both fatal and non-fatal crashes nationally. 
Accident, vehicle, and person provides a comprehensive description about the crash containing information on location, people and vehicle involved, environmental conditions, and important pre-crash movements. 

2. **Data Cleaning, Standardization, and Merging** 
We handled mainly missing values, filter out unrealistic values, and recoded columns and entries so that it is consistent across all years. We then merged across tables (accident, people, and vehicles) and years ensure all entries are unique. A SQLite pipeline was built when completing this. Finally, we combined the CRSS and FARS data together to create a master table that will become the input of our model. 

3. **Road/Intersection Matching**
As FARS provides longtitude and latitude data for each crash, we matched the location to actual roads on OpenStreetMap to gain context. After we found Georgia's AADT data by signals online, we were able to scrape that information and match each crash in Georgia against specific intersections traffic count to get its exposure. We were also able to aggregate by days and day of week.

4. **Modeling Approach**
(1) Likelihood Ratio Modeling 
We applied this model on our cleaned CRSS data to calculate the relative crash risk between left/right/straight turning movements.
We computed the likelihood ratio for left/right/straight movements under different conditions such as weather, visibility, time of day, and etc. We were able to use turning ratio from Georgia as exposure adjustment for this model. We run this model on data with all vehicles and on trucks specifically. 

(2) Turning Ratio
We used Georgia AADT data to calculate exposure estimates (left/right/through proportions) aggregated by intersection type (3-way and 4-way) and days/day of week. The turning ratios were then used to adjust the likelihood ratio model to obtain crash risk.

(3) Crash Probabilities

## Authors and Contacts
**Brigid Christine Meisenbacher** - bcm2167@columbia.edu
**Jose Murguia Fuentes** - jm5890@columbia.edu
**Renee Hsiao** - jh4947@columbia.edu
**Ziran Lin** - zl3509@columbia.edu

## Acknowledgement 
- **TA:** Adam S. Kelleher
- **Mentors:** Adam Litke, Zsolt Lattmann
- **Data Providers:** NHTSA, OpenStreetMap, GDOT