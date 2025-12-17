import numpy as np 
import pandas as pd

# ===================
# FUNCTIONS
# ===================

# get CRSS data, exposure adjusted LR uses only the Southern Region CRSS sample
def get_crss_data(filepath = "crss_combined.csv"):
    """
    Load + preprocess CRSS data and combined vehicle and accident dataframes 

    Parameters
    ----------
    filepath : str
        Path to crss_combined.csv

    Returns
    -------
    combined_vehicle : pd.DataFrame
    combined_accident : pd.DataFrame
    LEFT, RIGHT, STRAIGHT : pd.DataFrame  (each contains CASENUM)
    """

    crss_combined = pd.read_csv(filepath)

    # filter, only Southern Region
    crss_combined = crss_combined[crss_combined["REGION"] == 3].reset_index(drop = True)

     # create vehicle base
    combined_vehicle = (crss_combined[["CASENUM","VEH_NO","WEIGHT","P_CRASH1"]].drop_duplicates())

    # create accident base
    combined_accident = (crss_combined[["ID","CASENUM","URBANICITY","REGION","WEIGHT","YEAR",
                                    "VE_TOTAL","MONTH","DAY_WEEK","HOUR","RELJCT2",
                                    "TYP_INT","LGT_COND","WEATHER"]].drop_duplicates())
    
    
    # filter for 3 and 4 way intersections
    combined_accident = combined_accident[combined_accident["TYP_INT"].isin([2, 3])]

    valid_cases = combined_accident["CASENUM"].unique()
    combined_vehicle = combined_vehicle[combined_vehicle["CASENUM"].isin(valid_cases)]  

    turning_ratios = pd.DataFrame({
    "TYP_INT": [2, 3],         
    "left":  [0.166, 0.175],
    "straight": [0.76, 0.756],
    "right": [0.074, 0.069]
    })
    # merge with accident 
    combined_accident = combined_accident.merge(turning_ratios, on = "TYP_INT", how = "left")

    # recode 2 to 4 
    combined_accident["TYP_INT"] = combined_accident["TYP_INT"].replace({2:4})

    # Define LEFT, RIGHT, STRAIGHT
    LEFT = combined_vehicle.loc[combined_vehicle["P_CRASH1"]==11, ["CASENUM"]].drop_duplicates()
    RIGHT = combined_vehicle.loc[combined_vehicle["P_CRASH1"]==10, ["CASENUM"]].drop_duplicates()
    STRAIGHT = combined_vehicle.loc[combined_vehicle["P_CRASH1"]==1, ["CASENUM"]].drop_duplicates()


    return combined_vehicle, combined_accident, LEFT, RIGHT, STRAIGHT 

# get likelihood ratios
def comparison_f(df1, df2, var, data= None):
    """
    Compare total accidents for df1 vs df2 across a categorical variable var.

    Parameters
    ----------
    df1, df2 : pd.DataFrame
        Vehicle subsets with CASENUM
    var : str
        Column name in data to group by
    k : float
        Scaling factor
    data : pd.DataFrame
        Accident-level data with WEIGHT and CASENUM
    """
    assert data is not None
    # Drop rows where TYP_INT is NA
    data = data.dropna(subset=["TYP_INT"])

    #go through each var 
    values = sorted(data[var].dropna().unique())

    results = []
    
    for val in values:
        
        asum = data.loc[data[var]==val].merge(df1, on="CASENUM")["WEIGHT"].sum()
        bsum = data.loc[data[var]==val].merge(df2, on="CASENUM")["WEIGHT"].sum()

        amean = data.loc[data[var]==val]["left"].mean()
        bmean = data.loc[data[var]==val]["right"].mean()
        
        ratio = (asum/amean) / (bsum/bmean) if (bsum/bmean) != 0 else np.nan

        results.append({
            var: val,
            "LR": ratio
        })

    return pd.DataFrame(results)
    

if __name__ == "__main__":
    
    # get data, change trucks, exposure to get desired results
    combined_vehicle, combined_accident, LEFT, RIGHT, STRAIGHT = get_crss_data(filepath = "crss_combined.csv")

    char = ["MONTH", "DAY_WEEK", "HOUR", "LGT_COND", "WEATHER"]
    # compute likelihood ratio, k = 0.42 for exposure adjusted result
    for c in char:
        result = comparison_f(LEFT, RIGHT, c, data = combined_accident)

        #download results
        result.to_csv(f"adjusted_left_right_{c}.csv", index=False)


    
