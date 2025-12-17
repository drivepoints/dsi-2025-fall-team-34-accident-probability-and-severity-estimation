import numpy as np
import pandas as pd

# get CRSS data, exposure adjusted LR uses only the Southern Region CRSS sample
def get_crss_data(filepath = "crss_combined.csv", truck = False):
    """
    Load + preprocess CRSS data and combined vehicle and accident dataframes 
    Truck vs. All Vehicles

    Parameters
    ----------
    filepath : str
        Path to crss_combined.csv
    truck : bool
        If True, keep BODY_TYP 60-79 (trucks)

    Returns
    -------
    combined_vehicle : pd.DataFrame
    combined_accident : pd.DataFrame
    LEFT, RIGHT, STRAIGHT : pd.DataFrame  (each contains CASENUM)
    """

    # get crss file 
    crss_combined = pd.read_csv(filepath)
    
    # if truck and if not 
    if truck:
        crss_combined = crss_combined[crss_combined["BODY_TYP"].between(60, 79)].reset_index(drop = True)

    # create vehicle base
    combined_vehicle = (
    crss_combined[["CASENUM","VEH_NO","WEIGHT","P_CRASH1"]]
    .drop_duplicates()
    )

    # create accident base
    combined_accident = (
        crss_combined[
            [
            "ID","CASENUM","URBANICITY","REGION","WEIGHT","YEAR",
            "VE_TOTAL","MONTH","DAY_WEEK","HOUR","RELJCT2",
            "TYP_INT","LGT_COND","WEATHER"
            ]
        ].drop_duplicates()
    )
    # Define LEFT, RIGHT, STRAIGHT
    LEFT = combined_vehicle.loc[combined_vehicle["P_CRASH1"]==11, ["CASENUM"]].drop_duplicates()
    RIGHT = combined_vehicle.loc[combined_vehicle["P_CRASH1"]==10, ["CASENUM"]].drop_duplicates()
    STRAIGHT = combined_vehicle.loc[combined_vehicle["P_CRASH1"]==1, ["CASENUM"]].drop_duplicates()


    return combined_vehicle, combined_accident, LEFT, RIGHT, STRAIGHT 

def comparison_f(df1, df2, var, k, data= None):
    """
    Compare total accidents for df1 vs df2 across a categorical variable var.

    Parameters
    ----------
    df1, df2 : pd.DataFrame
        Vehicle subsets with CASENUM
    var : str
        Column name in data to group by
    k : list 
        list of scaling factors
    data : pd.DataFrame
        Accident-level data with WEIGHT and CASENUM
    """
    assert data is not None
    # Drop rows where TYP_INT is NA
    data = data.dropna(subset=["TYP_INT"])
    
    values = sorted(data[var].dropna().unique())

    results = []

    for val in values:
        a = data.loc[(data["TYP_INT"] != 1) & (data[var]==val)].merge(df1, on="CASENUM")["WEIGHT"].sum()
        b = data.loc[(data["TYP_INT"] != 1) & (data[var]==val)].merge(df2, on="CASENUM")["WEIGHT"].sum()
        ratio = a / b * k if b != 0 else np.nan
        #above_baseline = ratio > baseline * 1.01 if not np.isnan(ratio) else np.nan

        results.append({var: val, f"k = {k}" : ratio})

    return pd.DataFrame(results)



if __name__ == "__main__":

    combined_vehicle, combined_accident, LEFT, RIGHT, STRAIGHT = get_crss_data(filepath = "crss_combined.csv",
                                                                               truck = False)
    
    k = [0.25, 0.5, 1, 1.5, 2]
    char = ["MONTH", "DAY_WEEK", "HOUR", "LGT_COND", "WEATHER"]

    # go through all the characteristics 
    for c in char:
        # go through each assumed k 
        lr = None
        for i in k:
            r1 = comparison_f(LEFT, RIGHT, c, i, data = combined_accident)
            if lr is None:
                lr = r1
            else:
                lr = pd.merge(lr, r1, on = c, how = "outer")
        # download csv outputs of likelihood ratios under different scaling factors
        lr.to_csv(f"left_right_{c}.csv", index=False)

            