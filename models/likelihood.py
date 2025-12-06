import numpy as np
import pandas as pd




# ====================
# RESULTS
# ====================

def comparison_f(df1, df2, var, k, data= None):
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
    
    values = sorted(data[var].dropna().unique())

    # Baseline totals
    #a0 = data.loc[data["TYP_INT"]!=1].merge(df1,on="CASENUM")["WEIGHT"].sum()
    #b0 = data.loc[data["TYP_INT"]!=1].merge(df2,on="CASENUM")["WEIGHT"].sum()
    #baseline = a0 / b0 * k if b0 != 0 else np.nan

    results = []

    for val in values:
        a = data.loc[(data["TYP_INT"] != 1) & (data[var]==val)].merge(df1, on="CASENUM")["WEIGHT"].sum()
        b = data.loc[(data["TYP_INT"] != 1) & (data[var]==val)].merge(df2, on="CASENUM")["WEIGHT"].sum()
        ratio = a / b * k if b != 0 else np.nan
        #above_baseline = ratio > baseline * 1.01 if not np.isnan(ratio) else np.nan

        results.append({var: val, f"k = {k}" : ratio})

    return pd.DataFrame(results)



if __name__ == "__main__":
    # get crss file 
    crss_combined = pd.read_csv("crss_combined.csv")
    truck = False
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
        lr.to_csv(f"left_right_{c}.csv", index=False)

            