import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import numpy as np
from haversine import haversine_vector, Unit

def get_nearest_signal_id(m=500):
    # function that assign closest signal id to the corresponding accident on fars

    farscolumns = ["TYP_INT", "STATE", "YEAR", "MONTH", "DAY", "DAY_WEEK", "HOUR", "P_CRASH1", "LGT_COND", "WEATHER", "LONGITUD", "LATITUDE" ]
    exportcolumns = ["signalID", "longitude", "latitude"]

    fars = pd.read_csv("data/fars_combined.csv", usecols = farscolumns)
    gdot = fars[fars["STATE"] == 13].copy().reset_index(drop = True)
    export = pd.read_csv("data/Export.csv", usecols = exportcolumns)

    # remove nas and none intersections 
    intersections = [2, 3, 4, 5, 6, 7]
    gdot = gdot.dropna(subset=["LATITUDE", "LONGITUD"])

    # Extract coordinates
    gdot_coords = np.vstack((gdot["LATITUDE"], gdot["LONGITUD"])).T
    export_coords = np.vstack((export["latitude"], export["longitude"])).T

    # Build KDTree and query nearest neighbor
    tree = cKDTree(export_coords)
    distances_deg, indices = tree.query(gdot_coords, k=1)

    # Convert degree distances to meters (~111 km per degree)
    approx_meters = distances_deg * 111_000  

    # Compute accurate geodesic distance (optional but better)
    matched_export_coords = export.iloc[indices][["latitude", "longitude"]].values
    accurate_meters = haversine_vector(gdot_coords, matched_export_coords, Unit.METERS)

    # Filter by threshold 
    threshold_m = m
    mask = accurate_meters <= threshold_m

    matched = gdot.loc[mask].copy()
    matched["nearest_latitude"] = matched_export_coords[mask][:, 0]
    matched["nearest_longitude"] = matched_export_coords[mask][:, 1]
    matched["distance_m"] = accurate_meters[mask]

    # Optionally join other export columns for matched points
    matched_export = export.iloc[indices[mask]].reset_index(drop=True)
    matched = pd.concat([matched.reset_index(drop=True), matched_export.add_suffix("_export")], axis=1)

    # recode stateid 
    matched = matched.rename(columns={"signalID_export": "SignalID", "DAY_WEEK": "dow"})

    keepcols = ["SignalID", "YEAR", "MONTH", "DAY", "dow", "HOUR","P_CRASH1", "TYP_INT", "LGT_COND", "WEATHER", 
                "LATITUDE", "LONGITUD", "distance_m"]
        
    final = matched[keepcols].copy()

    #Save results
    final.to_csv("data/gdot_with_nearest_signal_within_500m.csv", index=False)


def load_data():

    # Load data
    georgia_fars = pd.read_csv("data/gdot_with_nearest_signal_within_500m.csv")
    signal_hourly_raw = pd.read_csv("data/signal_hourly_combined.csv")

    # Create fars maneuver dataset

    # recode TYP_INT
    georgia_fars["TYP_INT"] = np.select(
        [
            georgia_fars["TYP_INT"] == 1,
            georgia_fars["TYP_INT"] == 2,
            georgia_fars["TYP_INT"].isin([3, 4])
        ],
        ["non-intersection", "four-way", "three-way"],
        default="other"
    )

    # recode maneuver
    georgia_fars["maneuver"] = np.select(
        [
            georgia_fars["P_CRASH1"] == 11,
            georgia_fars["P_CRASH1"] == 12,
            georgia_fars["P_CRASH1"] == 13,
        ],
        ["left", "right", "straight"],
        default="other"
    )

    # raw fatal accident count (grouped)
    tmp = (
        georgia_fars
        .groupby(["SignalID", "YEAR", "MONTH", "HOUR", "dow","TYP_INT", "maneuver"])
        .size()
        .reset_index(name="fatal_accident_count")
    )

    # average fatal accident count per (SignalID, MONTH, dow, TYP_INT, maneuver)
    georgia_fars_maneuver = (
        tmp.groupby(["SignalID", "MONTH", "HOUR", "dow", "TYP_INT", "maneuver"])
        .fatal_accident_count.mean()
        .reset_index()
    )

    # rename columns
    georgia_fars_maneuver = georgia_fars_maneuver.rename(columns={
        "TYP_INT": "intersection_type",
        "HOUR": "hour",
    })

    # Use hourly raw

    # recode TYP_INT
    signal_hourly_raw["intersection_type"] = np.select(
        [
            signal_hourly_raw["intersection_type"] == "4_way",
            signal_hourly_raw["intersection_type"] == "T_intersection"
        ],
        ["four-way", "three-way"],
        default="other"
    )

    # compute left/right/straight
    signal_hourly_raw["left"] = signal_hourly_raw.filter(regex="_L$").fillna(0).sum(axis=1)
    signal_hourly_raw["right"] = signal_hourly_raw.filter(regex="_R$").fillna(0).sum(axis=1)
    signal_hourly_raw["straight"] = signal_hourly_raw.filter(regex="_T$").fillna(0).sum(axis=1)

    signal_hourly_raw["other"] = (
        signal_hourly_raw["Vehicle_Vehicle_Total"].fillna(0)
        + signal_hourly_raw["Exit_Exit_Total"].fillna(0)
        - (signal_hourly_raw["left"] +
        signal_hourly_raw["right"] +
        signal_hourly_raw["straight"])
    )

    # assure date type
    signal_hourly_raw["date"] = pd.to_datetime(signal_hourly_raw["date"])
    signal_hourly_raw["MONTH"] = signal_hourly_raw["date"].dt.month
    signal_hourly_raw["dow"] = signal_hourly_raw["date"].dt.weekday + 1 # 1-7 Mo-Su
    signal_hourly_maneuver = (
        signal_hourly_raw[["SignalID", "intersection_type","date", "MONTH", "hour", "dow", "left", "right", "straight", "other"]]
        .melt(
            id_vars=["SignalID", "intersection_type", "date", "MONTH", "hour", "dow"],
            var_name="maneuver",
            value_name="total_traffic_volume"
        )
    )

    # Join with georgia_fars_maneuver

    df_maneuver = (
        signal_hourly_maneuver
        .merge(
            georgia_fars_maneuver,
            on=["SignalID", "MONTH", "hour", "dow", "maneuver"],
            how="left"
        )
    )

    # keep intersection type from FARS is available, otherwise, from hourly data
    df_maneuver["TYP_INT"] = np.where(
        df_maneuver["intersection_type_y"].notna(),
        df_maneuver["intersection_type_y"],
        df_maneuver["intersection_type_x"]
    )

    # drop intermediate columns
    df_maneuver  = df_maneuver .drop(columns=["intersection_type_y", "intersection_type_x"])

    # fill missing intersection type
    df_maneuver["TYP_INT"] = df_maneuver["TYP_INT"].fillna("other")

    # impute fatal accident count
    df_maneuver["fatal_accident_count"] = df_maneuver["fatal_accident_count"].fillna(0)

    # final columns
    df_maneuver = df_maneuver.rename(columns={
        "dow": "day_of_week",
        "MONTH": "month",
        "TYP_INT": "intersection_type"
    })[
        ["day_of_week", "month", "hour","intersection_type",
        "maneuver", "total_traffic_volume", "fatal_accident_count"]
    ]

    # Save final output

    df_maneuver.to_csv("data/df_maneuver.csv", index=False)

    return df_maneuver
