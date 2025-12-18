import numpy as np
import pandas as pd
import osmnx as ox
import geopandas as gpd
import time
import os 

ox.settings.use_cache = True
ox.settings.overpass_rate_limit = True
ox.settings.timeout = 600
ox.settings.requests_kwargs = {}                

# =================
# global 
# =================
Graphs = {}

state_map = {
    13: 'Georgia'
}

state_utm = {
    "Alabama": 26916, "Alaska": 3338, "Arizona": 26912, "Arkansas": 26915, "California": 26910,
    "Colorado": 26913, "Connecticut": 26918, "Delaware": 26918, "Florida": 26917, "Georgia": 26917,
    "Hawaii": 26904, "Idaho": 26911, "Illinois": 26916, "Indiana": 26916, "Iowa": 26915,
    "Kansas": 26914, "Kentucky": 26916, "Louisiana": 26915, "Maine": 26919, "Maryland": 26918,
    "Massachusetts": 26919, "Michigan": 26917, "Minnesota": 26915, "Mississippi": 26916, "Missouri": 26915,
    "Montana": 26912, "Nebraska": 26914, "Nevada": 26911, "New Hampshire": 26919, "New Jersey": 26918,
    "New Mexico": 26913, "New York": 26918, "North Carolina": 26917, "North Dakota": 26914, "Ohio": 26917,
    "Oklahoma": 26914, "Oregon": 26910, "Pennsylvania": 26918, "Rhode Island": 26919, "South Carolina": 26917,
    "South Dakota": 26914, "Tennessee": 26916, "Texas": 26914, "Utah": 26912, "Vermont": 26918,
    "Virginia": 26918, "Washington": 26910, "West Virginia": 26917, "Wisconsin": 26916, "Wyoming": 26912, "District of Columbia": 26918
}

crs_LL = 4326

# ========================
# FUNCTIONS 
# ========================

def loadstatesdata(state):
    # cache graphs to avoid repeated downloads
    os.makedirs("./graphs", exist_ok=True)
    path = f"./graphs/{state}.graphml"
    if os.path.exists(path):
        return ox.load_graphml(path)
    G = ox.graph_from_place(state + ", USA", network_type = "drive")
    ox.save_graphml(G, path)
    return G

def loadcrashdata():
    column = ["YEAR", "ST_CASE", "STATE", "LONGITUD", "LATITUDE", "TYP_INT", "P_CRASH1"] # ST_CASE or unique codes we create
    df = pd.read_csv("fars_combined.csv", usecols = column, encoding_errors = "replace")
    return df 

# deal with two way streets, group them into one
def twoway(segment_id):
    try:
        parts = str(segment_id).split("-")
        if len(parts) == 3:
            u, v, key = parts 
            return "-".join(sorted([u,v])) + f"-{key}"
        elif len(parts) == 2:
            u, v = parts 
            return "-".join(sorted([u,v]))
        else:
            return segment_id
    except Exception:
        return segment_id 

def matching(state, crashdf):
    #1) get state graph
    graph = loadstatesdata(state)

    #2) convert to nodes and edges
    nodes, edges = ox.graph_to_gdfs(graph, nodes = True, edges = True, fill_edge_geometry = True)

    #3) get unique ids for each segments
    edges["segment_id"] = edges.index.map(lambda i: f"{i[0]}-{i[1]}-{i[2]}")

    #4) get state utm 
    utm = state_utm[state]

    #5) convert edges to utm
    edges = edges.to_crs(utm)
    #print(edges.columns)

    #6) create gdf for crashpoints 
    crash_points = gpd.GeoDataFrame(crashdf, geometry =gpd.points_from_xy(crashdf.LONGITUD, crashdf.LATITUDE), crs = crs_LL)

    #7) get state boundary, map crash points
    state_poly = ox.geocode_to_gdf(state + ", USA").to_crs(crs_LL).geometry.iloc[0]
    pts_den = gpd.sjoin(crash_points, gpd.GeoDataFrame(geometry=[state_poly], crs = crs_LL),
                    predicate="within", how="inner").drop(columns=["index_right"])

    #8) convert to meters
    pts_den = pts_den.to_crs(utm)

    # spatial join crash points to nearest roads(edges) and attach road information to our geo df
    matched = gpd.sjoin_nearest(pts_den, edges[["segment_id","geometry", "highway", "maxspeed", "lanes"]],
    how="left", max_distance=40, distance_col="dist")

    # duplicates 
    matched["segment_id"] = matched["segment_id"].apply(twoway)
    matched = matched.drop_duplicates(subset=["YEAR", "ST_CASE", "segment_id"])

    #delete graph
    del graph, nodes, edges, pts_den, state_poly, crash_points
    # print
    print(f"{state}: matched")
    return matched
    

if __name__ == "__main__":
    # get crash data from master table
    crash_df = loadcrashdata()

    results = []

    #match for all state, one at a time
    for state in state_map:
        statedf = crash_df[crash_df["STATE"] == state ]
        start = time.time()
        statematched = matching(state_map[state], statedf)
        end = time.time()
        results.append(statematched)
        print(f"{state} match time: ", end - start)
    
    # reproject all state GeoDataFrames to a common CRS before merging
    for i in range(len(results)):
        results[i] = results[i].to_crs(4326)   # convert to lon/lat

    all = pd.concat(results, ignore_index=True)
    all.to_csv("FARSmatched_GA.csv", index=False)
