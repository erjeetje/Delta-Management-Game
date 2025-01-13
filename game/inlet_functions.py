import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from shapely import wkt
from scipy.spatial import cKDTree


def index_inlets_to_model_locations(water_inlets_data_gdf, model_output_index_gdf):
    model_output_gdf = model_output_index_gdf.copy()
    model_output_gdf = model_output_gdf[model_output_gdf['time'] == model_output_gdf['time'].iloc[0]]
    model_output_gdf = model_output_gdf.set_index("id")
    model_output_gdf["x"] = model_output_gdf.apply(lambda row: row["geometry"].x, axis=1)
    model_output_gdf["y"] = model_output_gdf.apply(lambda row: row["geometry"].y, axis=1)
    model_grid_coords = list(zip(model_output_gdf["x"], model_output_gdf["y"]))
    tree = cKDTree(model_grid_coords)

    def index_point(point_id, point, tree, model_output_gdf):
        xy = (point.x, point.y)
        try:
            distance, index = tree.query(xy)
            return model_output_gdf.iloc[index].name
        except Exception as e:
            print(f"Error querying tree for inlet {point_id}: {e}")
            return np.nan

    water_inlets_data_gdf["output_location"] = water_inlets_data_gdf.apply(
        lambda row: index_point(row.name, row["geometry"], tree, model_output_gdf), axis=1)
    water_inlets_data_gdf = water_inlets_data_gdf.dropna()
    return water_inlets_data_gdf

def get_inlet_salinity(water_inlets_data_gdf, model_output_gdf, turn):
    inlet_data = water_inlets_data_gdf.copy()
    model_output = model_output_gdf.copy()
    turn_model_output = model_output[model_output["turn"] == turn]
    turn_model_output = turn_model_output.set_index("id")
    turn_inlet_salinity = inlet_data.set_index("output_location").merge(
        turn_model_output[["time", "water_salinity", "salinity_category", "turn"]], left_index=True,
        right_index=True)
    turn_inlet_salinity = turn_inlet_salinity.reset_index()
    turn_inlet_salinity = turn_inlet_salinity.set_index("name")
    return turn_inlet_salinity


def get_exceedance_at_inlets(inlets_with_salinity):
    inlets_df = inlets_with_salinity.copy()
    inlets_df = inlets_df.reset_index()
    inlets_df['Num_days_exceedance_normal'] = 0
    inlets_df['Num_days_consecutive_normal'] = 0
    inlets_df['Num_days_exceedance_drought'] = 0
    inlets_df['Num_days_consecutive_drought'] = 0

    def calculate_max_consecutive_days_exceeding_threshold(values, threshold):
        current_streak = 0
        max_streak = 0

        for value in values:
            if value > threshold:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return max_streak

    for inlet_name in inlets_df['name'].unique():
        inlet_data = inlets_df[inlets_df['name'] == inlet_name]

        salinity_values = inlet_data['water_salinity'].values
        cl_threshold_normal = inlet_data.iloc[0]['CL_threshold_during_regular_operation_(mg/l)']
        cl_threshold_drought = inlet_data.iloc[0]['CL_threshold_during_drought_(mg/l)']

        if not pd.isna(cl_threshold_normal):
            num_days_exceeding_normal = int((salinity_values > cl_threshold_normal).sum())
            max_streak_normal = int(
                calculate_max_consecutive_days_exceeding_threshold(salinity_values, cl_threshold_normal))
        else:
            num_days_exceeding_normal = "not applicable"
            max_streak_normal = "not applicable"

        if not pd.isna(cl_threshold_drought):
            num_days_exceeding_drought = int((salinity_values > cl_threshold_drought).sum())
            max_streak_drought = int(
                calculate_max_consecutive_days_exceeding_threshold(salinity_values, cl_threshold_drought))
        else:
            num_days_exceeding_drought = "not applicable"
            max_streak_drought = "not applicable"

        inlets_df['Num_days_exceedance_normal'] = np.where((inlets_df['name'] == inlet_name), num_days_exceeding_normal,
                                                           inlets_df['Num_days_exceedance_normal'])
        inlets_df['Num_days_consecutive_normal'] = np.where((inlets_df['name'] == inlet_name),
                                                            max_streak_normal,
                                                            inlets_df['Num_days_consecutive_normal'])
        inlets_df['Num_days_exceedance_drought'] = np.where((inlets_df['name'] == inlet_name),
                                                            num_days_exceeding_drought,
                                                            inlets_df['Num_days_exceedance_drought'])
        inlets_df['Num_days_consecutive_drought'] = np.where((inlets_df['name'] == inlet_name),
                                                             max_streak_drought,
                                                             inlets_df['Num_days_consecutive_drought'])

    return inlets_df.set_index("name")