from matplotlib.colors import LogNorm, Normalize, CenteredNorm

def get_bbox(output_gdf, gdf_type="world"):
    obs_points_bbox = output_gdf.bounds
    if gdf_type == "world":
        x_min = obs_points_bbox["minx"].min()
        x_max = obs_points_bbox["maxx"].max()
        x_margin = 0.05 * (x_max - x_min)
        y_min = obs_points_bbox["miny"].min()
        y_max = obs_points_bbox["maxy"].max()
        y_margin = 0.05 * (y_max - y_min)
        bbox = [x_min - x_margin,
                x_max + x_margin,
                y_min - y_margin,
                y_max + y_margin]
    elif gdf_type == "game":
        bbox = [obs_points_bbox["minx"].min(),
                obs_points_bbox["maxx"].max(),
                obs_points_bbox["miny"].min(),
                obs_points_bbox["maxy"].max()]
    return bbox

def get_salinity_scale(output_gdf):
    salinity_range = Normalize(output_gdf["water_salinity"].min(),
                               output_gdf["water_salinity"].max())
    return salinity_range