import os
import geopandas as gpd
from PyQt5.QtGui import QPixmap
from matplotlib.colors import LogNorm, Normalize, CenteredNorm

def load_scenarios():
    # def add_scenario(ds):
    #     path_re = re.compile(r'(?P<scenario>(0_)?(?P<slr>\d+)mzss_(?P<discharge>\d+)m3s)(\\)')
    #     print(ds.encoding['source'])
    #     match = path_re.search(ds.encoding['source'])
    #     scenario = match.group("scenario")
    #     # new dataset contains dates with last day of month, let's keep it consistent
    #     result = ds.expand_dims(scenario=[scenario])
    #     return result
    #
    # demo_output_dir = r"C:\Users\Robert-Jan\Dropbox\Onderzoeker map\Salti game design\Prototyping\RMM coding\demonstrator_output_files"
    # obs_file = os.path.join(demo_output_dir, "model_obs_points.csv")
    # obs_points_model_df = pd.read_csv(obs_file)
    # obs_points_geo = gpd.GeoSeries.from_wkt(obs_points_model_df["geometry"], crs=28992)
    # obs_points_model_gdf = gpd.GeoDataFrame(obs_points_model_df, geometry=obs_points_geo)
    # obs_points_model_gdf = obs_points_model_gdf[["obs_id", "geometry", "branch_rank"]]
    #
    # model_path = pathlib.Path(r"C:\Users\Robert-Jan\Dropbox\Onderzoeker map\Salti game design\Prototyping\sobek_rmm_output")
    # model_files = list(model_path.glob("**/Integrated_Model_output/dflow1d/output/observations.nc"))
    #
    # ds_obs = xr.open_mfdataset(model_files, preprocess=add_scenario)
    # selected = ds_obs[["observation_id", "scenario", "water_salinity", 'water_velocity', 'water_level']]
    # df_obs = selected.to_dataframe()
    # df_obs["observation_id"] = df_obs["observation_id"].str.decode("utf-8")
    # df_obs["observation_id"] = df_obs["observation_id"].str.strip()
    # df_obs = df_obs.reset_index()
    #
    # df_may_idx = np.logical_and(df_obs["time"].dt.month == 5, df_obs["time"].dt.day < 15)
    # df_obs_selected = df_obs[df_may_idx]
    #
    # obs_points_df = pd.merge(df_obs_selected, obs_points_model_gdf, left_on="observation_id", right_on="obs_id")
    # obs_points_gdf = gpd.GeoDataFrame(obs_points_df, geometry=obs_points_df["geometry"])

    dir_path = os.path.dirname(os.path.realpath(__file__))
    scenario_location = os.path.join(dir_path, "input_files")
    scenario_model_file = os.path.join(scenario_location, "Bouke_model_output_NCR_scenarios.gpkg")
    obs_points_model_gdf = gpd.read_file(scenario_model_file)
    obs_points_model_gdf = obs_points_model_gdf.to_crs(epsg=3857)
    obs_points_bbox = obs_points_model_gdf.bounds
    x_min = obs_points_bbox["minx"].min()
    x_max = obs_points_bbox["maxx"].max()
    x_margin = 0.05 * (x_max - x_min)
    y_min = obs_points_bbox["miny"].min()
    y_max = obs_points_bbox["maxy"].max()
    y_margin = 0.05 * (y_max - y_min)
    world_bbox = [x_min - x_margin,
                  x_max + x_margin,
                  y_min - y_margin,
                  y_max + y_margin]
    salinity_range = Normalize(obs_points_model_gdf["water_salinity"].min(), obs_points_model_gdf["water_salinity"].max())
    #water_level_range = Normalize(obs_points_model_gdf["water_level"].min(), obs_points_model_gdf["water_level"].max())
    #low_velocity_value = obs_points_model_gdf["water_velocity"].min()
    #high_velocity_value = obs_points_model_gdf["water_velocity"].max()
    #water_velocity_range = CenteredNorm(halfrange=max(abs(low_velocity_value), abs(high_velocity_value)))
    #water_depth_range = Normalize(obs_points_model_gdf["water_depth"].min(), obs_points_model_gdf["water_depth"].max())
    scenario_game_file = os.path.join(scenario_location, "Bouke_game_output_NCR_scenarios.gpkg")
    obs_points_game_gdf = gpd.read_file(scenario_game_file)
    obs_points_bbox = obs_points_game_gdf.bounds
    game_bbox = [obs_points_bbox["minx"].min(),
                 obs_points_bbox["maxx"].max(),
                 obs_points_bbox["miny"].min(),
                 obs_points_bbox["maxy"].max()]
    return obs_points_model_gdf, obs_points_game_gdf, world_bbox, game_bbox, salinity_range #, water_level_range, water_velocity_range

def load_images():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    colorbar_location = os.path.join(dir_path, "input_files")
    colorbar_salinity_file = os.path.join(colorbar_location, "colorbar_salinity_small.png")
    colorbar_salinity = QPixmap(colorbar_salinity_file)
    colorbar_water_level_file = os.path.join(colorbar_location, "colorbar_water_level_small.png")
    colorbar_water_level = QPixmap(colorbar_water_level_file)
    colorbar_water_velocity_file = os.path.join(colorbar_location, "colorbar_water_velocity_small.png")
    colorbar_water_velocity = QPixmap(colorbar_water_velocity_file)
    #basemap_image_file = os.path.join(colorbar_location, "basemap.png")
    #basemap_image = plt.imread(basemap_image_file)
    return colorbar_salinity, colorbar_water_level, colorbar_water_velocity