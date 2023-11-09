import os
import geopandas as gpd
import pandas as pd
import numpy as np
import configparser as cp
from collections import OrderedDict
from math import sqrt
from shapely import line_interpolate_point, get_coordinates, force_2d
from shapely import geometry
from shapely.ops import linemerge
from scipy.spatial import cKDTree

class multidict(OrderedDict):
    _unique = 0   # class variable

    def __setitem__(self, key, val):
        if isinstance(val, dict):
            self._unique += 1
            key += str(self._unique)
        OrderedDict.__setitem__(self, key, val)

def process_nodes_branches(model_path):
    #ini_path = os.path.join(model_path, r"rmm_output\dflow1d")
    #network_ini = os.path.join(ini_path, r"NetworkDefinition.ini")
    network_ini = os.path.join(model_path, r"NetworkDefinition.ini")
    config = cp.ConfigParser(defaults=None, dict_type=multidict, strict=False)
    config.read(network_ini)

    nodes = []
    branches = []
    for i, section in enumerate(config.sections()):
        row = dict(config.items(section))
        if section.startswith("Node"):
            nodes.append(row)
        elif section.startswith("Branch"):
            branches.append(row)

    branch_df = pd.DataFrame(branches).rename(columns={"geometry": "wkt"})
    branch_geometry = gpd.GeoSeries.from_wkt(branch_df['wkt'], crs="EPSG:28992")
    branch_geometry = branch_geometry.apply(lambda geom: force_2d(geom))
    branch_gdf = gpd.GeoDataFrame(branch_df, geometry=branch_geometry)
    branch_gdf = branch_gdf.to_crs(epsg=28992) #4326

    node_df = pd.DataFrame(nodes)
    node_geometry = gpd.points_from_xy(node_df['x'], node_df['y'], crs="EPSG:28992")
    node_gdf = gpd.GeoDataFrame(node_df, geometry=node_geometry)
    node_gdf[['x', 'y']] = node_gdf[['x', 'y']].astype(float)
    node_gdf = node_gdf.to_crs(epsg=28992)  # 4326
    if False:
        branch_gdf.to_file("obs_points.geojson")
        node_gdf.to_file("obs_points.geojson")

    def splitby(x, dtype="float", delimiter=" "):
        if dtype == "float":
            return [float(x_i) for x_i in x.split(delimiter)]
        elif dtype == "string":
            return [str(x_i) for x_i in x.split(delimiter)]

    branch_df['x'] = branch_df["gridpointx"].apply(splitby)
    branch_df['y'] = branch_df["gridpointy"].apply(splitby)
    branch_df['offset'] = branch_df["gridpointoffsets"].apply(splitby)
    branch_df['grid_id'] = branch_df["gridpointids"].apply(splitby, dtype="string", delimiter=";")

    grid_points_df = branch_df.explode(['x', 'y', 'grid_id', 'offset'])
    grid_points_df = grid_points_df.drop(
        columns=["gridpointx", "gridpointy", "gridpointoffsets", "gridpointids", "wkt"])
    grid_points_df = grid_points_df.rename(columns={"id": "branch_id", "grid_id": "id"})
    grid_points_geometry = gpd.points_from_xy(grid_points_df['x'], grid_points_df['y'], crs="EPSG:28992")
    grid_points_gdf = gpd.GeoDataFrame(grid_points_df, geometry=grid_points_geometry)
    grid_points_gdf[['x', 'y']] = grid_points_gdf[['x', 'y']].astype(float)
    grid_points_gdf = grid_points_gdf.to_crs(epsg=28992)  # 4326
    if False:
        grid_points_gdf.to_file("grid_points.geojson")

    return branch_gdf, node_gdf, grid_points_gdf

def merge_subbranches(branches, subbranches):
    subbranches_geo = {}
    for name, subbranch in subbranches.items():
        subbranch_geo = []
        for segment in subbranch:
            for index, branch in branches.iterrows():
                if branch["main_branch"]:
                    if segment == branch["id"]:
                        line = geometry.shape(branch.geometry)
                        subbranch_geo.append(line)
        subbranches_geo.update({name: subbranch_geo})
    subbranches_merged = {}
    for name, subbranch in subbranches_geo.items():
        subbranch_geo = linemerge(subbranch)
        for coords in subbranch_geo.coords[:]:
            for coor in coords:
                if coor == np.nan:
                    print("found NaN")
        subbranches_merged.update({name: subbranch_geo})
    start_branch = []
    end_branch = []
    for key, values in subbranches.items():
        start_branch.append(values[0])
        end_branch.append(values[-1])
    gdf_dict = {"id": list(subbranches_merged.keys()), "start_node": start_branch, "end_node": end_branch,
                 "geometry": list(subbranches_merged.values())}
    merged_branches_gdf = gpd.GeoDataFrame(gdf_dict, crs="EPSG:28992")

    return merged_branches_gdf

def add_new_subbranch(branch_id, subbranches):
    for key, values in subbranches.items():
        if branch_id in values:
            return key

def add_previous_branches(branch_id, subbranches):
    def get_index(value, dictionary):
        for key, values in dictionary.items():
            if value in values:
                index = values.index(value)
                return key, index
    try:
        branch_ref, branch_index = get_index(branch_id, subbranches)
    except TypeError:
        return np.nan
    return list(subbranches[branch_ref][branch_index+1:])


def update_chainage(chainage, previous_branches, branches_model):
    previous_dist = 0
    for branch in previous_branches:
        dist = branches_model.at[branch, "distance"]
        previous_dist += dist
    return chainage + previous_dist

def process_obs_points(model_path, branch_model_gdf, merged_branches_model_gdf, subbranches):
    #ini_path = os.path.join(model_path, r"rmm_output\dflow1d")
    #obs_ini = os.path.join(ini_path, r"ObservationPoints.ini")
    obs_ini = os.path.join(model_path, r"ObservationPoints.ini")
    config = cp.ConfigParser(defaults=None, dict_type=multidict, strict=False)
    config.read(obs_ini)
    observation_points = []
    for i, section in enumerate(config.sections()):
        row = dict(config.items(section))
        observation_points.append(row)

    observation_points_df = pd.DataFrame(observation_points)
    observation_points_df = observation_points_df.drop(columns=["majorversion", "minorversion", "filetype", "name"])
    observation_points_df = observation_points_df.iloc[1:]
    observation_points_df = observation_points_df.rename(columns={"id": "obs_id", "branchid": "id"})

    branch_model_gdf["distance"] = branch_model_gdf["geometry"].length

    observation_points_df["new_id"] = observation_points_df.apply(
        lambda row: add_new_subbranch(row["id"], subbranches), axis=1)
    observation_points_df["prev_branches"] = observation_points_df.apply(
        lambda row: add_previous_branches(row["id"], subbranches), axis=1)
    observation_points_df = observation_points_df.dropna()
    observation_points_df['chainage'] = observation_points_df['chainage'].astype(float)
    observation_points_df["branch_chainage"] = observation_points_df.apply(
        lambda row: update_chainage(row["chainage"], row["prev_branches"], branch_model_gdf.set_index("id")), axis=1)
    observation_points_df = observation_points_df.set_index("new_id")

    branch_gdf_model_copy = branch_model_gdf.set_index("id")
    branch_gdf_model_copy = branch_gdf_model_copy.drop(
        columns=["name", "order", "wkt", "gridpointscount", "gridpointx", "gridpointy", "gridpointoffsets",
                 "gridpointids", "distance", "geometry", "main_branch"])

    merged_branches_model_gdf = merged_branches_model_gdf.set_index("start_node")
    merged_branches_gdf = pd.merge(
        merged_branches_model_gdf, branch_gdf_model_copy, left_index=True, right_index=True, how='inner')
    merged_branches_gdf = merged_branches_gdf.reset_index()
    merged_branches_gdf = merged_branches_gdf.rename(columns={"index": "start_node", "tonode": "to_node"})
    merged_branches_gdf = merged_branches_gdf.drop(columns=["fromnode"])
    merged_branches_gdf = merged_branches_gdf.set_index("end_node")
    merged_branches_gdf = pd.merge(merged_branches_gdf, branch_gdf_model_copy, left_index=True, right_index=True, how='inner')
    merged_branches_gdf = merged_branches_gdf.reset_index()
    merged_branches_gdf = merged_branches_gdf.drop(columns=["tonode"])
    merged_branches_gdf["distance"] = merged_branches_gdf["geometry"].length

    merged_branches_gdf = merged_branches_gdf.set_index("id")
    observation_points_df_updated_chainage = pd.merge(observation_points_df, merged_branches_gdf, left_index=True,
                                                      right_index=True, how='inner')
    observation_points_df_updated_chainage = observation_points_df_updated_chainage.rename(
        columns={"index": "old_index"})


    observation_points_df_updated_chainage["obs_location"] = observation_points_df_updated_chainage.apply(
        lambda row: multiline_interpolate_point(row['branch_chainage'], row['geometry']), axis=1)
    observation_points_df_updated_chainage = observation_points_df_updated_chainage.reset_index()

    observation_points_df_updated_chainage = observation_points_df_updated_chainage.drop(columns=["geometry"])
    observation_points_df_updated_chainage = observation_points_df_updated_chainage.set_index("obs_id")
    observation_points_df_updated_chainage = observation_points_df_updated_chainage.rename(
        columns={"obs_location": "geometry", "id": "branch_id_model", "new_id": "branch_id_game",
                 "chainage": "chainage_model", "branch_chainage": "chainage_game"})
    observation_points_gdf_updated_chainage = gpd.GeoDataFrame(observation_points_df_updated_chainage,
                                                               geometry=observation_points_df_updated_chainage[
                                                                   "geometry"], crs="EPSG:28992").reset_index()

    """
    branch_gdf = branch_gdf[["id", "wkt", "geometry"]].set_index("id")
    obs_df = pd.merge(observation_points_df, branch_gdf, left_index=True, right_index=True, how='inner')

    branch_geometry = gpd.GeoSeries.from_wkt(obs_df['wkt'], crs="EPSG:28992")
    obs_gdf = gpd.GeoDataFrame(obs_df, geometry=branch_geometry).reset_index()
    obs_gdf['chainage'] = obs_gdf['chainage'].astype(float)
    obs_gdf["obs_location"] = obs_gdf.apply(
        lambda row: multiline_interpolate_point(row['chainage'], row['geometry']), axis=1)
    obs_gdf = obs_gdf.drop(columns=["wkt", "geometry"])
    obs_gdf['x'] = obs_gdf['obs_location'].x
    obs_gdf['y'] = obs_gdf['obs_location'].y
    obs_gdf = obs_gdf.rename(columns={"obs_location": "geometry", "id": "branch_id", "obs_id": "id"})
    obs_gdf = gpd.GeoDataFrame(obs_gdf, geometry=obs_gdf["geometry"], crs="EPSG:28992")
    obs_gdf = obs_gdf.to_crs(epsg=28992)  # 4326
    if False:
        obs_gdf.to_file("obs_points.geojson")
    return obs_gdf
    """
    return observation_points_gdf_updated_chainage

def process_cross_sections(model_path, grid_points_gdf, branch_gdf):
    #ini_path = os.path.join(model_path, r"rmm_output\dflow1d")
    #cross_sec_def = os.path.join(ini_path, r"CrossSectionDefinitions.ini")
    #cross_sec_loc = os.path.join(ini_path, r"CrossSectionLocations.ini")
    cross_sec_def = os.path.join(model_path, r"CrossSectionDefinitions.ini")
    cross_sec_loc = os.path.join(model_path, r"CrossSectionLocations.ini")
    config = cp.ConfigParser(defaults=None, dict_type=multidict, strict=False)
    config.read(cross_sec_def)
    cross_sections_def = []
    for i, section in enumerate(config.sections()):
        row = dict(config.items(section))
        cross_sections_def.append(row)

    cross_sections_def_df = pd.DataFrame(cross_sections_def)
    cross_sections_def_df = cross_sections_def_df.drop(columns=["majorversion", "minorversion", "filetype"])
    cross_sections_def_df = cross_sections_def_df[1:].set_index("id")

    config = cp.ConfigParser(defaults=None, dict_type=multidict, strict=False)
    config.read(cross_sec_loc)
    cross_sections_loc = []
    for i, section in enumerate(config.sections()):
        row = dict(config.items(section))
        cross_sections_loc.append(row)

    cross_sections_loc_df = pd.DataFrame(cross_sections_loc)
    cross_sections_loc_df = cross_sections_loc_df.drop(columns=["majorversion", "minorversion", "filetype"])
    cross_sections_loc_df = cross_sections_loc_df[1:].set_index("id")
    cross_sections_df = pd.merge(cross_sections_def_df, cross_sections_loc_df, left_index=True, right_index=True,
                                 how='inner').reset_index()
    cross_sections_df = cross_sections_df.rename(columns={"id": "cross_id", "name": "grid_id", "branchid": "id"})
    cross_sections_df = cross_sections_df.set_index('id')

    grid_points_gdf = grid_points_gdf.reset_index()
    grid_points_gdf["id"] = grid_points_gdf["id"].str.replace("\.\d+$", '', regex=True)
    grid_points_gdf = grid_points_gdf.set_index("id")

    branch_geo_gdf = branch_gdf[["id", "wkt", "geometry"]].set_index("id")
    cross_df = pd.merge(cross_sections_df, branch_geo_gdf, left_index=True, right_index=True, how='inner')
    branch_geometry_new = gpd.GeoSeries.from_wkt(cross_df['wkt'], crs="EPSG:28992")
    branch_geometry_new = branch_geometry_new[~branch_geometry_new.index.duplicated()]
    cross_gdf = gpd.GeoDataFrame(cross_df, geometry=branch_geometry_new).reset_index()
    cross_gdf = cross_gdf.rename(columns={"cross_id": "id", "id": "branch_id"})
    cross_gdf['chainage'] = cross_gdf['chainage'].astype(float)
    cross_gdf["cross_location"] = cross_gdf.apply(
        lambda row: multiline_interpolate_point(row['chainage'], row['geometry']), axis=1)
    cross_gdf = cross_gdf.drop(columns=["wkt", "geometry"])
    cross_gdf = cross_gdf.rename(columns={"cross_location": "geometry"})
    cross_gdf['x'] = cross_gdf['geometry'].x
    cross_gdf['y'] = cross_gdf['geometry'].y
    cross_gdf = cross_gdf.set_geometry("geometry", crs="EPSG:28992")
    cross_gdf = cross_gdf.to_crs(epsg=28992)  # 4326

    cross_sections_df = cross_sections_df.reset_index()  # cross_gdf
    cross_sections_df = cross_sections_df.rename(columns={"id": "branch_id", "grid_id": "id"})
    cross_sections_df = cross_sections_df.set_index("id")
    grid_points_new_df = pd.merge(grid_points_gdf, cross_sections_df.drop(columns=["branch_id"]),
                                  left_index=True, right_index=True, how='inner')  # .reset_index()
    """
    grid_points_new_geometry = gpd.points_from_xy(grid_points_new_df['x'], grid_points_new_df['y'], crs="EPSG:28992")
    grid_points_new_gdf = gpd.GeoDataFrame(grid_points_new_df, geometry=grid_points_new_geometry)
    grid_points_new_gdf[['x', 'y']] = grid_points_new_gdf[['x', 'y']].astype(float)
    grid_points_new_gdf = grid_points_new_gdf.to_crs(epsg=28992)  # 4326
    if False:
        grid_points_new_gdf.to_file("03_grid_crosssections_id_matched.geojson")
    """

    unmatched_grid = grid_points_gdf
    unmatched_grid['grid_id'] = unmatched_grid.index
    unmatched_grid.drop(grid_points_new_df.index)
    unmatched_grid = unmatched_grid.set_index("branch_id")
    unmatched_cross = cross_gdf.set_index("grid_id").drop(grid_points_new_df.index)
    unmatched_cross["cross_location"] = unmatched_cross.apply(
        lambda row: multiline_interpolate_point(row['chainage'], row['geometry']), axis=1)
    unmatched_cross = unmatched_cross.set_index("branch_id")
    unmatched_grid_with_cross_ids = (index_cross_to_grid(unmatched_cross, unmatched_grid))
    unmatched_grid_with_cross_ids = unmatched_grid_with_cross_ids.dropna()

    grid_ref = unmatched_grid_with_cross_ids.set_index("cross_id")
    cross_ref = unmatched_cross.rename(columns={"id": "cross_id"}).set_index("cross_id").drop(
        columns=["x", "y", "geometry"])
    extra_grid_points_df = pd.merge(grid_ref, cross_ref, left_index=True, right_index=True, how='inner')
    extra_grid_points_df = extra_grid_points_df.reset_index()
    extra_grid_points_df["id"] = extra_grid_points_df["grid_id"]
    extra_grid_points_df = extra_grid_points_df.set_index("id")
    full_grid_df = pd.concat([grid_points_new_df.to_crs(epsg=28992), extra_grid_points_df.to_crs(epsg=28992)])
    assert (len(grid_points_new_df) + len(extra_grid_points_df)) == len(full_grid_df)

    grid_points_full_geometry = gpd.points_from_xy(full_grid_df['x'], full_grid_df['y'], crs="EPSG:28992")
    grid_points_full_gdf = gpd.GeoDataFrame(full_grid_df, geometry=grid_points_full_geometry)
    grid_points_full_gdf[['x', 'y']] = grid_points_full_gdf[['x', 'y']].astype(float)
    grid_points_full_gdf = grid_points_full_gdf.to_crs(epsg=28992)  # 4326
    if False:
        grid_points_full_gdf.to_file("04_grid_crosssections_id_matched_nearest_neighbours_supplement.geojson")
    return grid_points_full_gdf


def multiline_interpolate_point(chainage, line):
    line_points = get_coordinates(line)
    for i in range(len(line_points) - 1):
        dist = sqrt((line_points[i+1][0] - line_points[i][0])**2 + (line_points[i+1][1] - line_points[i][1])**2)
        if chainage <= dist:
            line = geometry.LineString([(line_points[i][0], line_points[i][1]), (line_points[i+1][0], line_points[i+1][1])])
            return line_interpolate_point(line, chainage)
        else:
            chainage = chainage - dist
    return


def index_cross_to_grid(cross_gdf, grid_gdf):
    grid_gdf = grid_gdf.reset_index()
    cross_gdf = cross_gdf.reset_index()
    grid_coor = np.array(grid_gdf[['x', 'y']])
    cross_coor = np.array(cross_gdf[['x', 'y']])
    cross_coor = cross_coor[~np.isnan(cross_coor).any(axis=1)]

    grid_tree = cKDTree(grid_coor)
    dist, index = grid_tree.query(cross_coor, distance_upper_bound=100)

    grid_index = []
    cross_index = []
    for i, x in enumerate(index):
        if x == grid_tree.n:
            continue
        if x in grid_index:
            dist1 = dist[grid_index.index(x)]
            dist2 = dist[i]
            if dist2 < dist1:
                cross_index.pop(grid_index.index(x))
                grid_index.pop(grid_index.index(x))
                grid_index.append(x)
                cross_index.append(i)
                continue
        grid_index.append(x)
        cross_index.append(i)
    assert len(grid_index) == len(cross_index)

    indices = dict(zip(grid_index, cross_index))
    for key, value in indices.items():
        grid_gdf.loc[key, "cross_id"] = cross_gdf.loc[value, "id"]
    return grid_gdf