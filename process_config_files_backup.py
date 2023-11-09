import os
import geopandas as gpd
import pandas as pd
import numpy as np
import configparser as cp
from collections import OrderedDict
from math import sqrt
from shapely import line_interpolate_point, get_coordinates, force_2d
from shapely.geometry import LineString
from scipy.spatial import cKDTree

class multidict(OrderedDict):
    _unique = 0   # class variable

    def __setitem__(self, key, val):
        if isinstance(val, dict):
            self._unique += 1
            key += str(self._unique)
        OrderedDict.__setitem__(self, key, val)

def process_nodes_branches(model_path):
    ini_path = os.path.join(model_path, r"rmm_output\dflow1d")
    network_ini = os.path.join(ini_path, r"NetworkDefinition.ini")
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

def load_obs_points(model_path, branch_gdf):
    ini_path = os.path.join(model_path, r"rmm_output\dflow1d")
    obs_ini = os.path.join(ini_path, r"ObservationPoints.ini")
    config = cp.ConfigParser(defaults=None, dict_type=multidict, strict=False)
    config.read(obs_ini)
    observation_points = []
    for i, section in enumerate(config.sections()):
        row = dict(config.items(section))
        observation_points.append(row)

    observation_points_df = pd.DataFrame(observation_points)
    observation_points_df = observation_points_df.drop(columns=["majorversion", "minorversion", "filetype"])
    observation_points_df = observation_points_df.iloc[1:]
    observation_points_df = observation_points_df.rename(columns={"id": "obs_id", "branchid": "id"})
    observation_points_df = observation_points_df.set_index("id")

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

def process_cross_sections(model_path, grid_points_gdf, branch_gdf):
    ini_path = os.path.join(model_path, r"rmm_output\dflow1d")
    cross_sec_def = os.path.join(ini_path, r"CrossSectionDefinitions.ini")
    cross_sec_loc = os.path.join(ini_path, r"CrossSectionLocations.ini")
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
            line = LineString([(line_points[i][0], line_points[i][1]), (line_points[i+1][0], line_points[i+1][1])])
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