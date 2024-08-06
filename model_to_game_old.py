import pandas as pd
import geopandas as gpd
import numpy as np
import geojson
from shapely import geometry, intersects, get_coordinates, line_interpolate_point


def determine_main_branches(branches, subbranches, nodes):
    """
    code below is for branches and nodes as geojsons
    for node in nodes.features:
        node.properties["branches"] = []
    for branch in branches.features:
        # if any(branch.properties["id"] in sub_branch for sub_branch in subbranches_list):
        branch.properties["main_branch"] = False
        #if any(branch.properties["id"] in subbranch for subbranch in subbranches_list):
        if any(branch.properties["id"] in name for name in subbranches.values()):
            branch.properties["main_branch"] = True
        for node in nodes.features:
            if branch.properties["fromnode"] == node.properties["id"]:
                node.properties["branches"].append(branch.properties["id"])
            if branch.properties["tonode"] == node.properties["id"]:
                node.properties["branches"].append(branch.properties["id"])
    for node in nodes.features:
        node.properties["main_branch"] = False
        for branch_id in node.properties["branches"]:
            #if any(branch_id in subbranch for subbranch in subbranches_list):
            if any(branch_id in name for name in subbranches.values()):
                node.properties["main_branch"] = True
                break
    """
    nodes["branches"] = np.empty((len(nodes), 0)).tolist()
    branches["main_branch"] = False
    for index, branch in branches.iterrows():
        if any(branch["id"] in name for name in subbranches.values()):
            branches.loc[index, "main_branch"] = True
        for key, node in nodes.iterrows():
            if branch["fromnode"] == node["id"]:
                nodes.loc[key, "branches"].append(branch["id"])
            if branch["tonode"] == node["id"]:
                nodes.loc[key, "branches"].append(branch["id"])
    nodes["main_branch"] = False
    for key, node in nodes.iterrows():
        for branch_id in node["branches"]:
            # if any(branch_id in subbranch for subbranch in subbranches_list):
            if any(branch_id in name for name in subbranches.values()):
                nodes.loc[key, "main_branch"] = True
                break
    return branches, nodes

def determine_polygon_intersections(branches_gdf, polygons):
    def determine_polygons(line, polygons):
        """
        The nested for loops make this function rather slow, but it does correctly map the order of polygons
        """
        intersecting_polygons = []
        line_points = get_coordinates(line)
        for line_point in line_points:
            point = geometry.Point(line_point)
            for polygon in polygons.features:
                shape = geometry.shape(polygon.geometry)
                if shape.contains(point):
                    if polygon.id not in intersecting_polygons:
                        intersecting_polygons.append(polygon.id)
        if intersecting_polygons:
            return intersecting_polygons
        else:
            return np.nan

    branches_gdf["polygon_ids"] = branches_gdf.apply(
        lambda row: determine_polygons(row["geometry"], polygons), axis=1)
    return branches_gdf.dropna()


def draw_branch_network(hexagons, branches_gdf):
    # possibly, below could also be done with branches_gdf.drop(columns="geometry"),
    # but not sure if it then also drops crs
    branches_properties_df = branches_gdf[["id", "polygon_ids", "obs_count", "start_node", "end_node"]]
    branches_properties_df = branches_properties_df.set_index("id")
    for index, branch in branches_properties_df.iterrows():
        line_points = []
        for i in range(len(branch["polygon_ids"])):
            hexagon_ref = hexagons.features[branch["polygon_ids"][i]]
            shape = geometry.shape(hexagon_ref.geometry)
            mid_point = np.asarray(shape.centroid.xy)
            line_points.append(mid_point)
        if len(line_points) > 1:
            branch_line = geometry.LineString(line_points)
            branches_properties_df.loc[index, "geometry"] = branch_line
            branches_properties_df.loc[index, "distance"] = branch_line.length
        else:
            branches_properties_df.loc[index, "geometry"] = np.nan
            branches_properties_df.loc[index, "distance"] = np.nan
    branches_game_gdf = gpd.GeoDataFrame(branches_properties_df, geometry=branches_properties_df["geometry"])
    branches_game_gdf = branches_game_gdf.reset_index()
    branches_game_gdf = branches_game_gdf.rename(columns={"polygon_ids": "hexagon_ids"})
    return branches_game_gdf


def create_game_obs_points(obs_points_model_gdf, branches_game_gdf):
    def update_obs_point_geometry(obs_branch_id, obs_branch_rank, branches):
        branches = branches.set_index("id")
        branch_dist = branches.loc[obs_branch_id, "distance"]
        branch_obs_count = branches.loc[obs_branch_id, "obs_count"]
        branch_spacing = branch_dist / branch_obs_count
        obs_to_line = -0.5 * branch_spacing + obs_branch_rank * branch_spacing
        point = line_interpolate_point(branches.loc[obs_branch_id, "geometry"], obs_to_line)
        return point

    obs_points_game_df = obs_points_model_gdf.drop(columns="geometry")
    obs_points_game_df["geometry"] = obs_points_game_df.apply(
        lambda row: update_obs_point_geometry(row["branch_id_game"], row["branch_rank"], branches_game_gdf), axis=1)
    obs_points_game_gdf = gpd.GeoDataFrame(obs_points_game_df, geometry=obs_points_game_df["geometry"])
    return obs_points_game_gdf


def draw_branch_network_geojson(hexagons, branches_gdf):
    branch_network = []
    print(list(branches_gdf.columns))
    for index, branch in branches_gdf.iterrows():
        line_points = []
        for i in range(len(branch["polygon_ids"])):
            hexagon_ref = hexagons.features[branch["polygon_ids"][i]]
            shape = geometry.shape(hexagon_ref.geometry)
            mid_point = np.asarray(shape.centroid.xy)
            line_points.append(mid_point)
            """
            try:
                hexagon_next = hexagons.features[branch["polygon_ids"][i+1]]
                shape = geometry.shape(hexagon_ref.geometry)
                mid_point_next = np.asarray(shape.centroid.xy)
                edge_point = np.asarray((mid_point[0]+mid_point_next[0])/2, (mid_point[1]+mid_point_next[1])/2)
                line_points.append(edge_point)
            except IndexError:
                break
            """
        if len(line_points) > 1:
            branch_line = geometry.LineString(line_points)
            branch_property = {"hexagon_ids": branch["polygon_ids"]}
            branch_feature = geojson.Feature(id=branch["id"], geometry=branch_line, properties=branch_property)
            branch_network.append(branch_feature)
    branch_network = geojson.FeatureCollection(branch_network)
    return branch_network


def obs_points_to_polygons(obs_points_gdf, polygons):
    obs_points_grouped = obs_points_gdf.groupby("polygon_id")
    for polygon in polygons.features:
        polygon.properties["obs_ids"] = []
        try:
            polygon.properties["obs_ids"] = obs_points_grouped.get_group(polygon.id)["obs_id"].to_list()
        except KeyError:
            continue
    return obs_points_gdf, polygons


def update_obs_points(obs_points_gdf):
    obs_points_gdf['obs_count'] = obs_points_gdf.groupby('branch_id_game')['branch_id_game'].transform('count')
    obs_points_gdf['branch_rank'] = obs_points_gdf.groupby('branch_id_game')['chainage_game'].rank(ascending=True)
    obs_points_gdf['branch_rank'] = obs_points_gdf['branch_rank'].astype('int')
    obs_points_gdf['fractional_chainage'] = obs_points_gdf['chainage_game'] / obs_points_gdf['distance']
    return obs_points_gdf


def obs_points_per_branch(branches_gdf, obs_points_gdf):
    def get_obs_count(branch_id, obs_group):
        try:
            obs_count = obs_group.get_group(branch_id).iloc[0]["obs_count"]
        except KeyError:
            obs_count = 0
        return obs_count
    obs_group = obs_points_gdf.groupby("branch_id_game")
    branches_gdf["obs_count"] = branches_gdf.apply(lambda row: get_obs_count(row["id"], obs_group), axis=1)
    return branches_gdf


def find_branch_intersections(polygons, branches):
    for polygon in polygons.features:
        polygon.properties["branches"] = []
        polygon.properties["branch_crossing"] = {}
        polygon.properties["branch_crossing_coor"] = {}
        shape = geometry.shape(polygon.geometry)
        for index, branch in branches.iterrows():
            if branch["main_branch"]:
                if intersects(shape, branch["geometry"]):
                    polygon.properties["branches"].append(branch["id"])
                    polygon_bb = shape.boundary.coords
                    linestrings = [geometry.LineString(polygon_bb[k:k+2]) for k in range(len(polygon_bb) - 1)]
                    for i, linestring in enumerate(linestrings):
                        if intersects(linestring, branch["geometry"]):
                            midpoint = geometry.Point(
                                (linestring.coords[0][0] + linestring.coords[-1][0]) / 2,
                                (linestring.coords[0][1] + linestring.coords[-1][1]) / 2)
                            inters = linestring.intersection(branch["geometry"])
                            if inters.geom_type == "Point":
                                coords = np.asarray(inters.coords.xy)
                            elif inters.geom_type == "MultiPoint":
                                coords = np.asarray(l.coords.xy for l in inters.geoms)
                            else:
                                print(inters.geom_type)
                            for neighbour in polygon.properties["neighbours"]:
                                neighbour_polygon = polygons.features[neighbour]
                                neighbour_shape = geometry.shape(neighbour_polygon.geometry)
                                neighbour_bb = neighbour_shape.boundary.coords
                                neighbour_linestrings = [geometry.LineString(
                                    neighbour_bb[k:k+2]) for k in range(len(neighbour_bb) - 1)]
                                for neighbour_linestring in neighbour_linestrings:
                                    neighbour_midpoint = geometry.Point(
                                        (neighbour_linestring.coords[0][0] + neighbour_linestring.coords[-1][0]) / 2,
                                        (neighbour_linestring.coords[0][1] + neighbour_linestring.coords[-1][1]) / 2)
                                    if neighbour_midpoint.distance(midpoint) < 50:
                                        polygon.properties["branch_crossing"][neighbour_polygon.id] = branch["id"]
                                        #print("branch crossing between", polygon.id, "and", neighbour_polygon.id)
                                        #print(branch.properties["id"], coords)
                                        #polygon.properties["branch_crossing_coor"][branch.properties["id"]] = coords
    return polygons, branches


def match_nodes(polygons, nodes, subbranches):
    branch_names = [name for subbranch in subbranches.values() for name in subbranch]
    nodes["burification"] = False
    nodes["polygon_id"] = None
    for index, node in nodes.iterrows():
        branch_number = len(set(node["branches"]).intersection(branch_names))
        if ((branch_number != 0) and (branch_number != 2)):
            nodes.loc[index, "burification"] = True
            point = node.geometry
            for polygon in polygons.features:
                shape = geometry.shape(polygon.geometry)
                if shape.contains(point):
                    nodes.loc[index, "polygon_id"] = polygon.id
    return nodes


def draw_branch_network_old(hexagons):
    branch_network = []
    for hexagon in hexagons.features:
        shape = geometry.shape(hexagon.geometry)
        point1 = np.asarray(shape.centroid.xy)
        point1 = point1
        for i, (neighbour_id, branch) in enumerate(hexagon.properties["branch_crossing"].items()):
            midpoint_index = hexagon.properties["neighbours"].index(neighbour_id)
            point2 = np.asarray(hexagon.properties["neighbour_edge_midpoint"][midpoint_index])
            branch_line = geometry.LineString([point1, point2])
            branch_property = {"branch": branch}
            branch_feature = geojson.Feature(id=i, geometry=branch_line,
                                      properties=branch_property)
            branch_network.append(branch_feature)
    branch_network = geojson.FeatureCollection(branch_network)
    return branch_network