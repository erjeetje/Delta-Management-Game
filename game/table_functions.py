import geopandas as gpd

def get_board_gdf(hexagons):
    return gpd.GeoDataFrame.from_features(hexagons.features)

def update_hexagon_tracker(hexagons_board_gdf, hexagons_tracker_df, update=True):
    hexagons_tracker_df = hexagons_tracker_df.copy()
    hexagons_tracker_df = hexagons_tracker_df.drop(columns=['red_markers', 'blue_markers'])
    hexagons_board_gdf = hexagons_board_gdf.copy()
    hexagons_board_gdf = hexagons_board_gdf[["red_markers", "blue_markers"]]
    hexagons_tracker_df = hexagons_tracker_df.merge(hexagons_board_gdf, left_index=True, right_index=True)
    if update:
        hexagons_tracker_df['changed'] = (
                (hexagons_tracker_df['red_markers'] != hexagons_tracker_df['ref_red_markers']) |
                (hexagons_tracker_df['blue_markers'] != hexagons_tracker_df['ref_blue_markers']))
        print("changed hexagons:")
        print(hexagons_tracker_df[hexagons_tracker_df['changed'] == True])
    return hexagons_tracker_df
