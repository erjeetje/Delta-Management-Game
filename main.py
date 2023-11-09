# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os
import process_config_files as model_files
import process_game_files as game_files
import transform_functions as transform_func
import model_to_game as game_sync

class model_locations():
    def __init__(self):
        super(model_locations, self).__init__()
        self.load_variables()
        self.update_initial_variables()
        return

    def load_variables(self):
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.input_path = os.path.join(self.dir_path, 'input_files')
        # change the directory below to what works
        self.save_path = r"C:\Werkzaamheden\Onderzoek\2 SaltiSolutions\03 Low-fi game design\RMM coding (notebooks)\demonstrator_output_check_test"

        self.test = True

        # initial model variables
        self.subbranches = game_files.get_subbranches()
        self.branches_model_gdf, self.nodes_model_gdf, self.grid_points_model_gdf = model_files.process_nodes_branches(
            self.input_path)
        self.branches_model_gdf, self.nodes_game_gdf = game_sync.determine_main_branches(
            self.branches_model_gdf, self.subbranches, self.nodes_model_gdf)
        """
        It would make more sense if the obs_points are first only loaded and updated afterwards
        """
        self.merged_branches_model_gdf = model_files.merge_subbranches(self.branches_model_gdf, self.subbranches)
        self.obs_points_model_gdf = model_files.process_obs_points(
            self.input_path, self.branches_model_gdf, self.merged_branches_model_gdf, self.subbranches)
        self.grid_points_model_gdf = model_files.process_cross_sections(
            self.input_path, self.grid_points_model_gdf, self.branches_model_gdf)
        """
        print("branches")
        print(self.branches_model_gdf)
        print("nodes")
        print(self.nodes_model_gdf)
        print("obs_points")
        print(self.obs_points_model_gdf)
        print("grid points with cross sections")
        print(self.grid_points_model_gdf)
        """

        # initial game variables
        self.model_polygons = game_files.read_json_features(
            filename="hexagon_shapes_warped_new.json", path=self.input_path)
        self.game_hexagons = game_files.read_geojson(
            filename='hexagons_clean0.geojson', path=self.input_path)
        bbox = transform_func.get_bbox(self.model_polygons)
        self.transform_calibration = transform_func.create_calibration_file(bbox=bbox, save=False, path="")
        return

    def update_initial_variables(self):
        self.game_hexagons = game_files.add_geometry_dimension(self.game_hexagons) # possibly can be removed later with live connection
        self.model_polygons = game_files.add_geometry_dimension(self.model_polygons) # possibly can be removed later with live connection
        self.model_polygons = transform_func.transform(self.model_polygons, self.transform_calibration,
                                                        export="warped", path="")
        self.model_polygons, self.obs_points_model_gdf = game_files.index_points_to_polygons(
            self.model_polygons, self.obs_points_model_gdf)
        self.game_hexagons = game_files.find_neighbours(self.game_hexagons)
        self.game_hexagons = game_files.find_neighbour_edges(self.game_hexagons) # not so sure this is still needed
        self.game_hexagons = game_files.match_hexagon_properties(
            self.game_hexagons, self.model_polygons, "obs_ids")
        self.model_polygons = game_files.match_hexagon_properties(
            self.model_polygons, self.game_hexagons, "neighbours")

        #self.branches_game_gdf, self.nodes_game_gdf = game_sync.determine_main_branches(
        #    self.branches_model_gdf, self.subbranches, self.nodes_model_gdf)

        self.obs_points_model_gdf, self.model_polygons = game_sync.obs_points_to_polygons(
            self.obs_points_model_gdf, self.model_polygons)
        self.obs_points_model_gdf = game_sync.update_obs_points(self.obs_points_model_gdf)
        self.merged_branches_model_gdf = game_sync.obs_points_per_branch(
            self.merged_branches_model_gdf, self.obs_points_model_gdf)
        self.merged_branches_model_gdf = game_sync.determine_polygon_intersections(
            self.merged_branches_model_gdf, self.model_polygons)
        self.merged_branches_game_gdf = game_sync.draw_branch_network(
            self.game_hexagons, self.merged_branches_model_gdf)
        self.obs_points_game_gdf = game_sync.create_game_obs_points(
            self.obs_points_model_gdf, self.merged_branches_game_gdf)
        """

        self.model_polygons, self.branches_model_gdf = game_sync.find_branch_intersections(
            self.model_polygons, self.branches_model_gdf)

        self.game_hexagons = game_files.match_hexagon_properties(
            self.game_hexagons,  self.model_polygons, ["branches", "branch_crossing"])

        self.nodes_game_gdf = game_sync.match_nodes(self.model_polygons, self.nodes_game_gdf, self.subbranches)
        """

        if self.test:
            save_gdf = self.merged_branches_game_gdf
            save_gdf.to_csv(os.path.join(self.save_path, "game_merged_branches.csv"), index=False)
            save_gdf["hexagon_ids"] = save_gdf["hexagon_ids"].astype(str)
            save_gdf.to_file(os.path.join(self.save_path, "game_branch_network_test.geojson"))
            game_files.save_geojson(self.game_hexagons, filename="game_hexagons_test.geojson", path=self.save_path)
            game_files.save_geojson(self.model_polygons, filename="model_polygons_test.geojson", path=self.save_path)
            save_gdf2 = self.merged_branches_model_gdf
            save_gdf2.to_csv(os.path.join(self.save_path, "model_branches.csv"), index=False)
            save_gdf2["polygon_ids"] = save_gdf2["polygon_ids"].astype(str)
            save_gdf2.to_file(os.path.join(self.save_path, "model_merged_branches.geojson"))
            self.branches_model_gdf.to_file(os.path.join(self.save_path, "separate_branches.geojson"))
            save_gdf3 = self.obs_points_model_gdf
            save_gdf3.to_csv(os.path.join(self.save_path, "model_obs_points.csv"), index=False)
            save_gdf3["prev_branches"] = save_gdf3["prev_branches"].astype(str)
            save_gdf3.to_file(os.path.join(self.save_path, "model_observation_points.geojson"))
            save_gdf4 = self.obs_points_game_gdf
            save_gdf4.to_csv(os.path.join(self.save_path, "game_obs_points.csv"), index=False)
            save_gdf4["prev_branches"] = save_gdf4["prev_branches"].astype(str)
            save_gdf4.to_file(os.path.join(self.save_path, "game_observation_points.geojson"))
        return




def main():
    locations = model_locations()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()