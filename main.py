# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os
import sys
import geopandas as gpd
import contextily as ctx
import matplotlib.collections
import matplotlib.style

matplotlib.style.use("fast")
import process_config_files as model_files
import process_game_files as game_files
import transform_functions as transform_func
import model_to_game as game_sync


from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QMessageBox,
                             QLabel, QDialog, QDesktopWidget, QMainWindow,
                             QHBoxLayout, QVBoxLayout)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
#from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.figure import Figure
from matplotlib.colors import Normalize



class ApplicationWindow(QMainWindow):
    def __init__(self, scenarios, viz_tracker, bbox):
        super().__init__()
        self._main = QWidget()
        self.setWindowTitle('Delta Management Game demonstrator')
        self.setCentralWidget(self._main)
        self.scenarios = scenarios
        self.viz_tracker = viz_tracker
        self.selected_scenario = self.viz_tracker.scenario
        self.selected_variable = self.viz_tracker.variable

        # possible to do: add a Normalize objects for each type of data to the VisualizationTracker class

        self.layout = QHBoxLayout(self._main)
        self.model_canvas = FigureCanvas(Figure(figsize=(5, 5)))
        # Ideally one would use self.addToolBar here, but it is slightly
        # incompatible between PyQt6 and other bindings, so we just add the
        # toolbar as a plain widget instead.
        self.figure_layout = QVBoxLayout(self._main)
        self.figure_layout.addWidget(NavigationToolbar(self.model_canvas, self))
        self.figure_layout.addWidget(self.model_canvas)
        self.layout.addLayout(self.figure_layout)

        self.control_widget = ControlWidget(gui=self, viz_tracker=viz_tracker)
        self.layout.addWidget(self.control_widget)
        self.add_plot_model(bbox)
        return

    def add_plot_model(self, bbox):
        self.ax = self.model_canvas.figure.subplots()
        self.ax.axis(bbox)
        # code below to add a basemap [TODO]
        ctx.add_basemap(self.ax, alpha=0.5, source=ctx.providers.OpenStreetMap.Mapnik)
        #ctx.add_basemap(self.ax, zoom=12, source=ctx.providers.Stamen.TonerLite)
        #ctx.add_basemap(self.ax, source=ctx.providers.OpenStreetMap.Mapnik)
        self.ax.set_axis_off()
        scenario_idx = self.scenarios["scenario"] == self.selected_scenario
        self.running_scenario = self.scenarios[scenario_idx]
        t_idx = self.viz_tracker.time_index
        t = self.running_scenario.iloc[t_idx]["time"]
        idx = self.running_scenario["time"] == t
        self.plot_data = self.running_scenario[idx]
        self.plot_data.plot(column=self.selected_variable, ax=self.ax, cmap="coolwarm")

        pcs = [child for child in self.ax.get_children() if isinstance(child, matplotlib.collections.PathCollection)]
        assert len(pcs) == 1, "expected 1 pathcollection after plotting"
        self.pc = pcs[0]

        self.model_timer = self.model_canvas.new_timer(40)
        self.model_timer.add_callback(self.update_plot_model)
        self.model_timer.start()
        return

    def update_plot_model(self):
        if self.selected_scenario != self.viz_tracker.scenario:
            self.selected_scenario = self.viz_tracker.scenario
            scenario_idx = self.scenarios["scenario"] == self.selected_scenario
            self.running_scenario = self.scenarios[scenario_idx]
            self.pc.set_norm(Normalize())
        if self.selected_variable != self.viz_tracker.variable:
            self.selected_variable = self.viz_tracker.variable
            if self.selected_variable == "water_salinity":
                color_map = "coolwarm"
            elif self.selected_variable == "water_level":
                color_map = "viridis"
            self.pc.set_cmap(color_map)
            self.pc.set_norm(Normalize()) #vmin=0, vmax=1
            return
        t = self.viz_tracker.get_time_index()
        idx = self.running_scenario["time"] == t
        self.plot_data = self.running_scenario[idx]
        self.pc.set_array(self.plot_data[self.selected_variable])
        self.ax.set_title(f"timestep: {t} - scenario {self.selected_scenario}")
        self.model_canvas.draw()
        self.viz_tracker.time_index = 1
        return


class GameVisualization(QWidget):
    def __init__(self, scenarios, viz_tracker):
        super().__init__()
        self.scenarios = scenarios
        self.viz_tracker=viz_tracker
        self.selected_scenario = self.viz_tracker.scenario
        self.selected_variable = self.viz_tracker.variable
        self.setWindowTitle('Game world visualization')
        self.game_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(NavigationToolbar(self.game_canvas, self))
        self.layout.addWidget(self.game_canvas)
        self.add_plot_model()
        return

    def add_plot_model(self):
        self.ax = self.game_canvas.figure.subplots()
        self.ax.set_axis_off()
        scenario_idx = self.scenarios["scenario"] == self.selected_scenario
        self.running_scenario = self.scenarios[scenario_idx]
        t_idx = self.viz_tracker.time_index
        t = self.running_scenario.iloc[t_idx]["time"]
        idx = self.running_scenario["time"] == t
        self.plot_data = self.running_scenario[idx]
        self.plot_data.plot(column=self.selected_variable, ax=self.ax, cmap="coolwarm", aspect=1)

        pcs = [child for child in self.ax.get_children() if isinstance(child, matplotlib.collections.PathCollection)]
        assert len(pcs) == 1, "expected 1 pathcollection after plotting"
        self.pc = pcs[0]

        self.game_timer = self.game_canvas.new_timer(40)
        self.game_timer.add_callback(self.update_plot_model)
        self.game_timer.start()
        return

    def update_plot_model(self):
        if self.selected_scenario != self.viz_tracker.scenario:
            self.selected_scenario = self.viz_tracker.scenario
            scenario_idx = self.scenarios["scenario"] == self.selected_scenario
            self.running_scenario = self.scenarios[scenario_idx]
            self.pc.set_norm(Normalize())
        if self.selected_variable != self.viz_tracker.variable:
            self.selected_variable = self.viz_tracker.variable
            if self.selected_variable == "water_salinity":
                color_map = "coolwarm"
            elif self.selected_variable == "water_level":
                color_map = "viridis"
            self.pc.set_cmap(color_map)
            self.pc.set_norm(Normalize())  # vmin=0, vmax=1
            return
        t = self.viz_tracker.get_time_index()
        idx = self.running_scenario["time"] == t
        self.plot_data = self.running_scenario[idx]
        self.pc.set_array(self.plot_data[self.selected_variable])
        self.ax.set_title(f"timestep: {t} - scenario {self.selected_scenario}")
        self.game_canvas.draw()
        return


class ControlWidget(QWidget):
    def __init__(self, gui, viz_tracker):
        super().__init__()
        self.gui = gui
        self.viz_tracker=viz_tracker
        self.setFixedSize(400, 800)
        #self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.initUI()
        self.show()  # app.exec_()

    def initUI(self):
        #lbl_update = QLabel('Update controls', self)
        #lbl_update.move(10, 40)
        #lbl_update.setFixedWidth(180)
        #lbl_update.setAlignment(Qt.AlignCenter)
        self.lbl_variable = QLabel('Variable selection', self)
        self.lbl_variable.move(10, 180)
        self.lbl_variable.setFixedWidth(380)
        self.lbl_variable.setAlignment(Qt.AlignCenter)

        self.btn_salinity = QPushButton('Salinity concentration', self)
        self.btn_salinity.clicked.connect(self.on_salinity_button_clicked)
        self.btn_salinity.resize(180, 40)
        self.btn_salinity.move(10, 240)
        self.btn_water_level = QPushButton('Water level', self)
        self.btn_water_level.clicked.connect(self.on_water_level_button_clicked)
        self.btn_water_level.resize(180, 40)
        self.btn_water_level.move(210, 240)

        self.lbl_boundary = QLabel('Boundary conditions', self)
        self.lbl_boundary.move(10, 480)
        self.lbl_boundary.setFixedWidth(380)
        self.lbl_boundary.setAlignment(Qt.AlignCenter)

        self.scenario4 = QPushButton('+3m SLR, 500 m/3s', self)
        self.scenario4.clicked.connect(self.on_scenario4_button_clicked)
        self.scenario4.resize(180, 40)
        self.scenario4.move(10, 540)
        self.scenario3 = QPushButton('+3m SLR, 2000 m/3s', self)
        self.scenario3.clicked.connect(self.on_scenario3_button_clicked)
        self.scenario3.resize(180, 40)
        self.scenario3.move(210, 540)
        self.scenario2 = QPushButton('+0m SLR, 500 m/3s', self)
        self.scenario2.clicked.connect(self.on_scenario2_button_clicked)
        self.scenario2.resize(180, 40)
        self.scenario2.move(10, 620)
        self.scenario1 = QPushButton('+0m SLR, 2000 m/3s', self)
        self.scenario1.clicked.connect(self.on_scenario1_button_clicked)
        self.scenario1.resize(180, 40)
        self.scenario1.move(210, 620)
        return

    def on_salinity_button_clicked(self):
        #print("Changing to salinity visualization")
        self.viz_tracker.variable = "water_salinity"
        return

    def on_water_level_button_clicked(self):
        #print("Changing to salinity visualization")
        #self.script.update("unit", "water_level")
        self.viz_tracker.variable = "water_level"
        return

    def on_scenario1_button_clicked(self):
        #print("Changing to scenario 1")
        #self.script.update("scenario", "1")
        self.viz_tracker.scenario = "0_0mzss_2000m3s"
        return

    def on_scenario2_button_clicked(self):
        #print("Changing to scenario 2")
        #self.script.update("scenario", "2")
        self.viz_tracker.scenario = "0_0mzss_0500m3s"
        return

    def on_scenario3_button_clicked(self):
        #print("Changing to scenario 3")
        #self.script.update("scenario", "3")
        self.viz_tracker.scenario = "3mzss_2000m3s"
        return

    def on_scenario4_button_clicked(self):
        #print("Changing to scenario 4")
        #self.script.update("scenario", "4")
        self.viz_tracker.scenario = "3mzss_0500m3s"
        return

class VisualizationTracker():
    def __init__(self, starting_scenario, starting_variable, time_steps, starting_time):
        self._scenario = starting_scenario
        self._variable = starting_variable
        self._time_steps = time_steps
        self._time_index = starting_time
        return

    def get_time_index(self):
        t_idx = self.time_index % len(self.time_steps)
        return self.time_steps[t_idx]

    @property
    def scenario(self):
        return self._scenario

    @property
    def variable(self):
        return self._variable

    @property
    def time_steps(self):
        return self._time_steps

    @property
    def time_index(self):
        return self._time_index

    @scenario.setter
    def scenario(self, scenario):
        self._scenario = scenario
        return

    @variable.setter
    def variable(self, variable):
        self._variable = variable
        return

    @time_steps.setter
    def time_steps(self, time_steps):
        self._time_steps = time_steps
        return

    @time_index.setter
    def time_index(self, time_index):
        self._time_index += time_index
        return




class model_locations():
    def __init__(self):
        super(model_locations, self).__init__()
        self.load_variables()
        #self.update_initial_variables()
        return

    def load_variables(self):
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.model_path = r"C:\Users\Robert-Jan\Dropbox\Onderzoeker map\Salti game design\Prototyping\sobek-rmm-vzm-j15_5-v4\sobek-rmm-vzm-j15_5-v4.dsproj_data\rmm_output\dflow1d"
        self.input_path = r"C:\Users\Robert-Jan\Dropbox\Onderzoeker map\Salti game design\Prototyping\RMM coding\demonstrator_input_files"
        # change the directory below to what works
        self.save_path = r"C:\Users\Robert-Jan\Dropbox\Onderzoeker map\Salti game design\Prototyping\RMM coding\demonstrator_output_files"

        self.test = True
        self.unit = "salinity"
        self.scenario = "1"
        return

        # initial model variables
        self.subbranches = game_files.get_subbranches()
        self.branches_model_gdf, self.nodes_model_gdf, self.grid_points_model_gdf = model_files.process_nodes_branches(
            self.model_path)
        self.branches_model_gdf, self.nodes_game_gdf = game_sync.determine_main_branches(
            self.branches_model_gdf, self.subbranches, self.nodes_model_gdf)
        """
        It would make more sense if the obs_points are first only loaded and updated afterwards
        """
        self.merged_branches_model_gdf = model_files.merge_subbranches(self.branches_model_gdf, self.subbranches)
        self.obs_points_model_gdf = model_files.process_obs_points(
            self.model_path, self.branches_model_gdf, self.merged_branches_model_gdf, self.subbranches)
        self.grid_points_model_gdf = model_files.process_cross_sections(
            self.model_path, self.grid_points_model_gdf, self.branches_model_gdf)
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
            save_gdf5 = self.grid_points_model_gdf
            save_gdf5.to_csv(os.path.join(self.save_path, "model_grid_points.csv"), index=False)
            save_gdf5.to_file(os.path.join(self.save_path, "model_grid_points.geojson"))
        return


    def update(self, type_update, variable):
        if type_update == "unit":
            self.unit = variable
        elif type_update == "scenario":
            self.scenario = variable
        print("received", type_update, "and", variable)
        return


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
    scenario_model_file = os.path.join(scenario_location, "obs_model_all_scenario_1_day.gpkg")
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
    scenario_game_file = os.path.join(scenario_location, "obs_game_all_scenario_1_day.gpkg")
    obs_points_game_gdf = gpd.read_file(scenario_game_file)
    return obs_points_model_gdf, obs_points_game_gdf, world_bbox

def main():
    model_scenarios, game_scenarios, world_bbox = load_scenarios()
    qapp = QApplication.instance()
    if not qapp:
        qapp = QApplication(sys.argv)
    time_steps = list(sorted(set(model_scenarios["time"])))
    time_index = 0
    starting_scenario = "0_0mzss_2000m3s"
    starting_variable = "water_salinity"
    viz_tracker = VisualizationTracker(
        starting_scenario=starting_scenario, starting_variable=starting_variable,
        time_steps=time_steps, starting_time=time_index)
    gui = ApplicationWindow(
        scenarios=model_scenarios, viz_tracker=viz_tracker, bbox=world_bbox)
    side_window = GameVisualization(
        scenarios=game_scenarios, viz_tracker=viz_tracker)
    gui.show()
    side_window.show()
    gui.activateWindow()
    gui.raise_()
    qapp.exec()
    #locations = model_locations()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()