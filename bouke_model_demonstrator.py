# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os
import sys
import geopandas as gpd
import contextily as ctx
import matplotlib.collections
import matplotlib.pyplot as plt
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
from PyQt5.QtGui import QPixmap
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
#from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm, Normalize, CenteredNorm
from matplotlib.cm import ScalarMappable



class ApplicationWindow(QMainWindow):
    def __init__(self, scenarios, viz_tracker, bbox, salinity_colorbar_image, water_level_colorbar_image,
                 water_velocity_image):
        super().__init__()
        self._main = QWidget()
        self.setWindowTitle('Delta Management Game demonstrator')
        self.setCentralWidget(self._main)
        self.setStyleSheet("background-color:white; font-weight: bold; font-size: 24")
        self.scenarios = scenarios
        self.viz_tracker = viz_tracker
        self.selected_scenario = self.viz_tracker.scenario
        self.selected_model_variable = self.viz_tracker.model_variable
        self.selected_game_variable = self.viz_tracker.game_variable
        self.salinity_colorbar_image = salinity_colorbar_image
        self.water_level_colorbar_image = water_level_colorbar_image
        self.water_velocity_image = water_velocity_image
        #self.basemap_image = basemap_image

        self.layout = QHBoxLayout(self._main)
        #self.fig, self.ax2 = plt.subplots()
        #self.ax2.imshow(basemap_image)
        self.model_canvas = FigureCanvas(Figure()) #figsize=(5, 5)
        # Ideally one would use self.addToolBar here, but it is slightly
        # incompatible between PyQt6 and other bindings, so we just add the
        # toolbar as a plain widget instead.
        self.figure_layout = QVBoxLayout(self._main)
        #self.figure_layout.addWidget(NavigationToolbar(self.model_canvas, self))
        self.figure_layout.addWidget(self.model_canvas, stretch=1)

        self.colorbar_title_layout = QHBoxLayout(self._main)
        self.colorbar_model_title_label = QLabel(self._main)
        self.colorbar_model_title_label.setText("Screen colorbar")
        self.colorbar_model_title_label.setStyleSheet("font-weight: bold; font-size: 36")
        self.colorbar_title_layout.addWidget(self.colorbar_model_title_label, alignment=Qt.AlignCenter)
        self.colorbar_game_title_label = QLabel(self._main)
        self.colorbar_game_title_label.setText("Board colorbar")
        self.colorbar_game_title_label.setStyleSheet("font-weight: bold; font-size: 36")
        self.colorbar_title_layout.addWidget(self.colorbar_game_title_label, alignment=Qt.AlignCenter)
        self.figure_layout.addLayout(self.colorbar_title_layout)

        self.colorbar_layout = QHBoxLayout(self._main)
        self.colorbar_model_label = QLabel(self._main)
        self.colorbar_model_label.setPixmap(self.salinity_colorbar_image)
        self.colorbar_model_label.resize(self.salinity_colorbar_image.width(), self.salinity_colorbar_image.height())
        self.colorbar_layout.addWidget(self.colorbar_model_label, alignment=Qt.AlignCenter)
        self.colorbar_game_label = QLabel(self._main)
        self.colorbar_game_label.setPixmap(self.salinity_colorbar_image)
        self.colorbar_game_label.resize(self.salinity_colorbar_image.width(), self.salinity_colorbar_image.height())
        self.colorbar_layout.addWidget(self.colorbar_game_label, alignment=Qt.AlignCenter)
        self.figure_layout.addLayout(self.colorbar_layout)
        self.layout.addLayout(self.figure_layout)

        self.control_widget = ControlWidget(gui=self, viz_tracker=viz_tracker)
        self.layout.addWidget(self.control_widget)
        self.add_plot_model(bbox)
        return

    def add_plot_model(self, bbox):
        self.ax = self.model_canvas.figure.subplots()
        #self.ax.axis(bbox)
        #ctx.add_basemap(self.ax, alpha=0.5, source=ctx.providers.CartoDB.PositronNoLabels)
        #ctx.add_basemap(self.ax, source=ctx.providers.Esri.WorldGrayCanvas, zoom=12)
        #ctx.add_basemap(self.ax, alpha=0.5, source=ctx.providers.OpenStreetMap.Mapnik)
        self.ax.set_axis_off()
        self.ax.set_position([0.1, 0.1, 0.8, 0.8])
        scenario_idx = self.scenarios["scenario"] == self.selected_scenario
        self.running_scenario = self.scenarios[scenario_idx]
        t_idx = self.viz_tracker.time_index
        t = self.running_scenario.iloc[t_idx]["time"]
        idx = self.running_scenario["time"] == t
        self.plot_data = self.running_scenario[idx]
        self.plot_data.plot(column=self.selected_model_variable, ax=self.ax, cmap="RdBu_r", markersize=150.0)

        pcs = [child for child in self.ax.get_children() if isinstance(child, matplotlib.collections.PathCollection)]
        assert len(pcs) == 1, "expected 1 pathcollection after plotting"
        self.pc = pcs[0]
        self.pc.set_norm(self.viz_tracker.salinity_norm)
        # code below adds a colorbar to the image, but makes the plots too slow
        #self.colorbar = ScalarMappable(self.viz_tracker.water_level_norm, cmap="Blues_r")
        #self.model_canvas.figure.colorbar(self.colorbar, ax=self.ax)

        self.model_timer = self.model_canvas.new_timer(40)
        self.model_timer.add_callback(self.update_plot_model)
        self.model_timer.start()
        return

    def update_plot_model(self):
        if self.selected_scenario != self.viz_tracker.scenario:
            self.selected_scenario = self.viz_tracker.scenario
            scenario_idx = self.scenarios["scenario"] == self.selected_scenario
            self.running_scenario = self.scenarios[scenario_idx]
        if self.selected_model_variable != self.viz_tracker.model_variable:
            self.selected_model_variable = self.viz_tracker.model_variable
            if self.selected_model_variable == "water_salinity":
                color_map = "RdBu_r"
                norm = self.viz_tracker.salinity_norm
            elif self.selected_model_variable == "water_level":
                color_map = "viridis_r"
                norm = self.viz_tracker.water_level_norm
            elif self.selected_model_variable == "water_velocity":
                color_map = "Spectral_r"
                norm = self.viz_tracker.water_velocity_norm
            elif self.selected_model_variable == "water_depth":
                color_map = "Blues_r"
                norm = self.viz_tracker.water_depth_norm
            self.pc.set_cmap(color_map)
            self.pc.set_norm(norm)
            self.update_colorbars(to_update="model")
        if self.selected_game_variable != self.viz_tracker.game_variable:
            self.selected_game_variable = self.viz_tracker.game_variable
            self.update_colorbars(to_update="game")
        t = self.viz_tracker.get_time_index()
        idx = self.running_scenario["time"] == t
        self.plot_data = self.running_scenario[idx]
        self.pc.set_array(self.plot_data[self.selected_model_variable])
        title_string = "scenario: " + self.selected_scenario + " - " + f"timestep: {t}"
        self.ax.set_title(title_string[:-8])
        self.model_canvas.draw()
        self.viz_tracker.time_index = 1
        return

    def update_colorbars(self, to_update="model"):
        if to_update == "model":
            if self.selected_model_variable == "water_salinity":
                self.colorbar_model_label.setPixmap(self.salinity_colorbar_image)
            if self.selected_model_variable == "water_level":
                self.colorbar_model_label.setPixmap(self.water_level_colorbar_image)
            if self.selected_model_variable == "water_velocity":
                self.colorbar_model_label.setPixmap(self.water_velocity_image)
        elif to_update == "game":
            if self.selected_game_variable == "water_salinity":
                self.colorbar_game_label.setPixmap(self.salinity_colorbar_image)
            if self.selected_game_variable == "water_level":
                self.colorbar_game_label.setPixmap(self.water_level_colorbar_image)
            if self.selected_game_variable == "water_velocity":
                self.colorbar_game_label.setPixmap(self.water_velocity_image)
        return


class GameVisualization(QWidget):
    def __init__(self, scenarios, viz_tracker, bbox):
        super().__init__()
        self.setStyleSheet("background-color:white;")
        #self.setFixedSize(1280, 720)
        #self.setWindowFlags(Qt.FramelessWindowHint)
        self.scenarios = scenarios
        self.viz_tracker=viz_tracker
        self.selected_scenario = self.viz_tracker.scenario
        self.selected_variable = self.viz_tracker.game_variable
        self.setWindowTitle('Game world visualization')
        x_min = -50
        x_max = 1445
        y_min = -20
        y_max = 1020
        dpi=96
        bbox = [x_min,
                x_max,
                y_min,
                y_max]
        self.game_canvas = FigureCanvas(Figure(figsize=((x_max-x_min)/dpi,(y_max-y_min)/dpi), dpi=dpi, tight_layout=True))
        self.game_canvas.setParent(self)
        #self.game_canvas.updateGeometry(self)
        self.layout = QVBoxLayout(self)
        #self.layout.addWidget(NavigationToolbar(self.game_canvas, self))
        self.layout.addWidget(self.game_canvas, stretch=1)

        display_monitor = 1
        monitor = QDesktopWidget().screenGeometry(display_monitor)
        self.move(monitor.left(), monitor.top())
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.showFullScreen()

        self.add_plot_model(bbox)
        return

    def add_plot_model(self, bbox):
        self.ax = self.game_canvas.figure.subplots()
        #self.ax.set_aspect(1)
        self.ax.axis(bbox)
        self.ax.set_axis_off()
        scenario_idx = self.scenarios["scenario"] == self.selected_scenario
        self.running_scenario = self.scenarios[scenario_idx]
        t_idx = self.viz_tracker.time_index
        t = self.running_scenario.iloc[t_idx]["time"]
        idx = self.running_scenario["time"] == t
        self.plot_data = self.running_scenario[idx]
        self.plot_data.plot(column=self.selected_variable, ax=self.ax, cmap="RdBu_r", aspect=1, markersize=200.0)

        pcs = [child for child in self.ax.get_children() if isinstance(child, matplotlib.collections.PathCollection)]
        assert len(pcs) == 1, "expected 1 pathcollection after plotting"
        self.pc = pcs[0]
        self.pc.set_norm(self.viz_tracker.salinity_norm)

        self.game_timer = self.game_canvas.new_timer(40)
        self.game_timer.add_callback(self.update_plot_model)
        self.game_timer.start()
        return

    def update_plot_model(self):
        if self.selected_scenario != self.viz_tracker.scenario:
            self.selected_scenario = self.viz_tracker.scenario
            scenario_idx = self.scenarios["scenario"] == self.selected_scenario
            self.running_scenario = self.scenarios[scenario_idx]
        if self.selected_variable != self.viz_tracker.game_variable:
            self.selected_variable = self.viz_tracker.game_variable
            if self.selected_variable == "water_salinity":
                color_map = "RdBu_r"
                norm = self.viz_tracker.salinity_norm
            elif self.selected_variable == "water_level":
                color_map = "viridis_r"
                norm = self.viz_tracker.water_level_norm
            elif self.selected_variable == "water_velocity":
                color_map = "Spectral_r"
                norm = self.viz_tracker.water_velocity_norm
            elif self.selected_variable == "water_depth":
                color_map = "Blues_r"
                norm = self.viz_tracker.water_depth_norm
            self.pc.set_cmap(color_map)
            self.pc.set_norm(norm)
            return
        t = self.viz_tracker.get_time_index()
        idx = self.running_scenario["time"] == t
        self.plot_data = self.running_scenario[idx]
        self.pc.set_array(self.plot_data[self.selected_variable])
        #self.ax.set_title(f"timestep: {t} - scenario {self.selected_scenario}")
        self.game_canvas.draw()
        return


class ControlWidget(QWidget):
    def __init__(self, gui, viz_tracker):
        super().__init__()
        #self.setStyleSheet("background-color:grey;")
        self.gui = gui
        self.viz_tracker = viz_tracker
        self.setFixedSize(400, 900)
        #self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.screen_highlight = None
        self.board_highlight = None
        self.scenario_highlight = None
        self.initUI()
        self.change_highlights()
        self.show()  # app.exec_()

    def initUI(self):
        self.lbl_screen_variable = QLabel('Screen variable selection', self)
        self.lbl_screen_variable.setStyleSheet("font-weight: bold; font-size: 24")
        self.lbl_screen_variable.move(10, 80)
        self.lbl_screen_variable.setFixedWidth(180)
        self.lbl_screen_variable.setAlignment(Qt.AlignCenter)

        self.btn_screen_salinity = QPushButton('Salinity concentration', self)
        self.btn_screen_salinity.clicked.connect(self.on_screen_salinity_button_clicked)
        self.btn_screen_salinity.resize(180, 80)
        self.btn_screen_salinity.move(10, 140)
        self.btn_screen_water_level = QPushButton('Water level', self)
        self.btn_screen_water_level.clicked.connect(self.on_screen_water_level_button_clicked)
        self.btn_screen_water_level.resize(180, 80)
        self.btn_screen_water_level.move(10, 260)
        self.btn_screen_water_velocity = QPushButton('Water velocity', self)
        self.btn_screen_water_velocity.clicked.connect(self.on_screen_water_velocity_button_clicked)
        self.btn_screen_water_velocity.resize(180, 80)
        self.btn_screen_water_velocity.move(10, 380)

        self.lbl_board_variable = QLabel('Board variable selection', self)
        self.lbl_board_variable.setStyleSheet("font-weight: bold; font-size: 24")
        self.lbl_board_variable.move(210, 80)
        self.lbl_board_variable.setFixedWidth(180)
        self.lbl_board_variable.setAlignment(Qt.AlignCenter)

        self.btn_board_salinity = QPushButton('Salinity concentration', self)
        self.btn_board_salinity.clicked.connect(self.on_board_salinity_button_clicked)
        self.btn_board_salinity.resize(180, 80)
        self.btn_board_salinity.move(210, 140)
        self.btn_board_water_level = QPushButton('Water level', self)
        self.btn_board_water_level.clicked.connect(self.on_board_water_level_button_clicked)
        self.btn_board_water_level.resize(180, 80)
        self.btn_board_water_level.move(210, 260)
        self.btn_board_water_velocity = QPushButton('Water velocity', self)
        self.btn_board_water_velocity.clicked.connect(self.on_board_water_velocity_button_clicked)
        self.btn_board_water_velocity.resize(180, 80)
        self.btn_board_water_velocity.move(210, 380)

        self.lbl_boundary = QLabel('scenario selection', self)
        self.lbl_boundary.setStyleSheet("font-weight: bold; font-size: 24")
        self.lbl_boundary.move(10, 520)
        self.lbl_boundary.setFixedWidth(380)
        self.lbl_boundary.setAlignment(Qt.AlignCenter)

        """
        self.scenario4 = QPushButton('+3m SLR, 500 m/3s', self)
        self.scenario4.clicked.connect(self.on_scenario4_button_clicked)
        self.scenario4.resize(180, 80)
        self.scenario4.move(10, 580)
        """
        self.scenario3 = QPushButton('2100he', self)
        self.scenario3.clicked.connect(self.on_scenario3_button_clicked)
        self.scenario3.resize(180, 80)
        self.scenario3.move(210, 580)
        self.scenario2 = QPushButton('2018', self)
        self.scenario2.clicked.connect(self.on_scenario2_button_clicked)
        self.scenario2.resize(180, 80)
        self.scenario2.move(10, 700)
        self.scenario1 = QPushButton('2017', self)
        self.scenario1.clicked.connect(self.on_scenario1_button_clicked)
        self.scenario1.resize(180, 80)
        self.scenario1.move(210, 700)
        return

    def on_screen_salinity_button_clicked(self):
        self.viz_tracker.model_variable = "water_salinity"
        self.change_highlights()
        return

    def on_screen_water_level_button_clicked(self):
        self.viz_tracker.model_variable = "water_level"
        self.change_highlights()
        return

    def on_screen_water_velocity_button_clicked(self):
        self.viz_tracker.model_variable = "water_velocity"
        self.change_highlights()
        return

    def on_board_salinity_button_clicked(self):
        self.viz_tracker.game_variable = "water_salinity"
        self.change_highlights()
        return

    def on_board_water_level_button_clicked(self):
        self.viz_tracker.game_variable = "water_level"
        self.change_highlights()
        return

    def on_board_water_velocity_button_clicked(self):
        self.viz_tracker.game_variable = "water_velocity"
        self.change_highlights()
        return

    def on_scenario1_button_clicked(self):
        self.viz_tracker.scenario = "2017"
        self.change_highlights()
        return

    def on_scenario2_button_clicked(self):
        self.viz_tracker.scenario = "2018"
        self.change_highlights()
        return

    def on_scenario3_button_clicked(self):
        self.viz_tracker.scenario = "2100he"
        self.change_highlights()
        return

    """
    def on_scenario4_button_clicked(self):
        self.viz_tracker.scenario = "3mzss_0500m3s"
        self.change_highlights()
        return
    """

    def change_highlights(self):
        if self.screen_highlight != self.viz_tracker.model_variable:
            self.screen_highlight = self.viz_tracker.model_variable
            self.btn_screen_salinity.setStyleSheet("background-color:lightgray;")
            self.btn_screen_water_level.setStyleSheet("background-color:lightgray;")
            self.btn_screen_water_velocity.setStyleSheet("background-color:lightgray;")
            if self.screen_highlight == "water_salinity":
                self.btn_screen_salinity.setStyleSheet("background-color:red;")
            elif self.screen_highlight == "water_level":
                self.btn_screen_water_level.setStyleSheet("background-color:blue;")
            elif self.screen_highlight == "water_velocity":
                self.btn_screen_water_velocity.setStyleSheet("background-color:green;")
        if self.board_highlight != self.viz_tracker.game_variable:
            self.board_highlight = self.viz_tracker.game_variable
            self.btn_board_salinity.setStyleSheet("background-color:lightgray;")
            self.btn_board_water_level.setStyleSheet("background-color:lightgray;")
            self.btn_board_water_velocity.setStyleSheet("background-color:lightgray;")
            if self.board_highlight == "water_salinity":
                self.btn_board_salinity.setStyleSheet("background-color:red;")
            elif self.board_highlight == "water_level":
                self.btn_board_water_level.setStyleSheet("background-color:blue;")
            elif self.board_highlight == "water_velocity":
                self.btn_board_water_velocity.setStyleSheet("background-color:green;")
        if self.scenario_highlight != self.viz_tracker.scenario:
            self.scenario_highlight = self.viz_tracker.scenario
            self.scenario1.setStyleSheet("background-color:lightgray;")
            self.scenario2.setStyleSheet("background-color:lightgray;")
            self.scenario3.setStyleSheet("background-color:lightgray;")
            if self.scenario_highlight == "2017":
                self.scenario1.setStyleSheet("background-color:cyan;")
            elif self.scenario_highlight == "2018":
                self.scenario2.setStyleSheet("background-color:magenta;")
            elif self.scenario_highlight == "2100he":
                self.scenario3.setStyleSheet("background-color:yellow;")
        return

class VisualizationTracker():
    def __init__(self, starting_scenario, starting_variable, time_steps, starting_time,
                 salinity_range): #, water_level_range, water_velocity_range
        self._scenario = starting_scenario
        self._model_variable = starting_variable
        self._game_variable = starting_variable
        self._time_steps = time_steps
        self._time_index = starting_time
        self._salinity_norm = salinity_range
        #self._water_level_norm = water_level_range
        #self._water_velocity_norm = water_velocity_range
        return

    def get_time_index(self):
        t_idx = self.time_index % len(self.time_steps)
        return self.time_steps[t_idx]

    @property
    def scenario(self):
        return self._scenario

    @property
    def model_variable(self):
        return self._model_variable

    @property
    def game_variable(self):
        return self._game_variable

    @property
    def time_steps(self):
        return self._time_steps

    @property
    def time_index(self):
        return self._time_index

    @property
    def salinity_norm(self):
        return self._salinity_norm

    """
    @property
    def water_level_norm(self):
        return self._water_level_norm

    @property
    def water_velocity_norm(self):
        return self._water_velocity_norm
    """

    @scenario.setter
    def scenario(self, scenario):
        self._scenario = scenario
        return

    @model_variable.setter
    def model_variable(self, variable):
        self._model_variable = variable
        return

    @game_variable.setter
    def game_variable(self, variable):
        self._game_variable = variable
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

def main():
    model_scenarios, game_scenarios, world_bbox, game_bbox, salinity_range = load_scenarios() # , water_level_range, water_velocity_range
    qapp = QApplication.instance()
    if not qapp:
        qapp = QApplication(sys.argv)
    time_steps = list(sorted(set(model_scenarios["time"])))
    time_index = 0
    starting_scenario = "2017"
    starting_variable = "water_salinity"
    viz_tracker = VisualizationTracker(
        starting_scenario=starting_scenario, starting_variable=starting_variable,
        time_steps=time_steps, starting_time=time_index, salinity_range=salinity_range)
    # ,water_level_range = water_level_range, water_velocity_range = water_velocity_range
    colorbar_salinity, colorbar_water_level, colorbar_water_velocity = load_images()
    gui = ApplicationWindow(
        scenarios=model_scenarios, viz_tracker=viz_tracker, bbox=world_bbox,
        salinity_colorbar_image=colorbar_salinity, water_level_colorbar_image=colorbar_water_level,
        water_velocity_image=colorbar_water_velocity)
    side_window = GameVisualization(
        scenarios=game_scenarios, viz_tracker=viz_tracker, bbox=game_bbox)
    gui.show()
    side_window.show()
    gui.activateWindow()
    gui.raise_()
    qapp.exec()
    #locations = model_locations()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()