#import os
#import contextily as ctx
import matplotlib.collections
import matplotlib.style

matplotlib.style.use("fast")

from PyQt5.QtWidgets import (QWidget, QPushButton, QLabel, QDesktopWidget, QMainWindow, QHBoxLayout, QVBoxLayout,
                             QLineEdit)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure


class ApplicationWindow(QMainWindow):
    def __init__(self, game, viz_tracker, bbox, salinity_colorbar_image, salinity_category_image):
        super().__init__()
        self._main = QWidget()
        self.setWindowTitle('Delta Management Game demonstrator')
        self.setCentralWidget(self._main)
        self.setStyleSheet("background-color:white; font-weight: bold; font-size: 24")
        self.game = game
        #self.scenarios = game.model_output_gdf
        self.viz_tracker = viz_tracker
        self.selected_scenario = self.viz_tracker.scenario
        self.selected_model_variable = self.viz_tracker.model_variable
        self.selected_game_variable = self.viz_tracker.game_variable
        self.salinity_colorbar_image = salinity_colorbar_image
        self.salinity_category_image = salinity_category_image
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
        scenario_idx = self.game.model_output_gdf["scenario"] == self.selected_scenario
        self.running_scenario = self.game.model_output_gdf[scenario_idx]
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
            scenario_idx = self.game.model_output_gdf["scenario"] == self.selected_scenario
            self.running_scenario = self.game.model_output_gdf[scenario_idx]
        if self.selected_model_variable != self.viz_tracker.model_variable:
            self.selected_model_variable = self.viz_tracker.model_variable
            if self.selected_model_variable == "water_salinity":
                color_map = "RdBu_r"
                norm = self.viz_tracker.salinity_norm
            elif self.selected_model_variable == "water_level":
                color_map = "viridis_r"
                norm = self.viz_tracker.water_level_norm
            elif self.selected_model_variable == "salinity_category":
                color_map = "RdYlBu_r"
                norm = self.viz_tracker.salinity_category_norm
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
            if self.selected_model_variable == "salinity_category":
                self.colorbar_model_label.setPixmap(self.salinity_category_image)
        elif to_update == "game":
            if self.selected_game_variable == "water_salinity":
                self.colorbar_game_label.setPixmap(self.salinity_colorbar_image)
            if self.selected_game_variable == "salinity_category":
                self.colorbar_game_label.setPixmap(self.salinity_category_image)
        return


class GameVisualization(QWidget):
    def __init__(self, game, viz_tracker, bbox):
        super().__init__()
        self.setStyleSheet("background-color:white;")
        #self.setFixedSize(1280, 720)
        #self.setWindowFlags(Qt.FramelessWindowHint)
        #self.scenarios = scenarios
        self.game = game
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
        scenario_idx = self.game.game_output_gdf["scenario"] == self.selected_scenario
        self.running_scenario = self.game.game_output_gdf[scenario_idx]
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
            scenario_idx = self.game.game_output_gdf["scenario"] == self.selected_scenario
            self.running_scenario = self.game.game_output_gdf[scenario_idx]
        if self.selected_variable != self.viz_tracker.game_variable:
            print(9)
            self.selected_variable = self.viz_tracker.game_variable
            if self.selected_variable == "water_salinity":
                color_map = "RdBu_r"
                norm = self.viz_tracker.salinity_norm
            elif self.selected_variable == "water_level":
                color_map = "viridis_r"
                norm = self.viz_tracker.water_level_norm
            elif self.selected_variable == "salinity_category":
                print(10)
                color_map = "RdYlBu_r"
                norm = self.viz_tracker.salinity_category_norm
                print(11)
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
        self.lbl_screen_variable.move(10, 20)
        self.lbl_screen_variable.setFixedWidth(180)
        self.lbl_screen_variable.setAlignment(Qt.AlignCenter)

        self.btn_screen_salinity = QPushButton('Salinity concentration', self)
        self.btn_screen_salinity.clicked.connect(self.on_screen_salinity_button_clicked)
        self.btn_screen_salinity.resize(180, 60)
        self.btn_screen_salinity.move(10, 80)
        self.btn_screen_salinity_category = QPushButton('Salinity (categorized)', self)
        self.btn_screen_salinity_category.clicked.connect(self.on_screen_salinity_category_button_clicked)
        self.btn_screen_salinity_category.resize(180, 60)
        self.btn_screen_salinity_category.move(10, 160)

        """
        self.btn_screen_water_velocity = QPushButton('Water velocity', self)
        self.btn_screen_water_velocity.clicked.connect(self.on_screen_water_velocity_button_clicked)
        self.btn_screen_water_velocity.resize(180, 80)
        self.btn_screen_water_velocity.move(10, 380)
        """

        self.lbl_board_variable = QLabel('Board variable selection', self)
        self.lbl_board_variable.setStyleSheet("font-weight: bold; font-size: 24")
        self.lbl_board_variable.move(210, 20)
        self.lbl_board_variable.setFixedWidth(180)
        self.lbl_board_variable.setAlignment(Qt.AlignCenter)

        self.btn_board_salinity = QPushButton('Salinity concentration', self)
        self.btn_board_salinity.clicked.connect(self.on_board_salinity_button_clicked)
        self.btn_board_salinity.resize(180, 60)
        self.btn_board_salinity.move(210, 80)
        self.btn_board_salinity_category = QPushButton('Salinity (categorized)', self)
        self.btn_board_salinity_category.clicked.connect(self.on_board_salinity_category_button_clicked)
        self.btn_board_salinity_category.resize(180, 60)
        self.btn_board_salinity_category.move(210, 160)

        self.textbox = QLineEdit(self)
        self.textbox.resize(380, 60)
        self.textbox.move(10, 240)

        self.btn_update = QPushButton('Change delta', self)
        self.btn_update.clicked.connect(self.on_update_button_clicked)
        self.btn_update.resize(380, 60)
        self.btn_update.move(10, 320)
        self.btn_update.setStyleSheet("background-color:lightgray;")

        self.btn_run_model = QPushButton('Run model', self)
        self.btn_run_model.clicked.connect(self.on_run_model_button_clicked)
        self.btn_run_model.resize(380, 60)
        self.btn_run_model.move(10, 400)
        self.btn_run_model.setStyleSheet("background-color:lightgray;")
        """
        self.btn_board_water_velocity = QPushButton('Water velocity', self)
        self.btn_board_water_velocity.clicked.connect(self.on_board_water_velocity_button_clicked)
        self.btn_board_water_velocity.resize(180, 80)
        self.btn_board_water_velocity.move(210, 380)
        """

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
        self.scenario4 = QPushButton('2100he (+1m SLR) +\n widened NWW', self)
        self.scenario4.clicked.connect(self.on_scenario4_button_clicked)
        self.scenario4.resize(180, 80)
        self.scenario4.move(210, 700)
        self.scenario3 = QPushButton('2100le (+1m SLR) +\n deepened NWW & Nieuwe Maas', self)
        self.scenario3.clicked.connect(self.on_scenario3_button_clicked)
        self.scenario3.resize(180, 80)
        self.scenario3.move(10, 700)
        self.scenario2 = QPushButton('2018 +\n partly deepened NWW', self)
        self.scenario2.clicked.connect(self.on_scenario2_button_clicked)
        self.scenario2.resize(180, 80)
        self.scenario2.move(210, 580)
        self.scenario1 = QPushButton('2017', self)
        self.scenario1.clicked.connect(self.on_scenario1_button_clicked)
        self.scenario1.resize(180, 80)
        self.scenario1.move(10, 580)
        return

    def on_update_button_clicked(self):
        textboxValue = self.textbox.text()
        self.gui.game.get_changes(textboxValue)
        return

    def on_run_model_button_clicked(self):
        self.gui.game.update()
        return

    def on_screen_salinity_button_clicked(self):
        self.viz_tracker.model_variable = "water_salinity"
        self.change_highlights()
        return

    def on_screen_salinity_category_button_clicked(self):
        self.viz_tracker.model_variable = "salinity_category"
        self.change_highlights()
        return

    def on_board_salinity_button_clicked(self):
        self.viz_tracker.game_variable = "water_salinity"
        self.change_highlights()
        return

    def on_board_salinity_category_button_clicked(self):
        print(1)
        self.viz_tracker.game_variable = "salinity_category"
        print(2)
        self.change_highlights()
        print(3)
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
        self.viz_tracker.scenario = "2100le"
        self.change_highlights()
        return

    def on_scenario4_button_clicked(self):
        self.viz_tracker.scenario = "2100he"
        self.change_highlights()
        return


    def change_highlights(self):
        if self.screen_highlight != self.viz_tracker.model_variable:
            self.screen_highlight = self.viz_tracker.model_variable
            self.btn_screen_salinity.setStyleSheet("background-color:lightgray;")
            self.btn_screen_salinity_category.setStyleSheet("background-color:lightgray;")
            if self.screen_highlight == "water_salinity":
                self.btn_screen_salinity.setStyleSheet("background-color:red;")
            elif self.screen_highlight == "salinity_category":
                self.btn_screen_salinity_category.setStyleSheet("background-color:blue;")
        if self.board_highlight != self.viz_tracker.game_variable:
            print(4)
            self.board_highlight = self.viz_tracker.game_variable
            self.btn_board_salinity.setStyleSheet("background-color:lightgray;")
            self.btn_board_salinity_category.setStyleSheet("background-color:lightgray;")
            if self.board_highlight == "water_salinity":
                self.btn_board_salinity.setStyleSheet("background-color:red;")
            elif self.board_highlight == "salinity_category":
                print(5)
                self.btn_board_salinity_category.setStyleSheet("background-color:blue;")
                print(6)
        if self.scenario_highlight != self.viz_tracker.scenario:
            self.scenario_highlight = self.viz_tracker.scenario
            self.scenario1.setStyleSheet("background-color:lightgray;")
            self.scenario2.setStyleSheet("background-color:lightgray;")
            self.scenario3.setStyleSheet("background-color:lightgray;")
            self.scenario4.setStyleSheet("background-color:lightgray;")
            if self.scenario_highlight == "2017":
                self.scenario1.setStyleSheet("background-color:cyan;")
            elif self.scenario_highlight == "2018":
                self.scenario2.setStyleSheet("background-color:magenta;")
            elif self.scenario_highlight == "2100le":
                self.scenario3.setStyleSheet("background-color:yellow;")
            elif self.scenario_highlight == "2100he":
                self.scenario4.setStyleSheet("background-color:green;")
        return

class VisualizationTracker():
    def __init__(self, starting_scenario, starting_variable, time_steps, starting_time,
                 salinity_range, salinity_category): #, water_level_range, water_velocity_range
        self._scenario = starting_scenario
        self._model_variable = starting_variable
        self._game_variable = starting_variable
        self._time_steps = time_steps
        self._time_index = starting_time
        self._salinity_norm = salinity_range
        self._salinity_category_norm = salinity_category
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

    @property
    def salinity_category_norm(self):
        return self._salinity_category_norm

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
        print(7)
        self._game_variable = variable
        print(8)
        return

    @time_steps.setter
    def time_steps(self, time_steps):
        self._time_steps = time_steps
        return

    @time_index.setter
    def time_index(self, time_index):
        self._time_index += time_index
        return