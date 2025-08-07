#import os
#import contextily as ctx
import matplotlib.collections
import matplotlib.style

matplotlib.style.use("fast")

from PyQt5.QtWidgets import (QWidget, QPushButton, QLabel, QDesktopWidget, QMainWindow, QHBoxLayout, QVBoxLayout,
                             QLineEdit, QTableView)
from PyQt5.QtCore import Qt, QAbstractTableModel
from PyQt5.QtGui import QFont

from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from pandas import to_datetime
from functools import partial


class pandasModel(QAbstractTableModel):
    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None

class ApplicationWindow(QMainWindow):
    def __init__(self, game, viz_tracker, bbox, salinity_colorbar_image, salinity_category_image):
        super().__init__()
        self._main = QWidget()
        self.setWindowTitle('Delta Management Game demonstrator')
        self.setCentralWidget(self._main)
        self.setStyleSheet("background-color:white; font-weight: bold; font-size: 24")
        self.game = game
        self.viz_tracker = viz_tracker
        self.selected_turn = self.viz_tracker.turn
        self.selected_run = self.viz_tracker.run
        self.scenario = self.viz_tracker.scenario
        self.selected_variable = self.viz_tracker.viz_variable
        #self.selected_game_variable = self.viz_tracker.game_variable
        self.salinity_colorbar_image = salinity_colorbar_image
        self.salinity_category_image = salinity_category_image
        #self.basemap_image = basemap_image
        self.window = QVBoxLayout(self._main)

        self.score_widget = ScoreWidget(gui=self, viz_tracker=viz_tracker)
        self.window.addWidget(self.score_widget)

        self.layout = QHBoxLayout(self._main)
        self.model_canvas = FigureCanvas(Figure()) #figsize=(5, 5)
        # Ideally one would use self.addToolBar here, but it is slightly
        # incompatible between PyQt6 and other bindings, so we just add the
        # toolbar as a plain widget instead.
        self.figure_layout = QVBoxLayout(self._main)
        #self.figure_layout.addWidget(NavigationToolbar(self.model_canvas, self))
        self.figure_layout.addWidget(self.model_canvas, stretch=1)

        self.colorbar_label = QLabel(self._main)
        self.colorbar_label.setPixmap(self.salinity_colorbar_image)
        self.colorbar_label.resize(self.salinity_colorbar_image.width(), self.salinity_colorbar_image.height())
        self.figure_layout.addWidget(self.colorbar_label, alignment=Qt.AlignCenter)

        #forcing_conditions_df = self.game.forcing_conditions
        #forcing_conditions_df = forcing_conditions_df[forcing_conditions_df["turn"] == self.viz_tracker.turn]
        #forcing_conditions_df = forcing_conditions_df.drop(columns="turn")
        #forcing_conditions = pandasModel(forcing_conditions_df)
        self.forcing_table = QTableView()
        #self.forcing_table.setModel(forcing_conditions)
        #self.forcing_table.resize(600, 300)
        self.figure_layout.addWidget(self.forcing_table) #, alignment=Qt.AlignCenter)
        self.figure_widget = QWidget()
        self.figure_widget.setLayout(self.figure_layout)
        self.figure_widget.setFixedWidth(800)

        #self.layout.addLayout(self.figure_layout)
        self.layout.addWidget(self.figure_widget)

        #self.colorbar_title_layout = QHBoxLayout(self._main)
        #self.colorbar_model_title_label = QLabel(self._main)
        #self.colorbar_model_title_label.setText("Screen colorbar")
        #self.colorbar_model_title_label.setStyleSheet("font-weight: bold; font-size: 36")
        #self.colorbar_title_layout.addWidget(self.colorbar_model_title_label, alignment=Qt.AlignCenter)
        #self.colorbar_game_title_label = QLabel(self._main)
        #self.colorbar_game_title_label.setText("Board colorbar")
        #self.colorbar_game_title_label.setStyleSheet("font-weight: bold; font-size: 36")
        #self.colorbar_title_layout.addWidget(self.colorbar_game_title_label, alignment=Qt.AlignCenter)
        #self.figure_layout.addLayout(self.colorbar_title_layout)

        #self.colorbar_layout = QHBoxLayout(self._main)
        #self.colorbar_model_label = QLabel(self._main)
        #self.colorbar_model_label.setPixmap(self.salinity_colorbar_image)
        #self.colorbar_model_label.resize(self.salinity_colorbar_image.width(), self.salinity_colorbar_image.height())
        #self.colorbar_layout.addWidget(self.colorbar_model_label, alignment=Qt.AlignCenter)
        #self.colorbar_game_label = QLabel(self._main)
        #self.colorbar_game_label.setPixmap(self.salinity_colorbar_image)
        #self.colorbar_game_label.resize(self.salinity_colorbar_image.width(), self.salinity_colorbar_image.height())
        #self.colorbar_layout.addWidget(self.colorbar_game_label, alignment=Qt.AlignCenter)
        #self.figure_layout.addLayout(self.colorbar_layout)
        #self.layout.addLayout(self.figure_layout)

        self.inlet_layout = QVBoxLayout(self._main)
        # Ideally one would use self.addToolBar here, but it is slightly
        # incompatible between PyQt6 and other bindings, so we just add the
        # toolbar as a plain widget instead.

        # self.figure_layout.addWidget(NavigationToolbar(self.model_canvas, self))
        self.inlet_canvas = FigureCanvas(Figure())
        self.inlet_layout.addWidget(self.inlet_canvas, stretch=1)
        self.inlet_widget = InletWidget(gui=self, viz_tracker=viz_tracker)
        self.inlet_layout.addWidget(self.inlet_widget)
        self.inlet_plots = FigureCanvas(Figure())  # figsize=(5, 5)
        self.inlet_layout.addWidget(self.inlet_plots, stretch=1)
        self.layout.addLayout(self.inlet_layout)

        self.control_widget = ControlWidget(gui=self, viz_tracker=viz_tracker)
        self.layout.addWidget(self.control_widget)
        self.window.addLayout(self.layout)
        self.add_plot_model(bbox)
        self.show_forcing_conditions()
        self.plot_inlet_indicators()
        self.plot_salinity_inlets()
        return

    def add_plot_model(self, bbox):
        self.ax = self.model_canvas.figure.subplots()
        #self.ax.axis(bbox)
        #ctx.add_basemap(self.ax, alpha=0.5, source=ctx.providers.CartoDB.PositronNoLabels)
        #ctx.add_basemap(self.ax, source=ctx.providers.Esri.WorldGrayCanvas, zoom=12)
        #ctx.add_basemap(self.ax, alpha=0.5, source=ctx.providers.OpenStreetMap.Mapnik)
        self.ax.set_axis_off()
        self.ax.set_position([0.1, 0.1, 0.8, 0.8])
        turn_idx = (self.game.model_drought_output_gdf["turn"] == self.selected_turn) & (
                self.game.model_drought_output_gdf["run"] == self.selected_run)
        self.running_simulation = self.game.model_drought_output_gdf[turn_idx]
        t_idx = self.viz_tracker.time_index
        t = self.running_simulation.iloc[t_idx]["time"]
        idx = self.running_simulation["time"] == t
        self.plot_data = self.running_simulation[idx]
        self.plot_data.plot(column=self.selected_variable, ax=self.ax, cmap="RdBu_r", markersize=150.0)
        #self.plot_data.plot(column=self.selected_variable, ax=self.ax, cmap="coolwarm", markersize=150.0)

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
        if self.selected_turn != self.viz_tracker.turn or self.selected_run != self.viz_tracker.run:
            self.selected_turn = self.viz_tracker.turn
            self.selected_run = self.viz_tracker.run
            turn_idx = (self.game.model_drought_output_gdf["turn"] == self.selected_turn) & (
                    self.game.model_drought_output_gdf["run"] == self.selected_run)
            self.running_simulation = self.game.model_drought_output_gdf[turn_idx]
            self.viz_tracker.update_scenario()
            self.scenario = self.viz_tracker.scenario
        if self.selected_variable != self.viz_tracker.viz_variable:
            self.selected_variable = self.viz_tracker.viz_variable
            if self.selected_variable == "water_salinity":
                color_map = "RdBu_r"
                #color_map = "coolwarm"
                norm = self.viz_tracker.salinity_norm
            elif self.selected_variable == "water_level":
                color_map = "viridis_r"
                norm = self.viz_tracker.water_level_norm
            elif self.selected_variable == "salinity_category":
                color_map = "RdYlBu_r"
                norm = self.viz_tracker.salinity_category_norm
            elif self.selected_variable == "water_velocity":
                color_map = "Spectral_r"
                norm = self.viz_tracker.water_velocity_norm
            elif self.selected_variable == "water_depth":
                color_map = "Blues_r"
                norm = self.viz_tracker.water_depth_norm
            self.pc.set_cmap(color_map)
            self.pc.set_norm(norm)
            self.update_colorbars()
        t = self.viz_tracker.get_time_index()
        idx = self.running_simulation["time"] == t
        self.plot_data = self.running_simulation[idx]
        self.pc.set_array(self.plot_data[self.selected_variable])
        title_string = "turn: " + str(self.selected_turn) + " - run: " + str(self.selected_run) + " - " + self.scenario + " - " + f"timestep: {t}"
        self.ax.set_title(title_string[:-8])
        self.model_canvas.draw()
        self.viz_tracker.time_index = 1
        return

    def update_colorbars(self):
        if self.selected_variable == "water_salinity":
            self.colorbar_label.setPixmap(self.salinity_colorbar_image)
        if self.selected_variable == "salinity_category":
            self.colorbar_label.setPixmap(self.salinity_category_image)
        return

    def update_colorbars_old(self, to_update="model"):
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

    def plot_inlet_indicators(self):
        self.inlet_canvas.figure.clf()
        ax = self.inlet_canvas.figure.subplots()
        ax.set_axis_off()

        model_output = self.game.model_drought_output_gdf.copy()
        #model_output = model_output[model_output["turn"] == self.viz_tracker.turn]
        model_output = model_output[
            (model_output["turn"] == self.viz_tracker.turn) & (model_output["run"] == self.viz_tracker.run)]
        model_output = model_output[model_output["time"] == model_output.iloc[0]["time"]]
        if model_output.empty:
            print("there seems to be no output data yet")
        else:
            model_output.plot(ax=ax, color="deepskyblue", markersize=50)

        inlet_data = self.game.inlet_salinity_tracker.copy()
        #inlet_data = inlet_data[inlet_data["turn"] == self.viz_tracker.turn]
        inlet_data = inlet_data[
            (inlet_data["turn"] == self.viz_tracker.turn) & (inlet_data["run"] == self.viz_tracker.run)]
        if inlet_data.empty:
            print("there seems to be no inlet data yet")
        else:
            inlet_data = inlet_data[inlet_data["time"] == inlet_data.iloc[0]["time"]]
            inlet_data = inlet_data.reset_index()
            cmap = LinearSegmentedColormap.from_list("", ["green", "orange", "red"])
            inlet_data.plot("score_indicator", ax=ax, cmap=cmap, vmin=1, vmax=3, markersize=300)  # , legend=True)
            for x, y, label in zip(inlet_data["geometry"].x, inlet_data["geometry"].y, inlet_data["name"]):
                ax.annotate(label, xy=(x, y), xytext=(0, 12), ha='center', textcoords="offset points", size=10)
        self.inlet_canvas.draw()
        return

    def plot_salinity_inlets(self, combined_plot=True):
        to_plot = self.viz_tracker.inlet_to_plot
        self.inlet_plots.figure.clf()
        ax = self.inlet_plots.figure.subplots()
        inlet_data = self.game.inlet_salinity_tracker.copy()
        if combined_plot:
            inlet_data = inlet_data[inlet_data["turn"] == self.viz_tracker.turn]
            run_idx = inlet_data['run'].unique()
        else:
            inlet_data = inlet_data[
                (inlet_data["turn"] == self.viz_tracker.turn) & (inlet_data["run"] == self.viz_tracker.run)]
        if inlet_data.empty:
            print("there seems to be no inlet data yet")
        else:
            inlet_data = inlet_data.reset_index()
            inlet_data = inlet_data[inlet_data['name'] == to_plot]
            ax.set_facecolor('lightgray')

            if combined_plot:
                markers = ['o', 's', 'h', 'v', 'D']
                color = ['blue', 'darkgreen', 'orange', 'gold', 'purple']

                for idx in run_idx:
                    run_data = inlet_data[inlet_data['run'] == idx]
                    time_steps = to_datetime(run_data['time']).dt.strftime('%Y-%m-%d')
                    salinity_values = run_data['water_salinity'].values

                    ax.plot(time_steps, salinity_values, marker=markers[idx-1], linestyle='-', color=color[idx-1], label='Salinity (mg/l) run %d' % idx)
            else:
                time_steps = to_datetime(inlet_data['time']).dt.strftime('%Y-%m-%d')
                salinity_values = inlet_data['water_salinity'].values
                ax.plot(time_steps, salinity_values, marker='o', linestyle='-', color='blue',
                        label='Salinity (mg/l) run %d' % self.viz_tracker.turn)

            cl_threshold_normal = inlet_data.iloc[0]['CL_threshold_during_regular_operation_(mg/l)']
            cl_threshold_drought = inlet_data.iloc[0]['CL_threshold_during_drought_(mg/l)']

            ax.axhline(y=cl_threshold_normal, color='green', linestyle=(0, (3, 1.5, 1, 1.5)),
                        label='CL Threshold Normal')  # Custom dash-dot pattern: (dash length, gap length, dot length, gap length)

            ax.axhline(y=cl_threshold_drought, color='red', linestyle=(0, (5, 1.5, 1, 1.5)),
                        label='CL Threshold Drought')

            ax.set_title(f"Salinity at {to_plot}", fontsize=16)
            ax.set_xlabel("Time (day)", fontsize=14)
            ax.set_ylabel("Salinity (mg/l)", fontsize=14)

            ax.set_ylim(ymin=0)

            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
            ax.set_yticks(ax.get_yticks())
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)

            ax.spines['bottom'].set_color('black')
            ax.spines['left'].set_color('black')
            ax.spines['bottom'].set_linewidth(1)
            ax.spines['left'].set_linewidth(1)

            ax.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)

            text_to_plot = ("Number of days above threshold (normal): " + str(
                inlet_data.iloc[0]['Num_days_exceedance_normal']) + "\n" +
                            "Number of consecutive days above threshold (normal): " + str(
                        inlet_data.iloc[0]['Num_days_consecutive_normal']) + "\n" +
                            "Number of days above threshold (drought): " + str(
                        inlet_data.iloc[0]['Num_days_exceedance_drought']) + "\n" +
                            "Number of consecutive days above threshold (drought): " + str(
                        inlet_data.iloc[0]['Num_days_consecutive_drought']))

            #ax.text(1.0, -0.4, text_to_plot, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes,
            #        fontsize=16)
            ax.text(0.96, 0.04, text_to_plot, horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes,
                    fontsize=11)

            ax.legend(loc='best', fontsize=10)
        self.inlet_plots.draw()
        return

    def show_forcing_conditions(self):
        forcing_conditions_df = self.game.forcing_conditions.copy()
        forcing_conditions_df = forcing_conditions_df[
            (forcing_conditions_df["turn"] == self.viz_tracker.turn) &
            (forcing_conditions_df["run"] == self.viz_tracker.run) &
            (forcing_conditions_df["type"] == "drought")] # in case simulation types become selectable, add self.game.simulations to viz_tracker and select accordingly
        forcing_conditions_df = forcing_conditions_df.drop(columns=["turn", "run", "type"])
        forcing_conditions = pandasModel(forcing_conditions_df)
        self.forcing_table.setModel(forcing_conditions)
        return

    def show_turn_button(self, turn, run):
        self.control_widget.turn_buttons[turn][run-1].setEnabled(True)
        if turn == 2:
            self.control_widget.on_turn2_button_clicked(run)
        elif turn == 3:
            self.control_widget.on_turn3_button_clicked(run)
        """
        if turn == 2:
            self.control_widget.btn_turn2.setEnabled(True)
            self.control_widget.on_turn2_button_clicked()
        elif turn == 3:
            self.control_widget.btn_turn3.setEnabled(True)
            self.control_widget.on_turn3_button_clicked()
        """
        """
        elif turn == 4:
            self.control_widget.btn_turn4.setEnabled(True)
            self.control_widget.on_turn4_button_clicked()
        """
        return

    def change_inlet_plot(self):
        return


class GameVisualization(QWidget):
    def __init__(self, game, viz_tracker, bbox):
        super().__init__()
        self.setStyleSheet("background-color:white;")
        self.game = game
        self.viz_tracker=viz_tracker
        self.selected_turn = self.viz_tracker.turn
        self.selected_run = self.viz_tracker.run
        self.selected_variable = self.viz_tracker.viz_variable
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
        turn_idx = (self.game.game_drought_output_gdf["turn"] == self.selected_turn) & (
                    self.game.game_drought_output_gdf["run"] == self.selected_run)
        self.running_simulation = self.game.game_drought_output_gdf[turn_idx]
        t_idx = self.viz_tracker.time_index
        t = self.running_simulation.iloc[t_idx]["time"]
        idx = self.running_simulation["time"] == t
        self.plot_data = self.running_simulation[idx]
        self.plot_data.plot(column=self.selected_variable, ax=self.ax, cmap="RdBu_r", aspect=1, markersize=200.0)
        #self.plot_data.plot(column=self.selected_variable, ax=self.ax, cmap="coolwarm", aspect=1, markersize=200.0)

        pcs = [child for child in self.ax.get_children() if isinstance(child, matplotlib.collections.PathCollection)]
        assert len(pcs) == 1, "expected 1 pathcollection after plotting"
        self.pc = pcs[0]
        self.pc.set_norm(self.viz_tracker.salinity_norm)

        self.game_timer = self.game_canvas.new_timer(40)
        self.game_timer.add_callback(self.update_plot_model)
        self.game_timer.start()
        return

    def update_plot_model(self):
        if self.selected_turn != self.viz_tracker.turn or self.selected_run != self.viz_tracker.run:
            self.selected_turn = self.viz_tracker.turn
            self.selected_run = self.viz_tracker.run
            turn_idx = (self.game.game_drought_output_gdf["turn"] == self.selected_turn) & (
                        self.game.game_drought_output_gdf["run"] == self.selected_run)
            self.running_simulation = self.game.game_drought_output_gdf[turn_idx]
        if self.selected_variable != self.viz_tracker.viz_variable:
            self.selected_variable = self.viz_tracker.viz_variable
            if self.selected_variable == "water_salinity":
                color_map = "RdBu_r"
                #color_map = "coolwarm"
                norm = self.viz_tracker.salinity_norm
            elif self.selected_variable == "water_level":
                color_map = "viridis_r"
                norm = self.viz_tracker.water_level_norm
            elif self.selected_variable == "salinity_category":
                color_map = "RdYlBu_r"
                norm = self.viz_tracker.salinity_category_norm
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
        idx = self.running_simulation["time"] == t
        self.plot_data = self.running_simulation[idx]
        self.pc.set_array(self.plot_data[self.selected_variable])
        #self.ax.set_title(f"timestep: {t} - scenario {self.selected_scenario}")
        self.game_canvas.draw()
        return

class InletWidget(QWidget):
    def __init__(self, gui, viz_tracker):
        super().__init__()
        #self.setStyleSheet("background-color:grey;")
        self.gui = gui
        self.viz_tracker = viz_tracker
        self.setFixedSize(800, 80)
        #self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.inlet_highlight = None
        self.initUI()
        self.change_highlights()
        self.show()  # app.exec_()

    def initUI(self):
        start_x = 15
        width = 104
        height = 60
        spacing = 7
        spot_x = start_x
        spot_y = 10
        font_size = 9
        self.btn_inlet1 = QPushButton('Inlaatsluis\nBernisse', self)
        self.btn_inlet1.setFont(QFont('Arial', font_size))
        self.btn_inlet1.clicked.connect(lambda: self.on_inlet_button_clicked(0))
        self.btn_inlet1.resize(width, height)
        self.btn_inlet1.move(spot_x, spot_y)
        spot_x += width + spacing
        self.btn_inlet2 = QPushButton('Parksluizen', self)
        self.btn_inlet2.setFont(QFont('Arial', font_size))
        self.btn_inlet2.clicked.connect(lambda: self.on_inlet_button_clicked(1))
        self.btn_inlet2.resize(width, height)
        self.btn_inlet2.move(spot_x, spot_y)
        spot_x += width + spacing
        self.btn_inlet3 = QPushButton('Krimpen aan\nden IJssel', self)
        self.btn_inlet3.setFont(QFont('Arial', font_size))
        self.btn_inlet3.clicked.connect(lambda: self.on_inlet_button_clicked(2))
        self.btn_inlet3.resize(width + 10, height)
        self.btn_inlet3.move(spot_x, spot_y)
        spot_x += width + spacing + 10
        self.btn_inlet4 = QPushButton('Kinderdijk', self)
        self.btn_inlet4.setFont(QFont('Arial', font_size))
        self.btn_inlet4.clicked.connect(lambda: self.on_inlet_button_clicked(3))
        self.btn_inlet4.resize(width - 10, height)
        self.btn_inlet4.move(spot_x, spot_y)
        spot_x += width + spacing - 10
        self.btn_inlet5 = QPushButton('Krimpener-\nwaard', self)
        self.btn_inlet5.setFont(QFont('Arial', font_size))
        self.btn_inlet5.clicked.connect(lambda: self.on_inlet_button_clicked(4))
        self.btn_inlet5.resize(width, height)
        self.btn_inlet5.move(spot_x, spot_y)
        spot_x += width + spacing
        self.btn_inlet6 = QPushButton('Snelle\nSluis', self)
        self.btn_inlet6.setFont(QFont('Arial', font_size))
        self.btn_inlet6.clicked.connect(lambda: self.on_inlet_button_clicked(5))
        self.btn_inlet6.resize(width - 10, height)
        self.btn_inlet6.move(spot_x, spot_y)
        spot_x += width + spacing - 10
        self.btn_inlet6 = QPushButton('Boezemgemaal\nGouda', self)
        self.btn_inlet6.setFont(QFont('Arial', font_size))
        self.btn_inlet6.clicked.connect(lambda: self.on_inlet_button_clicked(6))
        self.btn_inlet6.resize(width + 20, height)
        self.btn_inlet6.move(spot_x, spot_y)
        return

    def on_inlet_button_clicked(self, btn_number):
        inlets = ['Inlaatsluis Bernisse', 'Parksluizen', 'Krimpen aan den IJssel', 'Kinderdijk', 'Krimpenerwaard',
                  'Snelle Sluis', 'Boezemgemaal Gouda']
        inlet_to_plot = inlets[btn_number]
        self.viz_tracker.inlet_to_plot = inlet_to_plot
        self.gui.plot_salinity_inlets()
        return

    def change_highlights(self):
        return

class ScoreWidget(QWidget):
    def __init__(self, gui, viz_tracker):
        super().__init__()
        #self.setStyleSheet("background-color:grey;")
        self.gui = gui
        self.viz_tracker = viz_tracker
        self.setFixedSize(1920, 50)
        #self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.initUI()
        self.update_text()
        self.change_highlights()
        self.show()  # app.exec_()

    def initUI(self):
        start_x = 30
        width = 300
        height = 60
        spacing = 50
        spot_x = start_x
        spot_y = 5
        font_size = 12
        self.lbl_green_inlets = QLabel('Number of green inlets', self)
        self.lbl_green_inlets.setFont(QFont('Arial', font_size))
        self.lbl_green_inlets.move(spot_x, spot_y)
        self.lbl_green_inlets.setFixedWidth(width)
        self.lbl_green_inlets.setAlignment(Qt.AlignLeft)
        spot_x += width + spacing

        self.lbl_orange_inlets = QLabel('Number of orange inlets', self)
        self.lbl_orange_inlets.setFont(QFont('Arial', font_size))
        self.lbl_orange_inlets.setStyleSheet("font-weight: bold; font-size: 24")
        self.lbl_orange_inlets.move(spot_x, spot_y)
        self.lbl_orange_inlets.setFixedWidth(width)
        self.lbl_orange_inlets.setAlignment(Qt.AlignLeft)
        spot_x += width + spacing

        self.lbl_red_inlets = QLabel('Number of red inlets', self)
        self.lbl_red_inlets.setFont(QFont('Arial', font_size))
        self.lbl_red_inlets.setStyleSheet("font-weight: bold; font-size: 24")
        self.lbl_red_inlets.move(spot_x, spot_y)
        self.lbl_red_inlets.setFixedWidth(width)
        self.lbl_red_inlets.setAlignment(Qt.AlignLeft)
        spot_x += width + spacing

        self.lbl_score = QLabel('Score', self)
        self.lbl_score.setFont(QFont('Arial', font_size))
        self.lbl_score.setStyleSheet("font-weight: bold; font-size: 24")
        self.lbl_score.move(spot_x, spot_y)
        self.lbl_score.setFixedWidth(width)
        self.lbl_score.setAlignment(Qt.AlignLeft)
        return

    def update_text(self):
        inlet_data = self.gui.game.inlet_salinity_tracker.copy()
        inlet_data = inlet_data[(inlet_data["turn"] == self.viz_tracker.turn) & (inlet_data["run"] == self.viz_tracker.run)]
        #df1 = df[(df.a != -1) & (df.b != -1)]
        if inlet_data.empty:
            print("there seems to be no inlet data yet")
        else:
            inlet_data = inlet_data[inlet_data["time"] == inlet_data.iloc[0]["time"]]
            inlet_data = inlet_data.reset_index()
            number_green = len(inlet_data[inlet_data['score_indicator'] == 1])
            #number_yellow = len(inlet_data[inlet_data['score_indicator'] == 2])
            number_orange = len(inlet_data[inlet_data['score_indicator'] == 2])
            number_red = len(inlet_data[inlet_data['score_indicator'] == 3])
            #score = ((number_green * 1 + number_yellow * 0.75 + number_orange * 0.5 + number_red * 0.25) / 6) * 100
            score = ((number_green * 1 + number_orange * 0.5) / len(inlet_data)) * 100
            self.lbl_green_inlets.setText('Number of green inlets: %d' % number_green)
            #self.lbl_yellow_inlets.setText('Number of yellow inlets: %d' % number_yellow)
            self.lbl_orange_inlets.setText('Number of orange inlets: %d' % number_orange)
            self.lbl_red_inlets.setText('Number of red inlets: %d' % number_red)
            self.lbl_score.setText('Score: %d percent' % score)
        return

    def change_highlights(self):
        return

class ControlWidget(QWidget):
    def __init__(self, gui, viz_tracker):
        super().__init__()
        #self.setStyleSheet("background-color:grey;")
        self.gui = gui
        self.viz_tracker = viz_tracker
        self.setFixedSize(200, 900)
        #self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self.screen_highlight = None
        self.board_highlight = None
        self.turn_highlight = None
        self.run_highlight = None
        self.initUI()
        self.change_highlights()
        self.show()  # app.exec_()

    def initUI(self):
        label_spacing = 30
        button_spacing = 15
        button_height = 60
        y_location = 0

        self.lbl_screen_variable = QLabel('Visualization', self)
        self.lbl_screen_variable.setFont(QFont('Arial', 12))
        self.lbl_screen_variable.setStyleSheet("font-weight: bold; font-size: 24")
        self.lbl_screen_variable.move(10, y_location)
        self.lbl_screen_variable.setFixedWidth(180)
        self.lbl_screen_variable.setAlignment(Qt.AlignCenter)

        y_location += label_spacing

        self.btn_screen_salinity = QPushButton('Salinity\nconcentration', self)
        self.btn_screen_salinity.setFont(QFont('Arial', 10))
        self.btn_screen_salinity.clicked.connect(self.on_salinity_button_clicked)
        self.btn_screen_salinity.resize(180, 60)
        self.btn_screen_salinity.move(10, y_location)

        y_location += (button_height + button_spacing)

        self.btn_screen_salinity_category = QPushButton('Salinity\n(categorized)', self)
        self.btn_screen_salinity_category.setFont(QFont('Arial', 10))
        self.btn_screen_salinity_category.clicked.connect(self.on_salinity_category_button_clicked)
        self.btn_screen_salinity_category.resize(180, 60)
        self.btn_screen_salinity_category.move(10, y_location)

        """
        self.btn_screen_water_velocity = QPushButton('Water velocity', self)
        self.btn_screen_water_velocity.clicked.connect(self.on_screen_water_velocity_button_clicked)
        self.btn_screen_water_velocity.resize(180, 80)
        self.btn_screen_water_velocity.move(10, 380)
        """

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
        """

        """
        self.textbox = QLineEdit(self)
        self.textbox.resize(380, 60)
        self.textbox.move(10, 240)

        
        self.btn_update = QPushButton('Change delta', self)
        self.btn_update.clicked.connect(self.on_update_button_clicked)
        self.btn_update.resize(380, 60)
        self.btn_update.move(10, 320)
        self.btn_update.setStyleSheet("background-color:lightgray;")
        """

        y_location = 180

        self.next_run = QLabel('Next run:', self)
        # self.lbl_model_info.setStyleSheet("font-weight: bold; font-size: 36")
        self.next_run.setFont(QFont('Arial', 12))
        self.next_run.move(10, y_location)
        self.next_run.setFixedWidth(180)
        self.next_run.setAlignment(Qt.AlignCenter)

        y_location += label_spacing

        self.lbl_model_info = QLabel('turn %s - run %d' % (self.gui.game.turn, self.gui.game.turn_count), self)
        #self.lbl_model_info.setStyleSheet("font-weight: bold; font-size: 36")
        self.lbl_model_info.setFont(QFont('Arial', 12))
        self.lbl_model_info.move(10, y_location)
        self.lbl_model_info.setFixedWidth(180)
        self.lbl_model_info.setAlignment(Qt.AlignCenter)

        y_location += label_spacing

        self.btn_run_model = QPushButton('Run model', self)
        self.btn_run_model.setFont(QFont('Arial', 10))
        self.btn_run_model.clicked.connect(self.on_run_model_button_clicked)
        self.btn_run_model.resize(180, 60)
        self.btn_run_model.move(10, y_location)
        self.btn_run_model.setStyleSheet("background-color:lightgray;")

        y_location += (button_height + button_spacing)

        self.btn_end_round = QPushButton('End round', self)
        self.btn_end_round.setFont(QFont('Arial', 10))
        self.btn_end_round.clicked.connect(self.on_end_round_button_clicked)
        self.btn_end_round.resize(180, 60)
        self.btn_end_round.move(10, y_location)
        self.btn_end_round.setStyleSheet("background-color:lightgray;")
        """
        self.btn_board_water_velocity = QPushButton('Water velocity', self)
        self.btn_board_water_velocity.clicked.connect(self.on_board_water_velocity_button_clicked)
        self.btn_board_water_velocity.resize(180, 80)
        self.btn_board_water_velocity.move(210, 380)
        """

        y_location = 400

        self.lbl_boundary = QLabel('Output selection', self)
        self.lbl_boundary.setFont(QFont('Arial', 12))
        #self.lbl_boundary.setStyleSheet("font-weight: bold; font-size: 36")
        self.lbl_boundary.move(10, y_location)
        self.lbl_boundary.setFixedWidth(180)
        self.lbl_boundary.setAlignment(Qt.AlignCenter)

        y_location += label_spacing

        self.lbl_turn1 = QLabel('2018:', self)
        self.lbl_turn1.setFont(QFont('Arial', 12))
        # self.lbl_boundary.setStyleSheet("font-weight: bold; font-size: 36")
        self.lbl_turn1.move(10, y_location)
        self.lbl_turn1.setFixedWidth(180)
        self.lbl_turn1.setAlignment(Qt.AlignCenter)

        y_location += label_spacing

        self.btn_turn1 = QPushButton('Reference', self)
        self.btn_turn1.setFont(QFont('Arial', 10))
        # self.btn_turn1 = QPushButton('2017', self)
        self.btn_turn1.clicked.connect(self.on_turn1_button_clicked)
        self.btn_turn1.resize(180, 60)
        self.btn_turn1.move(10, y_location)

        y_location += (button_height + button_spacing)

        self.lbl_turn2 = QLabel('2050 Hd - run:', self)
        self.lbl_turn2.setFont(QFont('Arial', 12))
        # self.lbl_boundary.setStyleSheet("font-weight: bold; font-size: 36")
        self.lbl_turn2.move(10, y_location)
        self.lbl_turn2.setFixedWidth(180)
        self.lbl_turn2.setAlignment(Qt.AlignCenter)

        y_location += label_spacing

        button_width = 50
        button_height = 60
        x_location = 10
        #y_location = 590
        x_spacing = 15
        y_spacing = 15

        self.btn_turn2_run1 = QPushButton('1', self)
        self.btn_turn2_run1.setFont(QFont('Arial', 12))
        # self.btn_turn2 = QPushButton('2018 +\n partly deepened NWW', self)
        output_update = partial(self.on_turn2_button_clicked, 1)
        #self.btn_turn2_run1.clicked.connect(self.on_turn2_run1_button_clicked)
        self.btn_turn2_run1.clicked.connect(output_update)
        self.btn_turn2_run1.resize(button_width, button_height)
        self.btn_turn2_run1.move(x_location, y_location)
        self.btn_turn2_run1.setEnabled(False)

        x_location += (x_spacing + button_width)
        if x_location >= 190:
            x_location = 10
            y_location += (y_spacing + button_height)

        self.btn_turn2_run2 = QPushButton('2', self)
        self.btn_turn2_run2.setFont(QFont('Arial', 12))
        # self.btn_turn2 = QPushButton('2018 +\n partly deepened NWW', self)
        output_update = partial(self.on_turn2_button_clicked, 2)
        #self.btn_turn2_run2.clicked.connect(self.on_turn2_run2_button_clicked)
        self.btn_turn2_run2.clicked.connect(output_update)
        self.btn_turn2_run2.resize(button_width, button_height)
        self.btn_turn2_run2.move(x_location, y_location)
        self.btn_turn2_run2.setEnabled(False)

        x_location += (x_spacing + button_width)
        if x_location >= 190:
            x_location = 10
            y_location += (y_spacing + button_height)

        self.btn_turn2_run3 = QPushButton('3', self)
        self.btn_turn2_run3.setFont(QFont('Arial', 12))
        # self.btn_turn2 = QPushButton('2018 +\n partly deepened NWW', self)
        output_update = partial(self.on_turn2_button_clicked, 3)
        self.btn_turn2_run3.clicked.connect(output_update)
        #self.btn_turn2_run3.clicked.connect(self.on_turn2_run3_button_clicked)
        self.btn_turn2_run3.resize(button_width, button_height)
        self.btn_turn2_run3.move(x_location, y_location)
        self.btn_turn2_run3.setEnabled(False)

        x_location += (x_spacing + button_width)
        if x_location >= 190:
            x_location = 10
            y_location += (y_spacing + button_height)

        self.btn_turn2_run4 = QPushButton('4', self)
        self.btn_turn2_run4.setFont(QFont('Arial', 12))
        # self.btn_turn2 = QPushButton('2018 +\n partly deepened NWW', self)
        output_update = partial(self.on_turn2_button_clicked, 4)
        self.btn_turn2_run4.clicked.connect(output_update)
        #self.btn_turn2_run4.clicked.connect(self.on_turn2_run4_button_clicked)
        self.btn_turn2_run4.resize(button_width, button_height)
        self.btn_turn2_run4.move(x_location, y_location)
        self.btn_turn2_run4.setEnabled(False)

        x_location += (x_spacing + button_width)
        if x_location >= 190:
            x_location = 10
            y_location += (y_spacing + button_height)

        self.btn_turn2_run5 = QPushButton('5', self)
        self.btn_turn2_run5.setFont(QFont('Arial', 12))
        # self.btn_turn2 = QPushButton('2018 +\n partly deepened NWW', self)
        output_update = partial(self.on_turn2_button_clicked, 5)
        self.btn_turn2_run5.clicked.connect(output_update)
        #self.btn_turn2_run5.clicked.connect(self.on_turn2_run5_button_clicked)
        self.btn_turn2_run5.resize(button_width, button_height)
        self.btn_turn2_run5.move(x_location, y_location)
        self.btn_turn2_run5.setEnabled(False)

        """
        self.btn_turn2 = QPushButton('2050 Hd', self)
        self.btn_turn2.setFont(QFont('Arial', 10))
        # self.btn_turn2 = QPushButton('2018 +\n partly deepened NWW', self)
        self.btn_turn2.clicked.connect(self.on_turn2_button_clicked)
        self.btn_turn2.resize(180, 60)
        self.btn_turn2.move(10, 740)
        self.btn_turn2.setEnabled(False)
        """

        y_location += (button_height + y_spacing)

        self.lbl_turn3 = QLabel('2100 Hd - run:', self)
        self.lbl_turn3.setFont(QFont('Arial', 12))
        # self.lbl_boundary.setStyleSheet("font-weight: bold; font-size: 36")
        self.lbl_turn3.move(10, y_location)
        self.lbl_turn3.setFixedWidth(180)
        self.lbl_turn3.setAlignment(Qt.AlignCenter)

        """
        self.btn_turn3 = QPushButton('2100 Hd', self)
        self.btn_turn3.setFont(QFont('Arial', 10))
        #self.btn_turn3 = QPushButton('2100le (+1m SLR) +\n deepened NWW & Nieuwe Maas', self)
        self.btn_turn3.clicked.connect(self.on_turn3_button_clicked)
        self.btn_turn3.resize(180, 60)
        self.btn_turn3.move(10, 820)
        self.btn_turn3.setEnabled(False)
        """

        x_location = 10
        y_location += label_spacing

        self.btn_turn3_run1 = QPushButton('1', self)
        self.btn_turn3_run1.setFont(QFont('Arial', 12))
        output_update = partial(self.on_turn3_button_clicked, 1)
        #self.btn_turn3_run2.clicked.connect(self.on_turn2_run1_button_clicked)
        self.btn_turn3_run1.clicked.connect(output_update)
        self.btn_turn3_run1.resize(button_width, button_height)
        self.btn_turn3_run1.move(x_location, y_location)
        self.btn_turn3_run1.setEnabled(False)

        x_location += (x_spacing + button_width)
        if x_location >= 190:
            x_location = 10
            y_location += (y_spacing + button_height)

        self.btn_turn3_run2 = QPushButton('2', self)
        self.btn_turn3_run2.setFont(QFont('Arial', 12))
        output_update = partial(self.on_turn3_button_clicked, 2)
        #self.btn_turn3_run2.clicked.connect(self.on_turn2_run2_button_clicked)
        self.btn_turn3_run2.clicked.connect(output_update)
        self.btn_turn3_run2.resize(button_width, button_height)
        self.btn_turn3_run2.move(x_location, y_location)
        self.btn_turn3_run2.setEnabled(False)

        x_location += (x_spacing + button_width)
        if x_location >= 190:
            x_location = 10
            y_location += (y_spacing + button_height)

        self.btn_turn3_run3 = QPushButton('3', self)
        self.btn_turn3_run3.setFont(QFont('Arial', 12))
        output_update = partial(self.on_turn3_button_clicked, 3)
        self.btn_turn3_run3.clicked.connect(output_update)
        #self.btn_turn3_run3.clicked.connect(self.on_turn2_run3_button_clicked)
        self.btn_turn3_run3.resize(button_width, button_height)
        self.btn_turn3_run3.move(x_location, y_location)
        self.btn_turn3_run3.setEnabled(False)

        x_location += (x_spacing + button_width)
        if x_location >= 190:
            x_location = 10
            y_location += (y_spacing + button_height)

        self.btn_turn3_run4 = QPushButton('4', self)
        self.btn_turn3_run4.setFont(QFont('Arial', 12))
        # self.btn_turn2 = QPushButton('2018 +\n partly deepened NWW', self)
        output_update = partial(self.on_turn3_button_clicked, 4)
        self.btn_turn3_run4.clicked.connect(output_update)
        #self.btn_turn2_run4.clicked.connect(self.on_turn2_run4_button_clicked)
        self.btn_turn3_run4.resize(button_width, button_height)
        self.btn_turn3_run4.move(x_location, y_location)
        self.btn_turn3_run4.setEnabled(False)

        x_location += (x_spacing + button_width)
        if x_location >= 190:
            x_location = 10
            y_location += (y_spacing + button_height)

        self.btn_turn3_run5 = QPushButton('5', self)
        self.btn_turn3_run5.setFont(QFont('Arial', 12))
        output_update = partial(self.on_turn3_button_clicked, 5)
        self.btn_turn3_run5.clicked.connect(output_update)
        #self.btn_turn3_run5.clicked.connect(self.on_turn2_run5_button_clicked)
        self.btn_turn3_run5.resize(button_width, button_height)
        self.btn_turn3_run5.move(x_location, y_location)
        self.btn_turn3_run5.setEnabled(False)

        self.turn_buttons = {1: [self.btn_turn1],
                             2: [self.btn_turn2_run1, self.btn_turn2_run2, self.btn_turn2_run3, self.btn_turn2_run4, self.btn_turn2_run5],
                             3: [self.btn_turn3_run1, self.btn_turn3_run2, self.btn_turn3_run3, self.btn_turn3_run4, self.btn_turn3_run5]}
        return

    def on_update_button_clicked(self):
        textboxValue = self.textbox.text()
        self.gui.game.get_changes(textboxValue)
        return

    def on_run_model_button_clicked(self):
        self.gui.game.update()
        self.lbl_model_info.setText('turn %s - run %d' % (self.gui.game.turn, self.gui.game.turn_count))
        return

    def on_end_round_button_clicked(self):
        #if self.gui.game.turn < 4:
        if self.gui.game.turn < 3:
            self.gui.game.end_round()
        self.lbl_model_info.setText('turn %s - run %d' % (self.gui.game.turn, self.gui.game.turn_count))
        #self.gui.game.update()
        return

    def on_salinity_button_clicked(self):
        self.viz_tracker.viz_variable = "water_salinity"
        self.change_highlights()
        return

    def on_salinity_category_button_clicked(self):
        self.viz_tracker.viz_variable = "salinity_category"
        self.change_highlights()
        return

    """
    def on_board_salinity_button_clicked(self):
        self.viz_tracker.game_variable = "water_salinity"
        self.change_highlights()
        return

    def on_board_salinity_category_button_clicked(self):
        self.viz_tracker.game_variable = "salinity_category"
        self.change_highlights()
        return
    """

    def on_turn1_button_clicked(self):
        self.viz_tracker.turn = 1
        self.viz_tracker.run = 1
        self.gui.show_forcing_conditions()
        self.gui.plot_inlet_indicators()
        self.gui.plot_salinity_inlets()
        self.gui.score_widget.update_text()
        self.change_highlights()
        return

    """
    def on_turn2_button_clicked(self):
        self.viz_tracker.turn = 2
        self.gui.show_forcing_conditions()
        self.gui.plot_inlet_indicators()
        self.gui.plot_salinity_inlets()
        self.gui.score_widget.update_text()
        self.change_highlights()
        return
    """

    def on_turn2_button_clicked(self, run):
        self.viz_tracker.turn = 2
        self.viz_tracker.run = run
        self.gui.show_forcing_conditions()
        self.gui.plot_inlet_indicators()
        self.gui.plot_salinity_inlets()
        self.gui.score_widget.update_text()
        self.change_highlights()
        return

    def on_turn3_button_clicked(self, run):
        self.viz_tracker.turn = 3
        self.viz_tracker.run = run
        self.gui.show_forcing_conditions()
        self.gui.plot_inlet_indicators()
        self.gui.plot_salinity_inlets()
        self.gui.score_widget.update_text()
        self.change_highlights()
        return

    def on_turn4_button_clicked(self):
        self.viz_tracker.turn = 4
        self.gui.show_forcing_conditions()
        self.gui.plot_inlet_indicators()
        self.gui.plot_salinity_inlets()
        self.gui.score_widget.update_text()
        self.change_highlights()
        return


    def change_highlights(self):
        if self.screen_highlight != self.viz_tracker.viz_variable:
            self.screen_highlight = self.viz_tracker.viz_variable
            self.btn_screen_salinity.setStyleSheet("background-color:lightgray;")
            self.btn_screen_salinity_category.setStyleSheet("background-color:lightgray;")
            if self.screen_highlight == "water_salinity":
                self.btn_screen_salinity.setStyleSheet("background-color:red;")
            elif self.screen_highlight == "salinity_category":
                self.btn_screen_salinity_category.setStyleSheet("background-color:blue;")
        """
        if self.board_highlight != self.viz_tracker.game_variable:
            self.board_highlight = self.viz_tracker.game_variable
            self.btn_board_salinity.setStyleSheet("background-color:lightgray;")
            self.btn_board_salinity_category.setStyleSheet("background-color:lightgray;")
            if self.board_highlight == "water_salinity":
                self.btn_board_salinity.setStyleSheet("background-color:red;")
            elif self.board_highlight == "salinity_category":
                self.btn_board_salinity_category.setStyleSheet("background-color:blue;")
        """
        if self.turn_highlight != self.viz_tracker.turn or self.run_highlight != self.viz_tracker.run:
            self.turn_highlight = self.viz_tracker.turn
            self.run_highlight = self.viz_tracker.run
            for key, btn_list in self.turn_buttons.items():
                for btn in btn_list:
                    btn.setStyleSheet("background-color:lightgray;")
            """
            self.btn_turn1.setStyleSheet("background-color:lightgray;")
            self.btn_turn2_run1.setStyleSheet("background-color:lightgray;")
            self.btn_turn2_run2.setStyleSheet("background-color:lightgray;")
            self.btn_turn2_run3.setStyleSheet("background-color:lightgray;")
            self.btn_turn2_run4.setStyleSheet("background-color:lightgray;")
            self.btn_turn2_run5.setStyleSheet("background-color:lightgray;")
            self.btn_turn3.setStyleSheet("background-color:lightgray;")
            #self.btn_turn4.setStyleSheet("background-color:lightgray;")
            
            if self.turn_highlight == 1:
                self.btn_turn1.setStyleSheet("background-color:cyan;")
            elif self.turn_highlight == 2:
                self.btn_turn2.setStyleSheet("background-color:magenta;")
            elif self.turn_highlight == 3:
                self.btn_turn3.setStyleSheet("background-color:yellow;")
            """
            if self.turn_highlight == 1:
                background_color="background-color:cyan;"
            elif self.turn_highlight == 2:
                background_color="background-color:magenta;"
            elif self.turn_highlight == 3:
                background_color="background-color:yellow;"
            self.turn_buttons[self.turn_highlight][self.run_highlight-1].setStyleSheet(background_color)
            #elif self.turn_highlight == 4:
            #    self.btn_turn4.setStyleSheet("background-color:green;")
        return



class VisualizationTracker():
    def __init__(self, starting_turn, scenarios, starting_variable, time_steps, starting_time,
                 salinity_range, salinity_category, inlet_to_plot): #, water_level_range, water_velocity_range
        self._turn = starting_turn
        self._run = 1
        self.scenarios = scenarios
        self._scenario = scenarios[0]
        self._viz_variable = starting_variable
        self._time_steps = time_steps
        self._time_index = starting_time
        self._salinity_norm = salinity_range
        self._salinity_category_norm = salinity_category
        self._inlet_to_plot = inlet_to_plot
        #self._water_level_norm = water_level_range
        #self._water_velocity_norm = water_velocity_range
        return

    def get_time_index(self):
        t_idx = self.time_index % len(self.time_steps)
        return self.time_steps[t_idx]

    def update_scenario(self):
        self.scenario = self.scenarios[self.turn-1]
        return

    @property
    def turn(self):
        return self._turn

    @property
    def run(self):
        return self._run

    @property
    def scenario(self):
        return self._scenario

    @property
    def viz_variable(self):
        return self._viz_variable

    """
    @property
    def game_variable(self):
        return self._game_variable
    """

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

    @property
    def inlet_to_plot(self):
        return self._inlet_to_plot

    """
    @property
    def water_level_norm(self):
        return self._water_level_norm

    @property
    def water_velocity_norm(self):
        return self._water_velocity_norm
    """

    @turn.setter
    def turn(self, turn):
        self._turn = turn
        return

    @run.setter
    def run(self, run):
        self._run = run
        return

    @scenario.setter
    def scenario(self, scenario):
        self._scenario = scenario
        return


    @viz_variable.setter
    def viz_variable(self, variable):
        self._viz_variable = variable
        return

    """
    @game_variable.setter
    def game_variable(self, variable):
        self._game_variable = variable
        return
    """

    @time_steps.setter
    def time_steps(self, time_steps):
        self._time_steps = time_steps
        return

    @time_index.setter
    def time_index(self, time_index):
        self._time_index += time_index
        return

    @inlet_to_plot.setter
    def inlet_to_plot(self, inlet_to_plot):
        self._inlet_to_plot = inlet_to_plot
        return