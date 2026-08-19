"""
Виджет для отображения графиков фильтров на основе matplotlib.
Встраивается в customtkinter через FigureCanvasTkAgg.
"""
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import customtkinter as ctk
from typing import Dict, List, Optional

# Цвета для разных фильтров
FILTER_COLORS = {
    'raw': 'gray',
    'exponential': 'blue',
    'kalman': 'green',
    'iir': 'red'
}

# Стили линий
FILTER_STYLES = {
    'raw': '--',
    'exponential': '-',
    'kalman': '-.',
    'iir': ':'
}


class PlotWidget(ctk.CTkFrame):
    """
    Виджет для отображения графика с несколькими линиями (по фильтрам).
    """

    def __init__(
        self,
        master,
        width: int = 800,
        height: int = 600,
        xlabel: str = "Время (пакеты)",
        ylabel: str = "Позиция X (м)",
        axis: str = 'x'  # 'x', 'y', 'z'
    ):
        """
        :param master: родительский виджет (CTkFrame или CTk)
        :param width: ширина виджета
        :param height: высота виджета
        :param xlabel: подпись оси X
        :param ylabel: подпись оси Y
        :param axis: какая ось отображается ('x', 'y', 'z')
        """
        super().__init__(master, width=width, height=height)
        self.pack_propagate(False)

        self.xlabel = xlabel
        self.ylabel = ylabel
        self.axis = axis  # 'x', 'y', 'z'

        # Данные: для каждого фильтра список значений (по оси Y)
        self.data: Dict[str, List[float]] = {
            'raw': [],
            'exponential': [],
            'kalman': [],
            'iir': []
        }
        # Время (номер пакета) – общее для всех
        self.time_values: List[int] = []

        # Какие фильтры сейчас видны
        self.visible_filters: List[str] = ['raw', 'exponential', 'kalman', 'iir']

        # Создаём фигуру и оси
        self.fig, self.ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
        self.fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)

        # Настройка осей
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.grid(True, linestyle='--', alpha=0.6)
        self.ax.set_title("Сравнение фильтров")

        # Встраиваем холст в customtkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True)

        # Словарь для хранения объектов линий (для обновления)
        self.lines: Dict[str, Optional[plt.Line2D]] = {name: None for name in self.data.keys()}

    def add_data_point(self, filter_name: str, value: float, time: int) -> None:
        """
        Добавляет новую точку для указанного фильтра.

        :param filter_name: имя фильтра ('raw', 'exponential', 'kalman', 'iir')
        :param value: значение позиции по выбранной оси (Y)
        :param time: номер пакета (время) – общий для всех фильтров
        """
        if filter_name not in self.data:
            return  # игнорируем неизвестный фильтр

        # Если это первый пакет, инициализируем список времени
        if len(self.time_values) == 0:
            self.time_values.append(time)
        else:
            # Добавляем время только если оно новое (если пакет приходит с новым номером)
            if time != self.time_values[-1]:
                self.time_values.append(time)

        # Добавляем значение для данного фильтра
        self.data[filter_name].append(value)

        # Если длина данных для этого фильтра превышает 10000, усекаем (чтобы не переполнять память)
        max_points = 10000
        if len(self.data[filter_name]) > max_points:
            self.data[filter_name] = self.data[filter_name][-max_points:]
            # Также усекаем время, если оно стало длиннее данных
            if len(self.time_values) > max_points:
                self.time_values = self.time_values[-max_points:]

    def set_visible_filters(self, filter_names: List[str]) -> None:
        """Устанавливает, какие фильтры отображать на графике."""
        self.visible_filters = filter_names

    def update_plot(self) -> None:
        """Перерисовывает график с текущими данными и видимыми фильтрами."""
        # Очищаем оси (но не удаляем настройки)
        self.ax.clear()
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.grid(True, linestyle='--', alpha=0.6)
        self.ax.set_title("Сравнение фильтров")

        # Если нет данных, просто показываем пустой график
        if len(self.time_values) == 0:
            self.canvas.draw()
            return

        # Для каждого фильтра, который должен быть виден, строим линию
        for filter_name in self.visible_filters:
            if filter_name not in self.data:
                continue
            y_data = self.data[filter_name]
            if len(y_data) == 0:
                continue
            # Берём только те точки, которые соответствуют длине y_data
            x_data = self.time_values[:len(y_data)]
            color = FILTER_COLORS.get(filter_name, 'black')
            style = FILTER_STYLES.get(filter_name, '-')
            label = filter_name.capitalize()  # для легенды
            self.ax.plot(x_data, y_data, color=color, linestyle=style,
                         linewidth=2, label=label)

        # Легенда
        self.ax.legend(loc='upper left')

        # Масштабирование осей
        self.ax.relim()
        self.ax.autoscale_view()

        # Перерисовываем холст
        self.canvas.draw()

    def clear_plot(self) -> None:
        """Очищает все данные и график."""
        for key in self.data:
            self.data[key].clear()
        self.time_values.clear()
        self.update_plot()

    def set_axis(self, axis: str) -> None:
        """
        Устанавливает, какая ось отображается (x, y, z).
        Обновляет подпись оси Y.
        """
        self.axis = axis
        self.ylabel = f"Позиция {axis.upper()} (м)"
        self.ax.set_ylabel(self.ylabel)
        self.update_plot()