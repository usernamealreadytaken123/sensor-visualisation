# Виджет для отображения графиков фильтров на основе matplotlib.
# Встраивается в customtkinter через FigureCanvasTkAgg.
# Поддерживает динамическое переключение между 3D-траекторией и 2D-осями времени.

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import customtkinter as ctk
import numpy as np
from collections import deque
from typing import Dict

# Конфигурация стилей согласно ТЗ (ключи строго согласованы с FilterManager и main.py)
FILTER_COLORS = {
    'RAW': 'gray',
    'EXP': 'blue',
    'IIR': 'red',
    'KALMAN': 'green'
}

FILTER_STYLES = {
    'RAW': '--',
    'EXP': '-',
    'IIR': ':',
    'KALMAN': '-.'
}

FILTER_LABELS = {
    'RAW': 'Сырые данные (RAW)',
    'EXP': 'Экспоненциальный',
    'IIR': 'БИХ-фильтр (IIR)',
    'KALMAN': 'Фильтр Калмана'
}


class PlotWidget(ctk.CTkFrame):
    def __init__(self, master, width: int = 800, height: int = 600, max_points: int = 10000):
        """
        Оптимизированный виджет визуализации траекторий ИНС девайса Realme.
        """
        super().__init__(master, width=width, height=height)
        self.pack_propagate(False)

        self.max_points = max_points
        self.view_mode = "3D"  # Текущий режим отображения: "3D", "X", "Y", "Z"
        
        # Настройки видимости каналов фильтрации (по умолчанию включены все)
        self.visible_filters = {key: True for key in FILTER_COLORS.keys()}

        # Высокоэффективное хранилище истории точек на базе очередей deque (фиксированная длина)
        self.history: Dict[str, deque] = {key: deque(maxlen=self.max_points) for key in FILTER_COLORS.keys()}
        self.time_steps = deque(maxlen=self.max_points)  # Индексы пакетов времени
        self.packet_index = 0

        # Инициализация контейнера под фигуру Matplotlib
        self.fig = plt.figure(figsize=(width/100, height/100), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True)

        self.ax = None
        self.lines: Dict[str, plt.Line2D] = {}

        # Первичное построение графического окна
        self._rebuild_axes()

    def _rebuild_axes(self):
        """Пересборка сцены при смене режима (3D <-> 2D). Очистка тяжелых элементов осей."""
        self.fig.clear()
        self.lines.clear()

        if self.view_mode == "3D":
            # Инициализация трехмерного пространства
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.set_xlabel("Позиция X (м)")
            self.ax.set_ylabel("Позиция Y (м)")
            self.ax.set_zlabel("Позиция Z (м)")
        else:
            # Инициализация двухмерных осей времени
            self.ax = self.fig.add_subplot(111)
            self.ax.set_xlabel("Время (пакеты)")
            self.ax.set_ylabel(f"Позиция {self.view_mode} (м)")
            self.ax.grid(True, linestyle='--', alpha=0.5)

        self.ax.set_title("Сравнение траекторий и алгоритмов фильтрации")

        # Создаем постоянные объекты линий ОДИН раз во избежание утечки памяти процессора
        for key in FILTER_COLORS.keys():
            color = FILTER_COLORS[key]
            style = FILTER_STYLES[key]
            label = FILTER_LABELS[key]

            if self.view_mode == "3D":
                line, = self.ax.plot([], [], [], color=color, linestyle=style, linewidth=1.8, label=label)
            else:
                line, = self.ax.plot([], [], color=color, linestyle=style, linewidth=1.8, label=label)
            
            line.set_visible(self.visible_filters[key])
            self.lines[key] = line

        self.ax.legend(loc="upper left")
        self.fig.tight_layout()
        self.canvas.draw()

    def add_data_point(self, filtered_packet: dict) -> None:
        self.time_steps.append(self.packet_index)
        self.packet_index += 1

        for key in self.history.keys():
            if key in filtered_packet:
                # Убираем .copy(), так как данные уже являются копиями
                self.history[key].append(filtered_packet[key])

    def update_plots(self) -> None:
        """
        Высокоскоростное обновление линий БЕЗ полной очистки ax.clear() экрана.
        """
        if len(self.time_steps) == 0:
            return

        has_data = False
        t_data = np.array(self.time_steps)

        # Меняем массивы точек внутри объектов линий напрямую (летучее обновление)
        for key, line in self.lines.items():
            if not self.visible_filters[key] or len(self.history[key]) == 0:
                line.set_visible(False)
                continue

            line.set_visible(True)
            has_data = True
            pts = np.array(self.history[key])

            if self.view_mode == "3D":
                line.set_data(pts[:, 0], pts[:, 1])
                line.set_3d_properties(pts[:, 2])
            else:
                axis_idx = {"X": 0, "Y": 1, "Z": 2}[self.view_mode]
                line.set_data(t_data, pts[:, axis_idx])

        if has_data:
            self.ax.relim()
            self.ax.autoscale_view()
            # Легкая ленивая перерисовка экрана
            self.canvas.draw_idle()

    def toggle_filters_visibility(self, filter_mask: dict) -> None:
        """Включение/выключение линий на основе чекбоксов из главного окна."""
        for key in self.visible_filters.keys():
            if key in filter_mask:
                self.visible_filters[key] = filter_mask[key]
        self.update_plots()

    def change_view_mode(self, new_mode: str) -> None:
        """Переключение проекции графиков (3D, X, Y, Z). Вызывается радиокнопками."""
        if self.view_mode == new_mode:
            return
        self.view_mode = new_mode
        self._rebuild_axes()
        self.update_plots()

    def clear_plots(self) -> None:
        """Полная очистка графиков и сброс истории (вызывается кнопкой 'Сбросить')."""
        for queue_obj in self.history.values():
            queue_obj.clear()
        self.time_steps.clear()
        self.packet_index = 0
        
        for line in self.lines.values():
            if self.view_mode == "3D":
                line.set_data([], [])
                line.set_3d_properties([])
            else:
                line.set_data([], [])

        self.canvas.draw_idle()