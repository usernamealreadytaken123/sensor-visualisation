import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import customtkinter as ctk
from network.udp_receiver import UDPReceiver
from processing.imu_processor import IMUProcessor
from processing.filter_manager import FilterManager
from visualization.plot_widget import PlotWidget

# ========== Настройки ==========
FILTER_ALPHA = 0.3
KALMAN_PROCESS_NOISE = 1e-4
KALMAN_MEASUREMENT_NOISE = 1e-2
IIR_CUTOFF = 0.1
IIR_ORDER = 2
UDP_PORT = 5005
# ===============================

# Глобальные объекты
receiver = None
processor = None
filter_manager = None
plot_widget = None

# Состояние
is_running = False
prev_timestamp = None

# Доступные фильтры (для чекбоксов)
FILTER_NAMES = ['raw', 'exponential', 'kalman', 'iir']
FILTER_DISPLAY = {
    'raw': 'Без фильтрации (RAW)',
    'exponential': 'Экспоненциальный',
    'kalman': 'Калмана',
    'iir': 'IIR / БИХ'
}

def on_packet(packet):
    global prev_timestamp, processor, filter_manager, plot_widget

    current_timestamp = packet['timestamp']
    if prev_timestamp is None:
        dt = 0.0
    else:
        dt = current_timestamp - prev_timestamp
        if dt <= 0:
            dt = 0.01
    prev_timestamp = current_timestamp

    accel = packet['accel']
    gyro = packet['gyro']
    quat = packet['quat']

    # Получаем сырые данные от IMUProcessor
    result = processor.process(accel, gyro, quat, dt)
    position_raw = result['position_raw'].tolist()  # [x, y, z]

    # Применяем все фильтры через FilterManager
    filtered_data = filter_manager.process(position_raw, dt)

    # Обновляем график для выбранных фильтров
    current_axis = plot_widget.axis  # 'x', 'y', 'z'
    axis_idx = {'x': 0, 'y': 1, 'z': 2}[current_axis]

    # Для каждого фильтра добавляем точку
    for filter_name in FILTER_NAMES:
        if filter_name in filtered_data:
            value = filtered_data[filter_name][axis_idx]
            plot_widget.add_data_point(filter_name, value, filter_manager.get_packet_count())

    # Обновляем график
    plot_widget.update_plot()

    # Обновляем счётчик в интерфейсе (если есть)
    if status_label is not None:
        status_label.configure(text=f"Пакетов: {filter_manager.get_packet_count()}")

def start_processing():
    global receiver, processor, filter_manager, plot_widget, is_running, prev_timestamp

    if is_running:
        return

    # Сброс состояния
    prev_timestamp = None
    plot_widget.clear_plot()
    filter_manager.reset()

    # Создаём IMUProcessor (он нам нужен только для вычисления сырых данных)
    processor = IMUProcessor(
        filter_type='exponential',  # всё равно не используется для сырых
        filter_alpha=FILTER_ALPHA
    )

    # Создаём FilterManager
    filter_manager = FilterManager(
        filter_alpha=FILTER_ALPHA,
        kalman_process_noise=KALMAN_PROCESS_NOISE,
        kalman_measurement_noise=KALMAN_MEASUREMENT_NOISE,
        iir_cutoff=IIR_CUTOFF,
        iir_order=IIR_ORDER
    )

    # Запускаем UDPReceiver
    receiver = UDPReceiver(port=UDP_PORT)
    receiver.set_callback(on_packet)

    if receiver.start():
        is_running = True
        btn_start.configure(text="Стоп")
        # Блокируем чекбоксы
        for cb in checkboxes.values():
            cb.configure(state="disabled")
        status_label.configure(text="Работает...", text_color="green")
        print("Приём данных начат.")
    else:
        status_label.configure(text="Ошибка запуска приёмника", text_color="red")
        print("Не удалось запустить UDP-приёмник.")

def stop_processing():
    global receiver, is_running

    if not is_running:
        return

    if receiver:
        receiver.stop()
        receiver = None

    is_running = False
    btn_start.configure(text="Начать")
    # Разблокируем чекбоксы
    for cb in checkboxes.values():
        cb.configure(state="normal")
    status_label.configure(text="Остановлен", text_color="orange")
    print("Приём данных остановлен.")

def toggle_processing():
    if is_running:
        stop_processing()
    else:
        start_processing()

def on_closing():
    stop_processing()
    root.destroy()

# ========== СОЗДАНИЕ GUI ==========
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

root = ctk.CTk()
root.title("ВОнТР - Визуальная Одометрия (сравнение фильтров)")
root.geometry("1200x700")
root.grid_columnconfigure(0, weight=3)  # график занимает больше места
root.grid_columnconfigure(1, weight=1)  # панель управления
root.grid_rowconfigure(0, weight=1)

# Левая часть – график
left_frame = ctk.CTkFrame(root)
left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
left_frame.grid_columnconfigure(0, weight=1)
left_frame.grid_rowconfigure(0, weight=1)

plot_widget = PlotWidget(left_frame, width=800, height=600, axis='x')
plot_widget.pack(fill="both", expand=True)

# Правая часть – панель управления
right_frame = ctk.CTkFrame(root, width=250)
right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
right_frame.grid_propagate(False)

# Заголовок
label_title = ctk.CTkLabel(
    right_frame,
    text="Выберите фильтры для отображения",
    font=("Arial", 16, "bold")
)
label_title.pack(pady=(20, 10))

# Чекбоксы для каждого фильтра
checkboxes = {}
filter_vars = {}

for fname in FILTER_NAMES:
    var = ctk.BooleanVar(value=True)  # по умолчанию все включены
    filter_vars[fname] = var
    cb = ctk.CTkCheckBox(
        right_frame,
        text=FILTER_DISPLAY[fname],
        variable=var,
        command=lambda: update_visible_filters()
    )
    cb.pack(pady=5, padx=20, anchor="w")
    checkboxes[fname] = cb

def update_visible_filters():
    """Обновляет список видимых фильтров на графике."""
    visible = [fname for fname, var in filter_vars.items() if var.get()]
    plot_widget.set_visible_filters(visible)
    plot_widget.update_plot()

# Переключатель оси
axis_label = ctk.CTkLabel(right_frame, text="Отображаемая ось:")
axis_label.pack(pady=(20, 5))

axis_var = ctk.StringVar(value="x")
axis_menu = ctk.CTkOptionMenu(
    right_frame,
    values=["x", "y", "z"],
    variable=axis_var,
    command=lambda choice: plot_widget.set_axis(choice)
)
axis_menu.pack(pady=(0, 20))

# Кнопка "Начать/Стоп"
btn_start = ctk.CTkButton(
    right_frame,
    text="Начать",
    font=("Arial", 16),
    command=toggle_processing
)
btn_start.pack(pady=20)

# Статус
status_label = ctk.CTkLabel(
    right_frame,
    text="Готов к работе",
    font=("Arial", 14)
)
status_label.pack(pady=10)

# Кнопка сброса графика
def reset_plot():
    if not is_running:
        plot_widget.clear_plot()
        filter_manager.reset()
        status_label.configure(text="Сброшено", text_color="yellow")
    else:
        status_label.configure(text="Сначала остановите приём", text_color="red")

btn_reset = ctk.CTkButton(
    right_frame,
    text="Сбросить график",
    command=reset_plot
)
btn_reset.pack(pady=10)

# Обработка закрытия окна
root.protocol("WM_DELETE_WINDOW", on_closing)

# Запуск основного цикла
root.mainloop()

"""
import customtkinter as ctk
from network.udp_receiver import UDPReceiver
from processing.imu_processor import IMUProcessor

# Создаём процессор с экспоненциальным фильтром
processor = IMUProcessor(filter_alpha=0.3)

prev_timestamp = None

def on_packet(packet):
    global prev_timestamp
    current_timestamp = packet['timestamp']
    if prev_timestamp is None:
        dt = 0.0
    else:
        dt = current_timestamp - prev_timestamp
        if dt <= 0:
            dt = 0.01
    prev_timestamp = current_timestamp

    accel = packet['accel']
    gyro = packet['gyro']
    quat = packet['quat']

    result = processor.process(accel, gyro, quat, dt)

    pos_raw = result['position_raw']
    pos_smooth = result['position']
    vel_raw = result['velocity_raw']
    vel_smooth = result['velocity']

    print(f"\n--- Пакет #{packet['packet_id']} | dt={dt:.3f}с ---")
    print(f"  POSITION RAW  : ({pos_raw[0]:.3f}, {pos_raw[1]:.3f}, {pos_raw[2]:.3f}) м")
    print(f"  POSITION SMOOTH: ({pos_smooth[0]:.3f}, {pos_smooth[1]:.3f}, {pos_smooth[2]:.3f}) м")
    print(f"  VELOCITY RAW  : ({vel_raw[0]:.3f}, {vel_raw[1]:.3f}, {vel_raw[2]:.3f}) м/с")
    print(f"  VELOCITY SMOOTH: ({vel_smooth[0]:.3f}, {vel_smooth[1]:.3f}, {vel_smooth[2]:.3f}) м/с")

def main():
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    root = ctk.CTk()
    root.title("ВОнТР - Визуальная Одометрия на Телефоне Realme")
    root.geometry("800x600")

    label = ctk.CTkLabel(
        root,
        text="Приложение «ВОнТР» v1.0\n\nЭкспоненциальный фильтр\nСмотрите вывод в консоли",
        font=("Arial", 20)
    )
    label.pack(pady=50)

    receiver = UDPReceiver(port=5005)
    receiver.set_callback(on_packet)

    if receiver.start():
        print("UDP-приёмник запущен. Ожидание данных...")
        print("=" * 60)
        print("RAW — без фильтрации | SMOOTH — после экспоненциального сглаживания")
        print("=" * 60)
    else:
        print("Не удалось запустить UDP-приёмник. Проверьте порт 5005.")
        label.configure(text="ОШИБКА: не удалось запустить UDP-приёмник")

    def on_closing():
        receiver.stop()
        print("Приложение завершено.")
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_closing)

    root.mainloop()

if __name__ == "__main__":
    main()
"""

    