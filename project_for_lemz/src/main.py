import os
import sys
import queue
import logging
import customtkinter as ctk

# Настройка путей для корректного импорта модулей из папки src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Импорт разработанных модулей системы ВОнТР
from network.udp_receiver import UDPReceiver
from processing.imu_processor import IMUProcessor
from processing.filter_manager import FilterManager
from ui.plot_widget import PlotWidget

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] (%(threadName)s) %(message)s')


class VOnTR_App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Конфигурация главного окна ---
        self.title("ВОнТР v_1 — Визуальная Одометрия Realme")
        self.geometry("1200x800")
        self.minsize(1000, 600)
        ctk.set_appearance_mode("System")  # Автоматическая темная/светлая тема
        ctk.set_default_color_theme("blue")

        # --- Инициализация сетевой архитектуры и ядра ---
        self.data_queue = queue.Queue(maxsize=5000)  # Потокобезопасная очередь передачи пакетов
        self.receiver = None                         # Объект сетевого приёмника UDP
        self.processor = IMUProcessor()             # Математический интегратор ИНС
        self.filter_manager = FilterManager()       # Управление каскадом фильтров
        
        self.is_gathering = False                   # Флаг активного приема
        self.packet_count = 0                       # Глобальный счетчик пакетов
        self._queue_timer_id = None                 # ID активного таймера Tkinter

        # Построение графического интерфейса
        self._create_widgets()

    def _create_widgets(self):
        """Создание и адаптивная компоновка элементов GUI с помощью резиновой Grid-сетки."""
        self.grid_columnconfigure(0, weight=25, minsize=260)  
        self.grid_columnconfigure(1, weight=75)
        self.grid_rowconfigure(0, weight=1)

        # =====================================================================
        # ЛЕВАЯ СЛУЖЕБНАЯ ПАНЕЛЬ
        # =====================================================================
        self.sidebar = ctk.CTkFrame(self, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.sidebar.grid_columnconfigure(0, weight=1)
        self.sidebar.grid_rowconfigure(0, weight=0)
        self.sidebar.grid_rowconfigure(1, weight=0)
        self.sidebar.grid_rowconfigure(2, weight=0)
        self.sidebar.grid_rowconfigure(3, weight=0)
        self.sidebar.grid_rowconfigure(4, weight=0)
        self.sidebar.grid_rowconfigure(5, weight=1)  # Пружина
        self.sidebar.grid_rowconfigure(6, weight=0)
        self.sidebar.grid_rowconfigure(7, weight=0)
        self.sidebar.grid_rowconfigure(8, weight=0)
        self.sidebar.grid_rowconfigure(9, weight=0)

        self.title_label = ctk.CTkLabel(self.sidebar, text="ВОнТР v_1", font=ctk.CTkFont(size=22, weight="bold"))
        self.title_label.grid(row=0, column=0, padx=20, pady=(10, 2), sticky="ew")
        
        self.subtitle_label = ctk.CTkLabel(self.sidebar, text="Инерциальная одометрия IMU", font=ctk.CTkFont(size=12, slant="italic"))
        self.subtitle_label.grid(row=1, column=0, padx=20, pady=(0, 10), sticky="ew")

        # Настройки подключения UDP
        self.net_frame = ctk.CTkFrame(self.sidebar)
        self.net_frame.grid(row=2, column=0, sticky="ew", padx=15, pady=2)
        self.net_frame.grid_columnconfigure(0, weight=1)

        self.net_title = ctk.CTkLabel(self.net_frame, text="Настройки UDP Connection", font=ctk.CTkFont(weight="bold"))
        self.net_title.grid(row=0, column=0, sticky="w", padx=15, pady=(8, 4))

        self.ip_label = ctk.CTkLabel(self.net_frame, text="IP Адрес:")
        self.ip_label.grid(row=1, column=0, sticky="w", padx=15, pady=(2, 0))
        self.ip_entry = ctk.CTkEntry(self.net_frame, height=26)
        self.ip_entry.insert(0, "127.0.0.1")
        self.ip_entry.grid(row=2, column=0, sticky="ew", padx=15, pady=(0, 4))

        self.port_label = ctk.CTkLabel(self.net_frame, text="Порт:")
        self.port_label.grid(row=3, column=0, sticky="w", padx=15, pady=(2, 0))
        self.port_entry = ctk.CTkEntry(self.net_frame, height=26)
        self.port_entry.insert(0, "5005")
        self.port_entry.grid(row=4, column=0, sticky="ew", padx=15, pady=(0, 10))

        # Переключатели алгоритмов фильтрации
        self.filter_frame = ctk.CTkFrame(self.sidebar)
        self.filter_frame.grid(row=3, column=0, sticky="ew", padx=15, pady=2)
        self.filter_frame.grid_columnconfigure(0, weight=1)

        self.filter_title = ctk.CTkLabel(self.filter_frame, text="Алгоритмы фильтрации", font=ctk.CTkFont(weight="bold"))
        self.filter_title.grid(row=0, column=0, sticky="w", padx=15, pady=(8, 4))

        self.cb_raw = ctk.CTkCheckBox(self.filter_frame, text="Сырые данные (RAW)", command=self._on_filter_changed)
        self.cb_raw.select()
        self.cb_raw.grid(row=1, column=0, sticky="w", padx=15, pady=3)

        self.cb_exp = ctk.CTkCheckBox(self.filter_frame, text="Экспоненциальный фильтр", command=self._on_filter_changed)
        self.cb_exp.select()
        self.cb_exp.grid(row=2, column=0, sticky="w", padx=15, pady=3)

        self.cb_iir = ctk.CTkCheckBox(self.filter_frame, text="БИХ-фильтр (IIR)", command=self._on_filter_changed)
        self.cb_iir.select()
        self.cb_iir.grid(row=3, column=0, sticky="w", padx=15, pady=3)

        self.cb_kalman = ctk.CTkCheckBox(self.filter_frame, text="Фильтр Калмана", command=self._on_filter_changed)
        self.cb_kalman.select()
        self.cb_kalman.grid(row=4, column=0, sticky="w", padx=15, pady=(3, 10))

        # Выбор проекции осей
        self.axis_frame = ctk.CTkFrame(self.sidebar)
        self.axis_frame.grid(row=4, column=0, sticky="ew", padx=15, pady=2)
        self.axis_frame.grid_columnconfigure(0, weight=1)

        self.axis_title = ctk.CTkLabel(self.axis_frame, text="Отображаемые оси", font=ctk.CTkFont(weight="bold"))
        self.axis_title.grid(row=0, column=0, sticky="w", padx=15, pady=(8, 4))
        
        self.axis_var = ctk.StringVar(value="X") # Ставим X по умолчанию, как в вашем plot_widget.py
        self.rb_3d = ctk.CTkRadioButton(self.axis_frame, text="3D Траектория", variable=self.axis_var, value="3D", command=self._on_axis_changed)
        self.rb_3d.grid(row=1, column=0, sticky="w", padx=15, pady=3)
        self.rb_x = ctk.CTkRadioButton(self.axis_frame, text="Ось X (Времени)", variable=self.axis_var, value="X", command=self._on_axis_changed)
        self.rb_x.grid(row=2, column=0, sticky="w", padx=15, pady=3)
        self.rb_y = ctk.CTkRadioButton(self.axis_frame, text="Ось Y (Времени)", variable=self.axis_var, value="Y", command=self._on_axis_changed)
        self.rb_y.grid(row=3, column=0, sticky="w", padx=15, pady=3)
        self.rb_z = ctk.CTkRadioButton(self.axis_frame, text="Ось Z (Времени)", variable=self.axis_var, value="Z", command=self._on_axis_changed)
        self.rb_z.grid(row=4, column=0, sticky="w", padx=15, pady=(3, 10))

        self.spacer = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.spacer.grid(row=5, column=0, sticky="nsew")

        self.info_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.info_frame.grid(row=6, column=0, sticky="ew", padx=15, pady=2)

        self.status_label = ctk.CTkLabel(self.info_frame, text="Статус: Готов", text_color="#1F6AA5", font=ctk.CTkFont(weight="bold"))
        self.status_label.pack(anchor="w", pady=1)
        
        self.counter_label = ctk.CTkLabel(self.info_frame, text="Принято пакетов: 0")
        self.counter_label.pack(anchor="w", pady=1)

        self.btn_reset = ctk.CTkButton(self.sidebar, text="Сбросить траекторию", fg_color="gray", hover_color="#555555", height=32, command=self.reset_trajectory)
        self.btn_reset.grid(row=7, column=0, sticky="ew", padx=15, pady=3)

        self.btn_stop = ctk.CTkButton(self.sidebar, text="ОСТАНОВИТЬ", fg_color="red", hover_color="darkred", state="disabled", height=32, command=self.stop_gathering)
        self.btn_stop.grid(row=8, column=0, sticky="ew", padx=15, pady=3)

        self.btn_start = ctk.CTkButton(self.sidebar, text="ЗАПУСТИТЬ СБОР", fg_color="green", hover_color="darkgreen", height=32, command=self.start_gathering)
        self.btn_start.grid(row=9, column=0, sticky="ew", padx=15, pady=(3, 15))

        # =====================================================================
        # ПРАВАЯ ПАНЕЛЬ С ГРАФИКОМ (MATPLOTLIB)
        # =====================================================================
        self.plot_container = ctk.CTkFrame(self)
        self.plot_container.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        self.plot_widget = PlotWidget(self.plot_container)
        self.plot_widget.pack(fill="both", expand=True, padx=5, pady=5)

    def start_gathering(self):
        """Инициализация и старт фонового сетевого потока приема."""
        if self._queue_timer_id is not None:
            self.after_cancel(self._queue_timer_id)
            self._queue_timer_id = None

        ip = self.ip_entry.get().strip()
        try:
            port = int(self.port_entry.get().strip())
        except ValueError:
            self.status_label.configure(text="Статус: Неверный порт!", text_color="red")
            return

        self.receiver = UDPReceiver(ip, port, self.data_queue)
        try:
            self.receiver.start()
            self.is_gathering = True
            
            self.btn_start.configure(state="disabled")
            self.btn_stop.configure(state="normal")
            self.btn_reset.configure(state="disabled")
            self.ip_entry.configure(state="disabled")
            self.port_entry.configure(state="disabled")
            
            self.status_label.configure(text="Статус: Приём данных...", text_color="green")
            # Запускаем циклический опрос сетевой очереди
            self._queue_timer_id = self.after(20, self.check_queue)
            
        except Exception as e:
            self.status_label.configure(text="Статус: Ошибка запуска!", text_color="red")
            logging.error(f"Критический сбой инициализации сокета: {e}")

    def stop_gathering(self):
        """Принудительная остановка фонового UDP-потока и сброс таймера."""
        self.is_gathering = False
        if self._queue_timer_id is not None:
            self.after_cancel(self._queue_timer_id)
            self._queue_timer_id = None

        if self.receiver:
            self.receiver.stop()
            self.receiver = None

        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.btn_reset.configure(state="normal")
        self.ip_entry.configure(state="normal")
        self.port_entry.configure(state="normal")
        
        self.status_label.configure(text="Статус: Остановлен", text_color="#1F6AA5")

    def reset_trajectory(self):
        """Полный сброс истории фильтров и очистка холста."""
        self.processor.reset()
        self.filter_manager.reset()
        
        # Безопасный вызов метода очистки в зависимости от версии вашего PlotWidget
        if hasattr(self.plot_widget, 'clear_plot'):
            self.plot_widget.clear_plot()
        else:
            self.plot_widget.clear_plots()
            
        self.packet_count = 0
        self.counter_label.configure(text="Принято пакетов: 0")
        
        while not self.data_queue.empty():
            try:
                self.data_queue.get_nowait()
            except queue.Empty:
                break
                
        self.status_label.configure(text="Статус: Графики очищены", text_color="cyan")

    def check_queue(self):
        """Фоновый опрос сетевой очереди с прямой распаковкой аргументов."""
        if not self.is_gathering:
            return

        packets_processed = 0
        
        while not self.data_queue.empty():
            try:
                packet = self.data_queue.get_nowait()
                
                if not isinstance(packet, dict):
                    self.data_queue.task_done()
                    continue

                acc_data = packet["accel"]
                if isinstance(acc_data, dict):
                    accel_raw = [float(acc_data.get('x', 0)), float(acc_data.get('y', 0)), float(acc_data.get('z', 0))]
                else:
                    accel_raw = [float(x) for x in acc_data]
                    
                gyro_data = packet["gyro"]
                if isinstance(gyro_data, dict):
                    gyro_raw = [float(gyro_data.get('x', 0)), float(gyro_data.get('y', 0)), float(gyro_data.get('z', 0))]
                else:
                    gyro_raw = [float(x) for x in gyro_data]
                    
                q_data = packet["quat"]
                if isinstance(q_data, dict):
                    quat = [float(q_data.get('x', 0)), float(q_data.get('y', 0)), float(q_data.get('z', 0)), float(q_data.get('w', 1))]
                else:
                    quat = [float(x) for x in q_data]
                if len(quat) == 3: 
                    quat.append(1.0)
                
                dt = 0.02  # Стабильный фиксированный шаг для эмулятора (50 Гц)
                
                # Вызов вашей математической функции process() с 4 аргументами
                raw_data = self.processor.process(accel_raw, gyro_raw, quat, dt)
                
                if raw_data:
                    filtered_data = self.filter_manager.apply_filters(raw_data)
                    
                    # Передаем отфильтрованные точки в массивы вашего PlotWidget
                    time_idx = int(self.packet_count)
                    key_mapping = {
                        "RAW": "raw",
                        "EXP": "exponential",
                        "IIR": "iir",
                        "KALMAN": "kalman"
                    }
                    
                    for f_max, f_str in key_mapping.items():
                        if f_max in filtered_data:
                            # Проверяем выбранную ось в виджете ('x', 'y' или 'z')
                            axis_char = getattr(self.plot_widget, "axis", "x").lower()
                            axis_idx = {"x": 0, "y": 1, "z": 2}[axis_char]
                            
                            val = float(filtered_data[f_max]["position"][axis_idx])
                            self.plot_widget.add_data_point(f_str, val, time_idx)
                    
                    packets_processed += 1
                    self.packet_count += 1
                
                self.data_queue.task_done()
            except Exception as e:
                print(f"СБОЙ КОНВЕЙЕРА В MAIN: {e}")

        if packets_processed > 0:
            self.counter_label.configure(text=f"Принято пакетов: {self.packet_count}")
            if hasattr(self.plot_widget, 'update_plot'):
                self.plot_widget.update_plot()
            else:
                self.plot_widget.update_plots()

        if self.is_gathering:
            self._queue_timer_id = self.after(20, self.check_queue)

    def _on_filter_changed(self):
        """Считывание чекбоксов и передача маски видимости линий графиков."""
        if hasattr(self.plot_widget, 'toggle_filters_visibility'):
            filter_mask = {
                "RAW": bool(self.cb_raw.get()),
                "EXP": bool(self.cb_exp.get()),
                "IIR": bool(self.cb_iir.get()),
                "KALMAN": bool(self.cb_kalman.get())
            }
            self.plot_widget.toggle_filters_visibility(filter_mask)

    def _on_axis_changed(self):
        """Переключение режимов отображения (Временная развертка X, Y, Z)."""
        if hasattr(self.plot_widget, 'change_view_mode'):
            selected_axis = self.axis_var.get()
            self.plot_widget.change_view_mode(selected_axis)


if __name__ == "__main__":
    app = VOnTR_App()
    try:
        app.mainloop()
    except KeyboardInterrupt:
        logging.info("Приложение завершило работу.")