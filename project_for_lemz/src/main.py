import customtkinter as ctk
from network.udp_receiver import UDPReceiver
from processing.imu_processor import IMUProcessor
import numpy as np

# Глобальный экземпляр процессора
processor = IMUProcessor()
prev_timestamp = None

def on_packet(packet):
    """
    Callback-функция, вызываемая при получении валидного UDP-пакета.
    Передаёт данные в IMUProcessor и выводит результат.
    """
    global prev_timestamp

    # Вычисляем dt (разница во времени между пакетами)
    current_timestamp = packet['timestamp']
    if prev_timestamp is None:
        dt = 0.0  # первый пакет
    else:
        dt = current_timestamp - prev_timestamp
        if dt <= 0:
            dt = 0.01  # защита от отрицательного или нулевого dt
    prev_timestamp = current_timestamp

    # Извлекаем данные из пакета
    accel = packet['accel']
    gyro = packet['gyro']
    quat = packet['quat']

    # Обрабатываем пакет в IMUProcessor
    result = processor.process(accel, gyro, quat, dt)

    # Выводим результат
    pos = result['position']
    vel = result['velocity']
    print(f"Пакет #{packet['packet_id']} | dt={dt:.3f}с | "
          f"Позиция: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) м | "
          f"Скорость: ({vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f}) м/с")

def main():
    # Настройка внешнего вида
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    # Создание главного окна
    root = ctk.CTk()
    root.title("ВОнТР - Визуальная Одометрия на Телефоне Realme")
    root.geometry("800x600")

    # Простая надпись для проверки
    label = ctk.CTkLabel(
        root,
        text="Приложение «ВОнТР» v1.0\n\nОбработка IMU-данных...",
        font=("Arial", 24)
    )
    label.pack(pady=25)

    # Запуск UDP-приёмника
    receiver = UDPReceiver(port=5005)
    receiver.set_callback(on_packet)

    if receiver.start():
        print("UDP-приёмник успешно запущен. Ожидание данных...")
    else:
        print("Не удалось запустить UDP-приёмник. Проверьте, не занят ли порт 5005.")
        label.configure(text="ОШИБКА: не удалось запустить UDP-приёмник")

    # При закрытии окна останавливаем приёмник
    def on_closing():
        receiver.stop()
        print("Приложение завершено.")
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_closing)

    # Запуск основного цикла
    root.mainloop()

if __name__ == "__main__":
    main()