#Модуль UDP-приёмника для получения телеметрических данных от мобильного устройства.

import socket
import json
import threading
import queue
import logging
# Настройка локального логирования для отладки сетевых пакетов
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] (%(threadName)s) %(message)s')

class UDPReceiver:
    def __init__(self, ip: str, port: int, data_queue: queue.Queue):
        """
        Класс для асинхронного приема телеметрии IMU по протоколу UDP.
        
        :param ip: IP-адрес для прослушивания (например, '0.0.0.0' для всех интерфейсов)
        :param port: UDP-порт (например, 5005)
        :param data_queue: Потокобезопасная очередь queue.Queue для передачи пакетов в UI
        """
        self.ip = ip
        self.port = port
        self.data_queue = data_queue
        
        self.sock = None
        self.is_running = False
        self.thread = None
        
    def start(self):
        """Инициализация сокета и запуск приёма данных в фоновом демоническом потоке."""
        if self.is_running:
            logging.warning("UDP-приёмник уже запущен.")
            return
            
        try:
            # Создаем UDP сокет (SOCK_DGRAM)
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            
            # Позволяет операционной системе повторно использовать порт сразу после закрытия
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Привязываем сокет к IP и Порту
            self.sock.bind((self.ip, self.port))
            
            # Устанавливаем таймаут блокирующих операций (recvfrom) в 1 секунду.
            # Это критически важно, чтобы при вызове stop() поток завершался, а не зависал вечно.
            self.sock.settimeout(1.0) 
            
        except Exception as e:
            logging.error(f"Не удалось инициализировать UDP сокет на {self.ip}:{self.port}. Ошибка: {e}")
            if self.sock:
                self.sock.close()
            raise e
            
        # Устанавливаем флаг работы и запускаем фоновый поток
        self.is_running = True
        self.thread = threading.Thread(target=self._listen_loop, name="UDP_Receiver_Thread", daemon=True)
        self.thread.start()
        logging.info(f"Сетевой поток UDP успешно запущен на {self.ip}:{self.port}")

    def stop(self):
        """Безопасная остановка сетевого потока и закрытие ресурсов сокета."""
        if not self.is_running:
            return
            
        logging.info("Остановка сетевого потока UDP")
        self.is_running = False
        
        # Ожидаем завершения фонового потока (максимум 1.5 секунды)
        if self.thread:
            self.thread.join(timeout=1.5)
            self.thread = None
            
        # Закрываем сетевой сокет
        if self.sock:
            try:
                self.sock.close()
            except Exception as e:
                logging.error(f"Ошибка при закрытии сокета: {e}")
            self.sock = None
            
        logging.info("Сетевой поток UDP полностью остановлен.")

    def _listen_loop(self):
        while self.is_running:
            try:
                # Читаем ровно 4096 байт
                data, addr = self.sock.recvfrom(4096)
                packet = json.loads(data.decode('utf-8'))
                
                # Упрощаем проверку: если есть хоть какие-то данные, кидаем в очередь
                if "accel" in packet:
                    self.data_queue.put_nowait(packet)
            except socket.timeout:
                continue
            except Exception as e:
                if self.is_running:
                    print(f"ОШИБКА В ПОТОКЕ ПРИЕМА: {e}") # Прямой вывод в консоль
                break