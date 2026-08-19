"""
Модуль UDP-приёмника для получения телеметрических данных от мобильного устройства.
"""

import socket
import json
import threading
import logging
from typing import Callable, Optional, Dict, Any

logger = logging.getLogger(__name__)


class UDPReceiver:
    """
    Класс для приёма UDP-датаграмм с JSON-пакетами.
    Работает в отдельном потоке.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 5005, buffer_size: int = 1024):
        """
        Инициализация приёмника.

        :param host: IP-адрес для привязки (0.0.0.0 - все интерфейсы)
        :param port: UDP-порт
        :param buffer_size: максимальный размер пакета в байтах
        """
        self.host = host
        self.port = port
        self.buffer_size = buffer_size

        self._socket: Optional[socket.socket] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callback: Optional[Callable[[Dict[str, Any]], None]] = None

        # Счётчики для статистики
        self.packets_received = 0
        self.packets_parsed = 0
        self.packets_errors = 0

    def set_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Устанавливает функцию, которая будет вызываться при получении валидного пакета.

        :param callback: функция, принимающая один аргумент - словарь с данными пакета
        """
        self._callback = callback

    def start(self) -> bool:
        """
        Запускает UDP-приёмник в отдельном потоке.

        :return: True если запуск успешен, иначе False
        """
        if self._running:
            logger.warning("Приёмник уже запущен")
            return False

        try:
            # Создаём UDP-сокет
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._socket.bind((self.host, self.port))
            # Устанавливаем таймаут на recvfrom (чтобы можно было прервать поток)
            self._socket.settimeout(0.5)

            self._running = True
            self._thread = threading.Thread(target=self._receive_loop, daemon=True)
            self._thread.start()

            logger.info(f"UDP-приёмник запущен на {self.host}:{self.port}")
            return True

        except Exception as e:
            logger.error(f"Ошибка запуска UDP-приёмника: {e}")
            return False

    def stop(self) -> None:
        """Останавливает приёмник и закрывает сокет."""
        if not self._running:
            return

        self._running = False
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        logger.info("UDP-приёмник остановлен")

    def _receive_loop(self) -> None:
        """Основной цикл приёма данных (выполняется в отдельном потоке)."""
        while self._running and self._socket:
            try:
                data, addr = self._socket.recvfrom(self.buffer_size)
                self.packets_received += 1

                # Декодируем и парсим JSON
                try:
                    json_str = data.decode('utf-8').strip()
                    packet = json.loads(json_str)

                    # Валидация обязательных полей
                    if self._validate_packet(packet):
                        self.packets_parsed += 1
                        if self._callback:
                            self._callback(packet)
                    else:
                        self.packets_errors += 1
                        logger.warning(f"Невалидный пакет от {addr}: пропущен")

                except json.JSONDecodeError as e:
                    self.packets_errors += 1
                    logger.warning(f"Ошибка парсинга JSON от {addr}: {e}")
                except UnicodeDecodeError as e:
                    self.packets_errors += 1
                    logger.warning(f"Ошибка декодирования UTF-8 от {addr}: {e}")

            except socket.timeout:
                # Таймаут нужен для проверки флага _running
                continue
            except Exception as e:
                logger.error(f"Ошибка в цикле приёма: {e}")
                break

        logger.info("Цикл приёма завершён")

    def _validate_packet(self, packet: dict) -> bool:
        """
        Проверяет наличие всех необходимых полей в пакете.

        :param packet: словарь с пакетом
        :return: True если пакет валиден, иначе False
        """
        required_fields = ['packet_id', 'timestamp', 'accel', 'gyro', 'quat']
        if not all(field in packet for field in required_fields):
            return False

        # Проверяем, что accel, gyro, quat содержат координаты x, y, z (или w, x, y, z для quat)
        for subfield in ['accel', 'gyro']:
            if not isinstance(packet[subfield], dict):
                return False
            if not all(axis in packet[subfield] for axis in ['x', 'y', 'z']):
                return False

        if not isinstance(packet['quat'], dict):
            return False
        if not all(axis in packet['quat'] for axis in ['w', 'x', 'y', 'z']):
            return False

        return True

    def get_stats(self) -> dict:
        """Возвращает статистику приёма."""
        return {
            'packets_received': self.packets_received,
            'packets_parsed': self.packets_parsed,
            'packets_errors': self.packets_errors,
        }