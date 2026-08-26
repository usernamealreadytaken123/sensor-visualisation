# Математическое ядро VIO с поддержкой разных фильтров.

import numpy as np
import logging
from scipy.spatial.transform import Rotation as R

class IMUProcessor:
    def __init__(self):
        """
        Математическое ядро ИНС. Вычисляет координаты траектории 
        на основе двойного интегрирования данных акселерометра с телефона.
        """
        self.reset()

    def reset(self):
        """Полный сброс навигационных параметров и возврат в начальную точку (0,0,0)."""
        self.last_timestamp = None    # Время предыдущего пакета для вычисления dt
        
        # Навигационные векторы состояния в мировой (земной) СК
        self.position = np.zeros(3)       # Позиция [X, Y, Z] в метрах
        self.velocity = np.zeros(3)       # Скорость [Vx, Vy, Vz] в м/с
        self.last_accel_nav = np.zeros(3) # Чистое ускорение на прошлом шаге (для трапеций)
        
        # Пороги для алгоритма детекции покоя (ZVU - Zero Velocity Update)
        # Помогают бороться с шумом, когда телефон неподвижен
        self.accel_static_threshold = 0.15  # м/с² (порог для ускорения)
        self.gyro_static_threshold = 0.05   # рад/с (порог для гироскопа)
        
        logging.info("Математическое ядро ИНС (IMUProcessor) успешно сброшено.")

    def process_packet(self, packet: dict) -> dict:
        """
        Конвейер обработки входящего UDP пакета телеметрии.
        
        :param packet: Десериализованный JSON-словарь от UDPReceiver
        :return: Словарь со структурированными сырыми навигационными данными для фильтров
        """
        # 1. Извлечение векторов и перевод в массивы numpy
        accel_body = np.array(packet["accel"])  # Ускорение в осях телефона
        gyro_body = np.array(packet["gyro"])    # Угловая скорость в осях телефона
        quat_raw = np.array(packet["quat"])     # Кватернион ориентации [x, y, z, w]

        # Метка времени уже в секундах
        current_timestamp = packet["timestamp"]  

        # 2. Инициализация при первом старте сессии
        if self.last_timestamp is None:
            self.last_timestamp = current_timestamp
            self.last_accel_nav = self._get_pure_accel_nav(accel_body, quat_raw)
            output = self._build_output_dict(quat_raw, accel_body)
            output["dt"] = 0.02  # Начальный дефолтный шаг
            return output

        # 3. Вычисление динамического шага времени (dt)
        dt = current_timestamp - self.last_timestamp
        
        # Защита от сетевого джиттера или зависших пакетов
        if dt <= 0 or dt > 0.5:
            dt = 0.02  # Резервный шаг, эквивалентный частоте 50 Гц

        # 4. Перенос ускорения в мировую СК и компенсация вектора гравитации
        accel_nav = self._get_pure_accel_nav(accel_body, quat_raw)

        # 5. Алгоритм детекции покоя (ZVU - Zero Velocity Update)
        # Если шум датчиков ниже порога, устройство считается неподвижным
        is_static = (np.linalg.norm(accel_nav) < self.accel_static_threshold and 
                     np.linalg.norm(gyro_body) < self.gyro_static_threshold)

        if is_static:
            self.velocity = np.zeros(3)  # Сбрасываем линейную ошибку скорости в ноль
            accel_nav = np.zeros(3)      # Обнуляем ускорение, чтобы позиция не дрейфовала
        else:
            # 6. Двойное интегрирование методом трапеций (повышенная точность)
            # Шаг 1: Находим скорость (V = V_prev + (a_prev + a_curr) / 2 * dt)
            self.velocity += 0.5 * (self.last_accel_nav + accel_nav) * dt
            
            # Шаг 2: Находим позицию траектории (P = P_prev + V * dt)
            self.position += self.velocity * dt

        # Сохраняем параметры текущего шага для следующей итерации конвейера
        self.last_timestamp = current_timestamp
        self.last_accel_nav = accel_nav

        # Формируем выходной пакет и подмешиваем dt для менеджера фильтров
        output = self._build_output_dict(quat_raw, accel_body)
        output["dt"] = dt 
        return output

    def _get_pure_accel_nav(self, accel_body: np.ndarray, quat: np.ndarray) -> np.ndarray:
        """Преобразование систем координат и вычитание силы тяжести g."""
        try:
            # Создаем объект пространственного вращения из кватерниона смартфона
            # Scipy ожидает формат кватерниона [x, y, z, w]
            rotation = R.from_quat(quat)
            
            # Поворачиваем вектор ускорения из телефонной СК в мировую СК
            accel_nav = rotation.apply(accel_body)
            
            # Гравитация вычитается как полноценный мировой вектор [0, 0, 9.81].
            # Это защищает оси X и Y от накопления бокового смещения при наклонах телефона.
            accel_nav -= np.array([0.0, 0.0, 9.81])
            
            return accel_nav
        except Exception as e:
            logging.error(f"Математическая ошибка трансформации осей ИНС: {e}")
            return np.zeros(3)

    def _build_output_dict(self, quat: np.ndarray, accel_body: np.ndarray) -> dict:
        """Формирование стандартизированного словаря для передачи в FilterManager."""
        return {
            "timestamp": self.last_timestamp,
            "position": self.position.copy(),      # Сырые координаты [X, Y, Z] в метрах
            "velocity": self.velocity.copy(),      # Скорость [Vx, Vy, Vz] в м/с
            "accel_body": accel_body.copy(),       # Сырое ускорение для анализа шумов
            "quat": quat.copy()                    # Кватернион поворота для 3D графики
        }
