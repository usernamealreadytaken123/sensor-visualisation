# Менеджер фильтров для одновременного применения нескольких фильтров к одним и тем же данным.
# Поддерживаемые фильтры:
# - RAW (без фильтрации)
# - Exponential (экспоненциальное сглаживание)
# - Kalman (фильтр Калмана)
#- IIR (БИХ-фильтр Баттерворта 2-го порядка)

import numpy as np
import logging

# =====================================================================
# РЕАЛИЗАЦИЯ МАТЕМАТИЧЕСКИХ АЛГОРИТМОВ ФИЛЬТРАЦИИ
# =====================================================================

class ExponentialFilter:
    def __init__(self, alpha: float = 0.1):
        # Экспоненциальное сглаживание (Low-pass filter).
        self.alpha = alpha
        self.prev_value = None

    def reset(self):
        self.prev_value = None

    def filter(self, current_value: np.ndarray) -> np.ndarray:
        if self.prev_value is None:
            self.prev_value = current_value.copy()
            return self.prev_value
        
        # Формула: Y_k = alpha * X_k + (1 - alpha) * Y_(k-1)
        self.prev_value = self.alpha * current_value + (1.0 - self.alpha) * self.prev_value
        return self.prev_value.copy()


class IIRFilter:
    def __init__(self, b: float = 0.15, a: float = 0.85):
        """
        Классический бесконечный импульсный фильтр (БИХ) 1-го порядка.
        Соответствует разностному уравнению из ТЗ.
        """
        self.b = b
        self.a = a
        self.prev_filtered = None

    def reset(self):
        self.prev_filtered = None

    def filter(self, current_value: np.ndarray) -> np.ndarray:
        if self.prev_filtered is None:
            self.prev_filtered = current_value.copy()
            return self.prev_filtered
        
        # Формула БИХ-фильтра 1-го порядка: Y_k = b * X_k + a * Y_(k-1)
        filtered = self.b * current_value + self.a * self.prev_filtered
        self.prev_filtered = filtered.copy()
        return filtered


class KalmanFilter3D:
    def __init__(self, q: float = 0.001, r: float = 0.05):
        """
        Упрощенный трехмерный фильтр Калмана для сглаживания координат траектории.
        Работает независимо по каждой из осей X, Y, Z.
        """
        self.q = q  # Дисперсия шума процесса (насколько доверяем физической модели)
        self.r = r  # Дисперсия шума измерений (насколько сильно шумит датчик)
        self.reset()

    def reset(self):
        self.x = None  # Оценка состояния [X, Y, Z]
        self.p = np.ones(3) * 1.0  # Ошибка ковариации оценки

    def filter(self, measurement: np.ndarray) -> np.ndarray:
        if self.x is None:
            self.x = measurement.copy()
            return self.x

        # Цикл по 3 осям (X, Y, Z)
        for i in range(3):
            # 1. Шаг предсказания (Модель постоянного состояния)
            # x_pred = x[i]
            p_pred = self.p[i] + self.q

            # 2. Шаг коррекции (Обновление по измерениям акселерометра)
            k_gain = p_pred / (p_pred + self.r)  # Коэффициент Калмана
            self.x[i] = self.x[i] + k_gain * (measurement[i] - self.x[i])
            self.p[i] = (1.0 - k_gain) * p_pred

        return self.x.copy()


# =====================================================================
# МЕНЕДЖЕР УПРАВЛЕНИЯ ФИЛЬТРАМИ (КОНВЕЙЕР ОБРАБОТКИ)
# =====================================================================

class FilterManager:
    def __init__(self):
        """
        Менеджер каскада фильтрации. Применяет все доступные алгоритмы 
        к одним и тем же сырым входным навигационным координатам.
        """
        # Инициализация каждого фильтра с оптимальными базовыми коэффициентами
        self.exp_filter = ExponentialFilter(alpha=0.12)
        self.iir_filter = IIRFilter(b=0.15, a=0.85)
        self.kalman_filter = KalmanFilter3D(q=0.005, r=0.08)
        
        logging.info("Менеджер фильтрации (FilterManager) успешно инициализирован.")

    def reset(self):
        """Полный сброс предыстории всех внутренних фильтров."""
        self.exp_filter.reset()
        self.iir_filter.reset()
        self.kalman_filter.reset()
        logging.info("Состояние всех алгоритмов фильтрации сброшено.")

    def apply_filters(self, raw_data: dict) -> dict:
        """
        Принимает сырые данные из IMUProcessor и параллельно накладывает сглаживание.
        
        :param raw_data: Словарь с ключами: position, velocity, accel_body, quat
        :return: Структурированный пакет со всеми отфильтрованными траекториями
        """
        raw_position = raw_data["position"] # Координаты [X, Y, Z] до фильтрации

        # Параллельный расчет каждого фильтра (сохраняет непрерывность предыстории)
        filtered_exp = self.exp_filter.filter(raw_position)
        filtered_iir = self.iir_filter.filter(raw_position)
        filtered_kalman = self.kalman_filter.filter(raw_position)

        # Формируем итоговый пакет. Эти ключи в точности совпадают с маской 
        # чекбоксов, которую мы заложили в main.py!
        return {
            "timestamp": raw_data.get("timestamp"), # Передаем временную метку дальше (для 2D осей)
            "quat": raw_data["quat"],               # Кватернион поворота для 3D анимации
            "RAW": raw_position,                     # Траектория без фильтров
            "EXP": filtered_exp,                     # Траектория после экспоненциального сглаживания
            "IIR": filtered_iir,                     # Траектория после БИХ-фильтра
            "KALMAN": filtered_kalman               # Траектория после фильтра Калмана
        }