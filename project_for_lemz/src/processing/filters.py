# Модуль фильтрации данных.
# Поддерживаемые фильтры:
# - Exponential (экспоненциальное сглаживание)
# - Kalman (одномерный фильтр Калмана для позиции и скорости)
# - IIR (БИХ-фильтр Баттерворта 2-го порядка) – требует scipy

# Модуль фильтрации данных.
# Поддерживаемые фильтры:
# - Exponential (экспоненциальное сглаживание)
# - Kalman (одномерный фильтр Калмана для позиции и скорости)
# - IIR (БИХ-фильтр Баттерворта 2-го порядка) – требует scipy

import numpy as np
from typing import List, Optional

# ===================== Базовые фильтры =====================

class ExponentialSmoothing:
    def __init__(self, alpha: float = 0.3, initial_value: float = 0.0):
        self.alpha = alpha
        self.value = float(initial_value)
        self._initialized = False

    def update(self, raw_value: float) -> float:
        if not self._initialized:
            self.value = float(raw_value)
            self._initialized = True
        else:
            self.value = self.alpha * float(raw_value) + (1.0 - self.alpha) * self.value
        return self.value

    def reset(self, value: float = 0.0):
        self.value = float(value)
        self._initialized = False

    def get(self) -> float:
        return self.value


class KalmanFilter1D:
    """
    Высокостабильный скалярный фильтр Калмана (Позиция-Скорость).
    Полностью защищен от сбоев размерностей NumPy.
    """
    def __init__(self, dt: float = 0.02, process_noise: float = 1e-4, measurement_noise: float = 1e-2, initial_pos: float = 0.0, initial_vel: float = 0.0):
        self.dt = dt
        self.proc_noise = process_noise
        self.meas_noise = measurement_noise
        self.reset(initial_pos, initial_vel)

    def reset(self, pos: float = 0.0, vel: float = 0.0):
        # Состояние системы (чистые float-скаляры Python)
        self.pos = float(pos)
        self.vel = float(vel)
        
        # Матрица ковариации ошибок оценки P (компоненты раздельно)
        self.P00 = 1.0
        self.P01 = 0.0
        self.P10 = 0.0
        self.P11 = 1.0
        self._initialized = False

    def update(self, measurement: float, dt: Optional[float] = None) -> float:
        z = float(measurement)
        if not self._initialized:
            self.pos = z
            self.vel = 0.0
            self._initialized = True
            return self.pos

        _dt = float(dt) if dt is not None else self.dt

        # 1. Шаг Прогноза (Динамика физики: Новая_Поз = Поз + Скорость * dt)
        pos_pred = self.pos + self.vel * _dt
        vel_pred = self.vel

        # Динамическое вычисление шума процесса Q
        q00 = self.proc_noise * (_dt**3) / 3.0
        q01 = self.proc_noise * (_dt**2) / 2.0
        q11 = self.proc_noise * _dt

        # Прогноз ковариации P_pred = F*P*F^T + Q
        p00_pred = self.P00 + _dt * (self.P10 + self.P01 + _dt * self.P11) + q00
        p01_pred = self.P01 + _dt * self.P11 + q01
        p10_pred = self.P10 + _dt * self.P11 + q01
        p11_pred = self.P11 + q11

        # 2. Вычисление коэффициентов Калмана K (Инновационный анализ)
        s = p00_pred + self.meas_noise
        k0 = p00_pred / s
        k1 = p10_pred / s

        # 3. Коррекция состояния по замеру с датчика
        innovation = z - pos_pred
        self.pos = pos_pred + k0 * innovation
        self.vel = vel_pred + k1 * innovation

        # Коррекция ковариации P = (I - K*H)*P_pred
        self.P00 = (1.0 - k0) * p00_pred
        self.P01 = (1.0 - k0) * p01_pred
        self.P10 = -k1 * p00_pred + p10_pred
        self.P11 = -k1 * p01_pred + p11_pred

        return self.pos

    def get_position(self) -> float:
        return self.pos


# ===================== IIR (БИХ) =====================
try:
    from scipy.signal import butter, lfilter, lfilter_zi
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("WARNING: scipy not installed. IIR filter will be unavailable.")

class IIRFilter:
    def __init__(self, cutoff: float = 0.1, order: int = 2, initial_value: float = 0.0):
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for IIR filter")
        self.cutoff = cutoff
        self.order = order
        self.initial_value = float(initial_value)
        self.b, self.a = butter(order, cutoff, btype='low', analog=False)
        self.zi = lfilter_zi(self.b, self.a) * self.initial_value

    def update(self, raw_value: float) -> float:
        y, self.zi = lfilter(self.b, self.a, [float(raw_value)], zi=self.zi)
        return float(y[0])

    def reset(self, value: float = 0.0):
        self.initial_value = float(value)
        self.zi = lfilter_zi(self.b, self.a) * float(value)


# ===================== Векторные обёртки =====================

class VectorFilter:
    def __init__(self, alpha: float = 0.3):
        self.filters = [ExponentialSmoothing(alpha) for _ in range(3)]

    def update(self, raw_vector: List[float]) -> List[float]:
        return [self.filters[i].update(raw_vector[i]) for i in range(3)]

    def reset(self, value: Optional[List[float]] = None):
        if value is None:
            value = [0.0, 0.0, 0.0]
        for i, f in enumerate(self.filters):
            f.reset(value[i])


class VectorKalmanFilter:
    def __init__(self, dt: float = 0.02, process_noise: float = 1e-4, measurement_noise: float = 1e-2):
        self.filters = [KalmanFilter1D(dt, process_noise, measurement_noise) for _ in range(3)]

    def update(self, raw_vector: List[float], dt: Optional[float] = None) -> List[float]:
        return [self.filters[i].update(raw_vector[i], dt) for i in range(3)]

    def reset(self, value: Optional[List[float]] = None):
        if value is None:
            value = [0.0, 0.0, 0.0]
        for i, f in enumerate(self.filters):
            f.reset(value[i])


class VectorIIRFilter:
    def __init__(self, cutoff: float = 0.1, order: int = 2):
        self.filters = [IIRFilter(cutoff, order) for _ in range(3)]

    def update(self, raw_vector: List[float]) -> List[float]:
        return [self.filters[i].update(raw_vector[i]) for i in range(3)]

    def reset(self, value: Optional[List[float]] = None):
        if value is None:
            value = [0.0, 0.0, 0.0]
        for i, f in enumerate(self.filters):
            f.reset(value[i])

