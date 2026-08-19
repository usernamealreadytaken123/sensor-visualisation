"""
Модуль фильтрации данных.
Поддерживаемые фильтры:
- Exponential (экспоненциальное сглаживание)
- Kalman (одномерный фильтр Калмана для позиции и скорости)
- IIR (БИХ-фильтр Баттерворта 2-го порядка) – требует scipy
"""
import numpy as np
from typing import List, Optional

# ===================== Базовые фильтры =====================

class ExponentialSmoothing:
    """Экспоненциальное сглаживание (первый порядок)"""
    def __init__(self, alpha: float = 0.3, initial_value: float = 0.0):
        self.alpha = alpha
        self.value = initial_value
        self._initialized = False

    def update(self, raw_value: float) -> float:
        if not self._initialized:
            self.value = raw_value
            self._initialized = True
        else:
            self.value = self.alpha * raw_value + (1.0 - self.alpha) * self.value
        return self.value

    def reset(self, value: float = 0.0):
        self.value = value
        self._initialized = False

    def get(self) -> float:
        return self.value


class KalmanFilter1D:
    """
    Одномерный фильтр Калмана для оценки положения и скорости.
    Состояние: [position, velocity]^T
    Измерение: position
    """
    def __init__(self, dt: float = 0.02, process_noise: float = 1e-4,
                 measurement_noise: float = 1e-2, initial_pos: float = 0.0,
                 initial_vel: float = 0.0):
        self.dt = dt
        self.Q = np.array([[process_noise * dt**3 / 3, process_noise * dt**2 / 2],
                           [process_noise * dt**2 / 2, process_noise * dt]], dtype=np.float64)
        self.R = measurement_noise
        self.H = np.array([[1.0, 0.0]], dtype=np.float64)
        self.F = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)

        self.x = np.array([initial_pos, initial_vel], dtype=np.float64)
        self.P = np.eye(2, dtype=np.float64) * 1.0
        self._initialized = False

    def update(self, measurement: float, dt: Optional[float] = None) -> float:
        if not self._initialized:
            self.x = np.array([measurement, 0.0], dtype=np.float64)
            self._initialized = True
            return measurement

        if dt is not None and dt != self.dt:
            self.dt = dt
            self.F[0, 1] = dt
            # Обновляем Q с учётом нового dt (приближённо)
            scale = dt / 0.02
            self.Q = np.array([[self.Q[0,0] * scale**3, self.Q[0,1] * scale**2],
                               [self.Q[1,0] * scale**2, self.Q[1,1] * scale]], dtype=np.float64)

        # Прогноз
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # Обновление
        K = P_pred @ self.H.T / (self.H @ P_pred @ self.H.T + self.R)
        self.x = x_pred + K * (measurement - self.H @ x_pred)
        self.P = (np.eye(2) - K @ self.H) @ P_pred

        return self.x[0]

    def get_position(self) -> float:
        return self.x[0]

    def get_velocity(self) -> float:
        return self.x[1]

    def reset(self, pos: float = 0.0, vel: float = 0.0):
        self.x = np.array([pos, vel], dtype=np.float64)
        self.P = np.eye(2, dtype=np.float64) * 1.0
        self._initialized = False


# ===================== IIR (БИХ) =====================
try:
    from scipy.signal import butter, lfilter, lfilter_zi
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("WARNING: scipy not installed. IIR filter will be unavailable.")

class IIRFilter:
    """БИХ-фильтр низких частот (Баттерворт 2-го порядка). Требует scipy."""
    def __init__(self, cutoff: float = 0.1, order: int = 2, initial_value: float = 0.0):
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for IIR filter")
        self.cutoff = cutoff
        self.order = order
        self.initial_value = initial_value
        self.b, self.a = butter(order, cutoff, btype='low', analog=False)
        self.zi = lfilter_zi(self.b, self.a) * initial_value

    def update(self, raw_value: float) -> float:
        y, self.zi = lfilter(self.b, self.a, [raw_value], zi=self.zi)
        return y[0]

    def reset(self, value: float = 0.0):
        self.initial_value = value
        self.zi = lfilter_zi(self.b, self.a) * value


# ===================== Векторные обёртки =====================

class VectorFilter:
    """Обёртка для экспоненциального сглаживания вектора (3 оси)."""
    def __init__(self, alpha: float = 0.3):
        self.filters = [ExponentialSmoothing(alpha) for _ in range(3)]

    def update(self, raw_vector: List[float]) -> List[float]:
        return [self.filters[i].update(raw_vector[i]) for i in range(3)]

    def reset(self, value: Optional[List[float]] = None):
        if value is None:
            value = [0.0, 0.0, 0.0]
        for i, f in enumerate(self.filters):
            f.reset(value[i])

    def get(self) -> List[float]:
        return [f.get() for f in self.filters]


class VectorKalmanFilter:
    """Обёртка для фильтра Калмана (по одному на ось)."""
    def __init__(self, dt: float = 0.02, process_noise: float = 1e-4,
                 measurement_noise: float = 1e-2):
        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.filters = [
            KalmanFilter1D(dt, process_noise, measurement_noise) for _ in range(3)
        ]

    def update(self, raw_vector: List[float], dt: Optional[float] = None) -> List[float]:
        if dt is not None:
            self.dt = dt
        return [self.filters[i].update(raw_vector[i], dt) for i in range(3)]

    def reset(self, value: Optional[List[float]] = None):
        if value is None:
            value = [0.0, 0.0, 0.0]
        for i, f in enumerate(self.filters):
            f.reset(value[i])

    def get(self) -> List[float]:
        return [f.get_position() for f in self.filters]


class VectorIIRFilter:
    """Обёртка для IIR-фильтра (по одному на ось). Требует scipy."""
    def __init__(self, cutoff: float = 0.1, order: int = 2):
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for IIR filter")
        self.cutoff = cutoff
        self.order = order
        self.filters = [IIRFilter(cutoff, order) for _ in range(3)]

    def update(self, raw_vector: List[float]) -> List[float]:
        return [self.filters[i].update(raw_vector[i]) for i in range(3)]

    def reset(self, value: Optional[List[float]] = None):
        if value is None:
            value = [0.0, 0.0, 0.0]
        for i, f in enumerate(self.filters):
            f.reset(value[i])