"""
Математическое ядро VIO с поддержкой разных фильтров.
"""
import numpy as np
from typing import Dict, Tuple, Optional
from .filters import (
    VectorFilter,
    VectorKalmanFilter,
    VectorIIRFilter
)

class IMUProcessor:
    def __init__(
        self,
        gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81),
        filter_type: str = 'exponential',  # 'exponential', 'kalman', 'iir'
        filter_alpha: float = 0.3,
        kalman_process_noise: float = 1e-4,
        kalman_measurement_noise: float = 1e-2,
        iir_cutoff: float = 0.1,
        iir_order: int = 2
    ):
        self.gravity = np.array(gravity, dtype=np.float64)

        # Сырые значения
        self.position_raw = np.zeros(3, dtype=np.float64)
        self.velocity_raw = np.zeros(3, dtype=np.float64)

        # Сглаженные значения (выход фильтра)
        self.position = np.zeros(3, dtype=np.float64)
        self.velocity = np.zeros(3, dtype=np.float64)
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0])

        # Выбираем фильтр
        self.filter_type = filter_type
        if filter_type == 'exponential':
            self._filter_position = VectorFilter(alpha=filter_alpha)
            self._filter_velocity = VectorFilter(alpha=filter_alpha)
        elif filter_type == 'kalman':
            self._filter_position = VectorKalmanFilter(
                dt=0.02,  # будет обновляться при вызове
                process_noise=kalman_process_noise,
                measurement_noise=kalman_measurement_noise
            )
            self._filter_velocity = VectorKalmanFilter(
                dt=0.02,
                process_noise=kalman_process_noise,
                measurement_noise=kalman_measurement_noise
            )
        elif filter_type == 'iir':
            self._filter_position = VectorIIRFilter(cutoff=iir_cutoff, order=iir_order)
            self._filter_velocity = VectorIIRFilter(cutoff=iir_cutoff, order=iir_order)
        else:
            raise ValueError(f"Unknown filter_type: {filter_type}")

        # Предыдущие значения для интегрирования
        self._prev_accel_linear = np.zeros(3, dtype=np.float64)
        self._prev_velocity = np.zeros(3, dtype=np.float64)
        self._first_packet = True

    def process(self, accel_raw: Dict[str, float], gyro_raw: Dict[str, float],
                quat: Dict[str, float], dt: float) -> Dict[str, np.ndarray]:

        accel_local = np.array([accel_raw['x'], accel_raw['y'], accel_raw['z']], dtype=np.float64)
        quat_arr = np.array([quat['w'], quat['x'], quat['y'], quat['z']], dtype=np.float64)

        self.orientation = quat_arr
        R = self._quaternion_to_rotation_matrix(quat_arr)
        accel_world = R @ accel_local
        accel_linear_world = self._subtract_gravity(accel_world)

        if not self._first_packet:
            v_new_raw = self._integrate_acceleration(
                self._prev_accel_linear,
                accel_linear_world,
                self.velocity_raw,
                dt
            )
        else:
            v_new_raw = self.velocity_raw.copy()
            self._first_packet = False

        if not self._first_packet:
            p_new_raw = self._integrate_velocity(
                self.velocity_raw,
                v_new_raw,
                self.position_raw,
                dt
            )
        else:
            p_new_raw = self.position_raw.copy()

        self.position_raw = p_new_raw
        self.velocity_raw = v_new_raw
        self._prev_accel_linear = accel_linear_world.copy()
        self._prev_velocity = self.velocity_raw.copy()

        # Применяем фильтр
        if self.filter_type == 'kalman':
            # Для Калмана передаём dt
            self.position = np.array(self._filter_position.update(self.position_raw.tolist(), dt))
            self.velocity = np.array(self._filter_velocity.update(self.velocity_raw.tolist(), dt))
        else:
            # Для экспоненциального и IIR (dt не нужен)
            self.position = np.array(self._filter_position.update(self.position_raw.tolist()))
            self.velocity = np.array(self._filter_velocity.update(self.velocity_raw.tolist()))

        return {
            'position': self.position,
            'velocity': self.velocity,
            'orientation': self.orientation,
            'position_raw': self.position_raw,
            'velocity_raw': self.velocity_raw
        }

    def reset(self) -> None:
        self.position_raw = np.zeros(3, dtype=np.float64)
        self.velocity_raw = np.zeros(3, dtype=np.float64)
        self.position = np.zeros(3, dtype=np.float64)
        self.velocity = np.zeros(3, dtype=np.float64)
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0])
        self._prev_accel_linear = np.zeros(3, dtype=np.float64)
        self._prev_velocity = np.zeros(3, dtype=np.float64)
        self._first_packet = True

        self._filter_position.reset([0.0, 0.0, 0.0])
        self._filter_velocity.reset([0.0, 0.0, 0.0])

    def _quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        w, x, y, z = q[0], q[1], q[2], q[3]
        return np.array([
            [1 - 2*y*y - 2*z*z,   2*x*y - 2*w*z,     2*x*z + 2*w*y],
            [2*x*y + 2*w*z,       1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y,       2*y*z + 2*w*x,     1 - 2*x*x - 2*y*y]
        ], dtype=np.float64)

    def _subtract_gravity(self, accel_world: np.ndarray) -> np.ndarray:
        return accel_world - self.gravity

    def _integrate_acceleration(self, a_prev: np.ndarray, a_curr: np.ndarray,
                                 v_prev: np.ndarray, dt: float) -> np.ndarray:
        return v_prev + (a_prev + a_curr) / 2.0 * dt

    def _integrate_velocity(self, v_prev: np.ndarray, v_curr: np.ndarray,
                             p_prev: np.ndarray, dt: float) -> np.ndarray:
        return p_prev + (v_prev + v_curr) / 2.0 * dt

    def get_state(self) -> Dict[str, np.ndarray]:
        return {
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'orientation': self.orientation.copy()
        }

    def get_raw_state(self) -> Dict[str, np.ndarray]:
        return {
            'position': self.position_raw.copy(),
            'velocity': self.velocity_raw.copy(),
            'orientation': self.orientation.copy()
        }