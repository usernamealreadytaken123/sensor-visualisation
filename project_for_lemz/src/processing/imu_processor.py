"""
Модуль математического ядра визуально-инерциальной одометрии (VIO).
Выполняет интегрирование ускорения, преобразование кватернионов,
вычитание гравитации и обновление состояния модели.
"""
import numpy as np
from typing import Dict, Tuple

class IMUProcessor:
    """
    Класс для обработки инерциальных данных: акселерометр, гироскоп, кватернион.
    Обновляет положение, скорость и ориентацию модели.
    """

    def __init__(self, gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)):
        """
        Инициализация процессора.

        :param gravity: вектор гравитации в мировой системе координат
        """
        self.gravity = np.array(gravity, dtype=np.float64)

        # Состояние модели
        self.position = np.zeros(3, dtype=np.float64)        # (x, y, z) в метрах
        self.velocity = np.zeros(3, dtype=np.float64)        # (vx, vy, vz) в м/с
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0])    # кватернион (w, x, y, z)

        # Предыдущие значения для интегрирования
        self._prev_accel_linear = np.zeros(3, dtype=np.float64)
        self._prev_velocity = np.zeros(3, dtype=np.float64)

        # Флаг первого пакета
        self._first_packet = True

    def process(self, accel_raw: Dict[str, float], gyro_raw: Dict[str, float],
                quat: Dict[str, float], dt: float) -> Dict[str, np.ndarray]:
        """
        Обрабатывает один IMU-пакет.

        :param accel_raw: словарь с ключами 'x', 'y', 'z' — ускорение в локальной системе (м/с²)
        :param gyro_raw: словарь с ключами 'x', 'y', 'z' — угловая скорость (рад/с)
        :param quat: словарь с ключами 'w', 'x', 'y', 'z' — кватернион ориентации
        :param dt: временной интервал между пакетами (секунды)
        :return: словарь с ключами 'position', 'velocity', 'orientation'
        """
        # 1. Преобразуем входные данные в numpy-массивы
        accel_local = np.array([accel_raw['x'], accel_raw['y'], accel_raw['z']], dtype=np.float64)
        gyro_local = np.array([gyro_raw['x'], gyro_raw['y'], gyro_raw['z']], dtype=np.float64)
        quat_arr = np.array([quat['w'], quat['x'], quat['y'], quat['z']], dtype=np.float64)

        # Сохраняем текущую ориентацию (кватернион)
        self.orientation = quat_arr

        # 2. Преобразуем ускорение из локальной системы в мировую
        R = self._quaternion_to_rotation_matrix(quat_arr)
        accel_world = R @ accel_local  # матричное умножение

        # 3. Вычитаем гравитацию
        accel_linear_world = self._subtract_gravity(accel_world)

        # 4. Интегрируем ускорение для получения скорости
        if not self._first_packet:
            v_new = self._integrate_acceleration(
                self._prev_accel_linear,
                accel_linear_world,
                self.velocity,
                dt
            )
        else:
            v_new = self.velocity.copy()
            self._first_packet = False

        # 5. Интегрируем скорость для получения положения
        if not self._first_packet:
            p_new = self._integrate_velocity(
                self.velocity,
                v_new,
                self.position,
                dt
            )
        else:
            p_new = self.position.copy()

        # 6. Обновляем состояние
        self.position = p_new
        self.velocity = v_new
        self._prev_accel_linear = accel_linear_world.copy()
        self._prev_velocity = self.velocity.copy()

        # 7. Возвращаем результат
        return {
            'position': self.position,
            'velocity': self.velocity,
            'orientation': self.orientation
        }

    def reset(self) -> None:
        """Сбрасывает состояние модели в начальное (0, 0, 0) с единичной ориентацией."""
        self.position = np.zeros(3, dtype=np.float64)
        self.velocity = np.zeros(3, dtype=np.float64)
        self.orientation = np.array([1.0, 0.0, 0.0, 0.0])
        self._prev_accel_linear = np.zeros(3, dtype=np.float64)
        self._prev_velocity = np.zeros(3, dtype=np.float64)
        self._first_packet = True

    def _quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """
        Преобразует кватернион (w, x, y, z) в матрицу поворота 3x3.
        """
        w, x, y, z = q[0], q[1], q[2], q[3]
        return np.array([
            [1 - 2*y*y - 2*z*z,   2*x*y - 2*w*z,     2*x*z + 2*w*y],
            [2*x*y + 2*w*z,       1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y,       2*y*z + 2*w*x,     1 - 2*x*x - 2*y*y]
        ], dtype=np.float64)

    def _subtract_gravity(self, accel_world: np.ndarray) -> np.ndarray:
        """Вычитает вектор гравитации из ускорения в мировой системе."""
        return accel_world - self.gravity

    def _integrate_acceleration(self, a_prev: np.ndarray, a_curr: np.ndarray,
                                 v_prev: np.ndarray, dt: float) -> np.ndarray:
        """
        Интегрирует ускорение методом трапеций.

        v_new = v_prev + (a_prev + a_curr) / 2 * dt
        """
        return v_prev + (a_prev + a_curr) / 2.0 * dt

    def _integrate_velocity(self, v_prev: np.ndarray, v_curr: np.ndarray,
                             p_prev: np.ndarray, dt: float) -> np.ndarray:
        """
        Интегрирует скорость методом трапеций.

        p_new = p_prev + (v_prev + v_curr) / 2 * dt
        """
        return p_prev + (v_prev + v_curr) / 2.0 * dt

    def get_state(self) -> Dict[str, np.ndarray]:
        """Возвращает текущее состояние модели."""
        return {
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'orientation': self.orientation.copy()
        }
