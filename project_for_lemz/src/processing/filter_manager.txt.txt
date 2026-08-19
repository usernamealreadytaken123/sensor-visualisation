"""
Менеджер фильтров для одновременного применения нескольких фильтров к одним и тем же данным.
Поддерживаемые фильтры:
- RAW (без фильтрации)
- Exponential (экспоненциальное сглаживание)
- Kalman (фильтр Калмана)
- IIR (БИХ-фильтр Баттерворта 2-го порядка)
"""
import numpy as np
from typing import Dict, List, Optional
from .filters import (
    VectorFilter,
    VectorKalmanFilter,
    VectorIIRFilter
)


class FilterManager:
    """
    Управляет несколькими фильтрами одновременно.
    Каждый фильтр применяется к одним и тем же сырым данным независимо.
    """

    def __init__(
        self,
        filter_alpha: float = 0.3,
        kalman_process_noise: float = 1e-4,
        kalman_measurement_noise: float = 1e-2,
        iir_cutoff: float = 0.1,
        iir_order: int = 2
    ):
        """
        Инициализация менеджера фильтров.

        :param filter_alpha: коэффициент сглаживания для экспоненциального фильтра
        :param kalman_process_noise: шум процесса для фильтра Калмана
        :param kalman_measurement_noise: шум измерения для фильтра Калмана
        :param iir_cutoff: частота среза для IIR-фильтра (0.0–0.5)
        :param iir_order: порядок IIR-фильтра
        """
        # Параметры фильтров
        self.filter_alpha = filter_alpha
        self.kalman_process_noise = kalman_process_noise
        self.kalman_measurement_noise = kalman_measurement_noise
        self.iir_cutoff = iir_cutoff
        self.iir_order = iir_order

        # Словарь фильтров
        self._filters: Dict[str, object] = {}

        # Словарь для хранения последних результатов каждого фильтра
        self._last_results: Dict[str, List[float]] = {}

        # Счётчик обработанных пакетов (для оси X графика)
        self.packet_count = 0

        # Инициализация фильтров
        self._init_filters()

    def _init_filters(self):
        """Создаёт все фильтры с текущими параметрами."""
        self._filters = {
            'raw': None,  # специальный случай — без фильтрации
            'exponential': VectorFilter(alpha=self.filter_alpha),
            'kalman': VectorKalmanFilter(
                dt=0.02,
                process_noise=self.kalman_process_noise,
                measurement_noise=self.kalman_measurement_noise
            ),
            'iir': VectorIIRFilter(cutoff=self.iir_cutoff, order=self.iir_order)
        }

        # Инициализация результатов
        self._last_results = {
            'raw': [0.0, 0.0, 0.0],
            'exponential': [0.0, 0.0, 0.0],
            'kalman': [0.0, 0.0, 0.0],
            'iir': [0.0, 0.0, 0.0]
        }

    def process(self, position_raw: List[float], dt: float) -> Dict[str, List[float]]:
        """
        Применяет все фильтры к сырым данным и возвращает результаты.

        :param position_raw: сырые координаты [x, y, z]
        :param dt: временной шаг (для Калмана)
        :return: словарь {имя_фильтра: [x, y, z]}
        """
        # Сырые данные (без фильтрации)
        self._last_results['raw'] = position_raw.copy()

        # Экспоненциальный фильтр
        self._last_results['exponential'] = self._filters['exponential'].update(position_raw)

        # Фильтр Калмана (передаём dt)
        self._last_results['kalman'] = self._filters['kalman'].update(position_raw, dt)

        # IIR-фильтр
        self._last_results['iir'] = self._filters['iir'].update(position_raw)

        # Увеличиваем счётчик пакетов
        self.packet_count += 1

        return self._last_results

    def reset(self) -> None:
        """Сбрасывает состояние всех фильтров и счётчик пакетов."""
        for filter_name, filter_obj in self._filters.items():
            if filter_obj is not None:
                filter_obj.reset([0.0, 0.0, 0.0])

        # Сбрасываем последние результаты
        for key in self._last_results:
            self._last_results[key] = [0.0, 0.0, 0.0]

        self.packet_count = 0

    def get_results(self) -> Dict[str, List[float]]:
        """Возвращает последние результаты всех фильтров."""
        return self._last_results

    def get_filter_names(self) -> List[str]:
        """Возвращает список доступных фильтров."""
        return list(self._filters.keys())

    def get_packet_count(self) -> int:
        """Возвращает количество обработанных пакетов."""
        return self.packet_count