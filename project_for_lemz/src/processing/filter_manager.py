# Менеджер фильтров для одновременного применения нескольких фильтров к одним и тем же данным.
# Поддерживаемые фильтры:
# - RAW (без фильтрации)
# - Exponential (экспоненциальное сглаживание)
# - Kalman (фильтр Калмана)
#- IIR (БИХ-фильтр Баттерворта 2-го порядка)

import numpy as np
import logging

# Импортируем полноценные векторные обертки из вашего файла filters.py
from processing.filters import VectorFilter, VectorIIRFilter, VectorKalmanFilter

class FilterManager:
    def __init__(self):
        """
        Менеджер каскада фильтрации. Применяет все доступные алгоритмы 
        к одним и тем же сырым входным навигационным координатам.
        """
        # Используем ваши классы из filters.py с оптимальными параметрами
        self.exp_filter = VectorFilter(alpha=0.12)
        self.iir_filter = VectorIIRFilter(cutoff=0.1, order=2)
        
        # Полноценный фильтр Калмана: оценивает координату И скорость [pos, vel]
        self.kalman_filter = VectorKalmanFilter(dt=0.02, process_noise=1e-4, measurement_noise=1e-2)
        
        logging.info("Менеджер фильтрации (FilterManager) успешно инициализирован.")

    def reset(self):
        """Полный сброс предыстории всех внутренних фильтров."""
        self.exp_filter.reset()
        self.iir_filter.reset()
        self.kalman_filter.reset()
        logging.info("Состояние всех алгоритмов фильтрации сброшено.")

    def apply_filters(self, raw_data: dict) -> dict:
        """
        Принимает сырые данные из IMUProcessor и параллельно накладывает сглаживание,
        гарантируя строгое приведение типов для графиков Matplotlib и Excel.
        """
        # Преобразуем входные координаты в список float, как требуют ваши фильтры в filters.py
        raw_position_list = [float(x) for x in raw_data["position"]]
        dt = raw_data.get("dt", 0.02)

        # 1. Расчет базовых фильтров (возвращают List[float])
        filtered_exp = self.exp_filter.update(raw_position_list)
        filtered_iir = self.iir_filter.update(raw_position_list)
        
        # 2. Расчет Калмана с защитой от стартового дрейфа скорости
        if dt <= 0.001 or np.linalg.norm(raw_data["position"]) == 0.0:
            filtered_kalman = raw_position_list
            self.kalman_filter.reset(raw_position_list)
        else:
            filtered_kalman = self.kalman_filter.update(raw_position_list, dt=dt)

        # КРИТИЧЕСКИЙ ФИКС: Принудительно конвертируем абсолютно ВСЕ выходные векторы 
        # в однородные одномерные numpy-массивы типа float64 фиксированной формы (3,)
        return {
            "timestamp": raw_data.get("timestamp"),
            "quat": np.asarray(raw_data["quat"], dtype=np.float64),
            "RAW": np.asarray(raw_data["position"], dtype=np.float64), 
            "EXP": np.asarray(filtered_exp, dtype=np.float64),
            "IIR": np.asarray(filtered_iir, dtype=np.float64),
            "KALMAN": np.asarray(filtered_kalman, dtype=np.float64)
        }
