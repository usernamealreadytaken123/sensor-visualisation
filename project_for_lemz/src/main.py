import os
import sys

# Гарантируем, что директория src/ доступна для импорта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ui.app import VOnTR_App


if __name__ == "__main__":
    app = VOnTR_App()
    try:
        app.mainloop()
    except KeyboardInterrupt:
        print("Приложение завершено.")