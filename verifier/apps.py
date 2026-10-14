import os
import sys
import threading

from django.apps import AppConfig


class VerifierConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'verifier'

    def ready(self):
        from .win_paths import patch_cv2_data_path
        patch_cv2_data_path()

        # Прогрів моделей лише в робочому процесі runserver (не в autoreloader і не для migrate/check)
        is_server = 'runserver' in sys.argv and (os.environ.get('RUN_MAIN') == 'true' or '--noreload' in sys.argv)
        if is_server:
            threading.Thread(target=self._warm_up, name='model-warm-up', daemon=True).start()

    @staticmethod
    def _warm_up():
        import time
        t0 = time.perf_counter()
        try:
            from .embedding_utils import warm_up
            from .liveness_core import analyze_frame
            import cv2
            import numpy as np
            warm_up()
            analyze_frame(cv2.imencode('.jpg', np.zeros((240, 320, 3), np.uint8))[1].tobytes())
            print(f"[warm-up] Моделі готові за {time.perf_counter() - t0:.0f} с", flush=True)
        except Exception as e:
            print(f"[warm-up] Не вдалося прогріти моделі: {e}", flush=True)
