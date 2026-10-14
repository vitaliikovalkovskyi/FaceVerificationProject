"""
Ядро детекції для вдосконаленої перевірки живості (перенесено з liveness-lab/liveness_core.py).

analyze_frame(jpeg_bytes) -> dict повертає ознаки одного кадру; логіка
процедури (калібрування, завдання, таймаути) виконується на клієнті в login.html.

MediaPipe ініціалізується ліниво: якщо бібліотеку не встановлено, Django
стартує як зазвичай, а endpoint повідомляє про недоступність — клієнт
тоді переходить на базову процедуру.
"""
import math
import threading

import cv2
import numpy as np

from .win_paths import ascii_path

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
NOSE_TIP = 1
UPPER_LID, LOWER_LID = 159, 145  # точки, які використовує базова процедура
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
    379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
    234, 127, 162, 21, 54, 103, 67, 109,
]

_face_mesh = None
_lock = threading.Lock()  # FaceMesh не потокобезпечний, а dev-сервер Django багатопотоковий


class LivenessUnavailable(Exception):
    pass


def _use_ascii_mediapipe_root():
    """
    C++-частина MediaPipe на Windows не відкриває файли графа й моделей, якщо
    шлях містить не-ASCII символи (напр. «Магістратура»). Корінь пакета
    обчислюється з solution_base.__file__, тому підставляємо короткий 8.3-шлях.
    """
    from mediapipe.python import solution_base
    solution_base.__file__ = ascii_path(solution_base.__file__)
    if not solution_base.__file__.isascii():
        raise LivenessUnavailable("шлях до mediapipe містить не-ASCII символи")


def _get_face_mesh():
    global _face_mesh
    if _face_mesh is None:
        try:
            import mediapipe as mp
        except ImportError as e:
            raise LivenessUnavailable("mediapipe не встановлено") from e
        _use_ascii_mediapipe_root()
        try:
            _face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Exception as e:
            raise LivenessUnavailable(f"не вдалося ініціалізувати FaceMesh: {e}") from e
    return _face_mesh


def _ear(pts):
    d = lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1])
    h = d(pts[0], pts[3])
    return (d(pts[1], pts[5]) + d(pts[2], pts[4])) / (2.0 * h) if h else 0.0


def analyze_frame(jpeg_bytes):
    frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return {"faceFound": False, "error": "decode"}
    h, w = frame.shape[:2]
    with _lock:
        res = _get_face_mesh().process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return {"faceFound": False}
    lm = res.multi_face_landmarks[0].landmark
    px = lambda i: (lm[i].x * w, lm[i].y * h)

    # --- EAR (піксельні координати) + відносне положення носа в контурі обличчя
    ear = (_ear([px(i) for i in LEFT_EYE]) + _ear([px(i) for i in RIGHT_EYE])) / 2.0
    xs = [px(i)[0] for i in FACE_OVAL]
    lo, hi = min(xs), max(xs)
    nose_ratio = (px(NOSE_TIP)[0] - lo) / (hi - lo) if hi > lo else 0.5

    # --- геометрія для перевірки безперервності обличчя (нормалізована шириною кадру)
    eye_dist = math.hypot(px(33)[0] - px(263)[0], px(33)[1] - px(263)[1]) / w
    face_width = (hi - lo) / w
    face_cx = ((hi + lo) / 2.0) / w

    # --- ознаки базової процедури: нормалізовані координати
    eye_gap = abs(lm[UPPER_LID].y - lm[LOWER_LID].y)
    nose_x_raw = lm[NOSE_TIP].x

    return {
        "faceFound": True,
        "ear": round(ear, 4),
        "noseRatio": round(nose_ratio, 4),
        "eyeGap": round(eye_gap, 5),
        "noseXRaw": round(nose_x_raw, 4),
        "eyeDist": round(eye_dist, 4),
        "faceWidth": round(face_width, 4),
        "faceCx": round(face_cx, 4),
    }
