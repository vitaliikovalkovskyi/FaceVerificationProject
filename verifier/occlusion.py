"""
verifier/occlusion.py

Виявлення маски (оклюзії нижньої половини обличчя) за ArcFace embedding
і динамічний поріг верифікації.

Маска сильно змінює embedding, тому достатньо лінійного класифікатора
(логістична регресія) поверх уже обчисленого вектора — без окремої моделі
і без додаткового часу. Ваги навчає команда `manage.py train_mask_classifier`
на фото з масками та без із mfr2 і LFW і зберігає в mask_classifier.npz:
  - "all"          — для реальної роботи системи;
  - "no_lfw"       — без LFW, для чесного бенчмарку на LFW;
  - "mfr2_fold0/1" — без людей mfr2 з відповідної половини, для чесного бенчмарку на mfr2
                     (фото кожної людини оцінює модель, яка її не бачила).

Динамічний поріг: якщо виявлено маску, поріг верифікації піднімається до
MASK_THRESHOLD, інакше діє персональний (~0.64). При вході 3 невдалі спроби
біометрією ведуть до OTP.
"""
import hashlib
import os

import numpy as np

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "mask_classifier.npz")

# Поріг для облич у масці. mfr2 (99 своїх / 4851 чужих пар у масці, кожен тест проти всіх еталонів):
#   0.64 — FRR 61.6 %, FAR 0.25 %;  0.75 — FRR 32.3 %, FAR 3.4 %;  0.82 — FRR 19.2 %, FAR 12 %.
# 0.82 близький до порогів EER на масках (mfr2 0.84–0.87).
MASK_THRESHOLD = 0.82
MASK_PROBABILITY = 0.5

_models = None


def _load() -> dict:
    global _models
    if _models is None:
        data = np.load(WEIGHTS_PATH)
        _models = {name[2:]: (data[name], float(data["b_" + name[2:]])) for name in data.files if name.startswith("w_")}
    return _models


def mfr2_fold(person_id: str) -> int:
    """Стабільний поділ людей mfr2 на дві половини (для чесної оцінки класифікатора)."""
    return int(hashlib.md5(person_id.encode()).hexdigest(), 16) % 2


def benchmark_model(dataset: str, test_person: str) -> str:
    """Модель, яка не бачила ні датасету LFW, ні (для mfr2) людину з тестового фото."""
    return "no_lfw" if dataset == "lfw" else f"mfr2_fold{mfr2_fold(test_person)}"


def available() -> bool:
    return os.path.exists(WEIGHTS_PATH)


def mask_probability(embedding: np.ndarray, model: str = "all") -> float:
    """Ймовірність того, що обличчя в масці (embedding — L2-нормалізований ArcFace)."""
    w, b = _load()[model]
    return float(1.0 / (1.0 + np.exp(-(np.dot(w, embedding) + b))))


def is_masked(embedding: np.ndarray, model: str = "all") -> bool:
    return mask_probability(embedding, model) >= MASK_PROBABILITY


def train_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1e-2, iterations: int = 3000, lr: float = 0.5):
    """Логістична регресія з балансуванням класів (масок значно менше, ніж чистих облич)."""
    w, b = np.zeros(X.shape[1]), 0.0
    sample_w = np.where(y, (len(y) - y.sum()) / max(y.sum(), 1), 1.0)
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-(X @ w + b)))
        g = (p - y) * sample_w
        w -= lr * (X.T @ g / len(y) + l2 * w)
        b -= lr * g.mean()
    return w, b
