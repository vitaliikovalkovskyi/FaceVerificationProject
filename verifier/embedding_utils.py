"""
verifier/embedding_utils.py

Модуль Multi-Reference Embedding для покращення точності верифікації
без донавчання моделі.

Підходи що реалізовані:
  1. embed_reference_photos()      — embedding кожного еталонного фото (один прохід)
  2. build_reference_embedding()   — усереднений вектор з N фото при реєстрації
  3. get_live_embedding()          — embedding живого фото (з align=True)
  4. compute_personal_threshold()  — адаптивний поріг під конкретну людину
  5. verify_with_embedding()       — пряме порівняння векторів через cosine
"""

import contextlib
import threading
import cv2
import numpy as np
from deepface import DeepFace
from deepface.modules import modeling, preprocessing
import deepface.detectors.RetinaFace as _df_retinaface
from retinaface import RetinaFace as _retinaface

# ── Константи ────────────────────────────────────────────────────────────────
MODEL_NAME      = "ArcFace"
DETECTOR        = "retinaface"
EMBEDDING_DIM   = 512    # ArcFace завжди 512

# ── Поріг верифікації (калібрування під FAR ≤ 1 %) ───────────────────────────
# Підібрано на половині людей mfr2+LFW з кешу бенчмарку (106 своїх / 9858 чужих пар),
# перевірено на іншій половині (89 / 9167): FAR 0.68 %, FRR 29.2 %.
# Для порівняння на тій самій перевірці: фіксований 0.68 — FAR 1.32 %, FRR 25.8 %;
# стара формула 0.68 + 0.5·intra (≤ 0.82) — FAR 9.9 %, FRR 7.9 %.
# Поправка в бік «більший розкид еталонів → м'якший поріг» на цих даних лише гіршала.
BASE_THRESHOLD  = 0.64   # глобальний поріг косинусної відстані
INTRA_REFERENCE = 0.44   # типова (медіанна) відстань між еталонними фото однієї людини
ADAPT_COEF      = -1.0   # більший розкид еталонів → суворіший поріг
ADAPT_MAX       = 0.02   # межа персональної поправки
TTA_ANGLES      = (-5, 5)


# ── RetinaFace без збільшення кадру (лише для входу) ─────────────────────────
# RetinaFace за замовчуванням збільшує кадр так, що коротша сторона = 1024 px
# (640×480 → 1365×1024), і на CPU один прохід триває ~13 с. На кадрі з
# веб-камери обличчя й так велике, тому при вході збільшення вимикаємо:
# ~4 с замість ~54 с за тієї ж точності (експеримент на mfr2, EER 0.089 проти 0.130).
# Перемикач діє лише в поточному потоці — бенчмарк і реєстрація не змінюються.
_tls = threading.local()


class _RetinaFaceProxy:
    def __getattr__(self, name):
        return getattr(_retinaface, name)

    def detect_faces(self, img_path, **kwargs):
        if getattr(_tls, "no_upscale", False):
            kwargs["allow_upscaling"] = False
        return _retinaface.detect_faces(img_path, **kwargs)


_df_retinaface.rf = _RetinaFaceProxy()


@contextlib.contextmanager
def retinaface_no_upscale():
    _tls.no_upscale = True
    try:
        yield
    finally:
        _tls.no_upscale = False


def _embed_face(face_bgr: np.ndarray) -> np.ndarray:
    """ArcFace embedding вже вирізаного й вирівняного обличчя (як усередині DeepFace.represent)."""
    model = modeling.build_model(MODEL_NAME)
    img = preprocessing.resize_image(face_bgr, (model.input_shape[1], model.input_shape[0]))
    img = preprocessing.normalize_input(img, "base")
    vec = np.array(model.forward(img), dtype=np.float32)
    return vec / np.linalg.norm(vec)


def _rotate(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)


# ── Внутрішня функція: отримати один вектор з файлу ──────────────────────────
def _get_embedding(img_path: str, enforce: bool = True) -> np.ndarray | None:
    """
    Повертає L2-нормалізований embedding або None якщо обличчя не знайдено.
    align=True — критично важливо для стабільності векторів.
    """
    try:
        result = DeepFace.represent(
            img_path=img_path,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR,
            enforce_detection=enforce,
            align=True,
        )
        vec = np.array(result[0]["embedding"], dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm == 0:
            return None
        return vec / norm
    except Exception:
        return None


# ── 1. Embedding еталонних фото (рахуються один раз) ─────────────────────────
def embed_reference_photos(photo_paths: list[str]) -> list[np.ndarray]:
    """
    Повертає embedding кожного еталонного фото, на якому знайдено обличчя.
    Результат передається і в build_reference_embedding, і в
    compute_personal_threshold, щоб не проганяти фото через мережі двічі.
    """
    embeddings = []
    for path in photo_paths:
        vec = _get_embedding(path, enforce=True)
        if vec is not None:
            embeddings.append(vec)
    return embeddings


# ── 2. Побудова еталонного вектора з кількох фото ────────────────────────────
def build_reference_embedding(embeddings: list[np.ndarray]) -> np.ndarray:
    """
    Будує усереднений L2-нормалізований embedding з кількох фото.

    Args:
        embeddings: результат embed_reference_photos (мінімум 1, оптимально 3-5)

    Returns:
        mean_embedding

    Raises:
        ValueError: якщо жодне фото не дало embedding
    """
    if not embeddings:
        raise ValueError("Жодне фото не містить виразного обличчя")

    mean_vec = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(mean_vec)
    if norm == 0:
        raise ValueError("Отриманий нульовий вектор")

    return mean_vec / norm


# ── 3. Embedding живого фото (з легкою TTA — 3 варіанти) ─────────────────────
def get_live_embedding(img_path: str) -> np.ndarray:
    """
    Отримує embedding живого фото.

    Обличчя шукається й вирівнюється RetinaFace один раз (без збільшення кадру;
    якщо так не знайдено — повторно зі збільшенням). TTA: вирізане обличчя
    в 3 варіантах (оригінал + ±5°) проходить лише через ArcFace, а
    embedding усереднюються — це компенсує нахил голови.

    Args:
        img_path: шлях до збереженого кадру з браузера

    Returns:
        L2-нормалізований embedding

    Raises:
        ValueError: якщо обличчя не знайдено
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Не вдалося прочитати файл: {img_path}")

    def detect():
        return DeepFace.extract_faces(
            img_path=img, detector_backend=DETECTOR, enforce_detection=True, align=True,
        )

    try:
        with retinaface_no_upscale():
            faces = detect()
    except ValueError:
        try:
            faces = detect()
        except ValueError:
            raise ValueError("Обличчя не знайдено на живому фото")

    # Найбільше обличчя — людина перед камерою
    best = max(faces, key=lambda f: f["facial_area"]["w"] * f["facial_area"]["h"])
    face = best["face"][:, :, ::-1]  # extract_faces повертає RGB, ArcFace у DeepFace отримує BGR

    variants = [face] + [_rotate(face, a) for a in TTA_ANGLES]
    mean_vec = np.mean([_embed_face(v) for v in variants], axis=0)
    return mean_vec / np.linalg.norm(mean_vec)


# ── 4. Адаптивний поріг під конкретну людину ─────────────────────────────────
def compute_personal_threshold(embeddings: list[np.ndarray]) -> float:
    """
    Обчислює персональний поріг верифікації на основі варіативності
    еталонних фото людини.

    Логіка: якщо еталонні фото сильно різняться між собою, їхній середній
    вектор «розмитий» і лежить ближче до чужих облич, тому поріг трохи
    суворішає; узгоджені еталони дозволяють трохи м'якший поріг.
    Поправка обмежена ±ADAPT_MAX навколо каліброваного BASE_THRESHOLD.

    Args:
        embeddings: результат embed_reference_photos

    Returns:
        float поріг в діапазоні [BASE_THRESHOLD − ADAPT_MAX, BASE_THRESHOLD + ADAPT_MAX]
    """
    if len(embeddings) < 2:
        return BASE_THRESHOLD

    # Cosine distance між всіма парами еталонів
    intra_distances = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            dist = float(1.0 - np.dot(embeddings[i], embeddings[j]))
            intra_distances.append(dist)

    mean_intra = float(np.mean(intra_distances))

    adjustment = np.clip(ADAPT_COEF * (mean_intra - INTRA_REFERENCE), -ADAPT_MAX, ADAPT_MAX)
    return float(BASE_THRESHOLD + adjustment)


# ── 5. Верифікація через cosine similarity ────────────────────────────────────
def verify_with_embedding(
    live_embedding: np.ndarray,
    ref_embedding: np.ndarray,
    threshold: float,
) -> dict:
    """
    Порівнює два L2-нормалізовані вектори через cosine distance.

    Args:
        live_embedding: вектор живого фото (з get_live_embedding)
        ref_embedding:  еталонний вектор (з БД)
        threshold:      поріг (personal_threshold або BASE_THRESHOLD)

    Returns:
        dict з ключами: verified, distance, threshold
    """
    live = np.array(live_embedding, dtype=np.float32)
    ref  = np.array(ref_embedding,  dtype=np.float32)

    # Переконуємось що нормалізовані
    live = live / (np.linalg.norm(live) + 1e-8)
    ref  = ref  / (np.linalg.norm(ref)  + 1e-8)

    cosine_dist = float(1.0 - np.dot(ref, live))
    cosine_dist = max(0.0, min(2.0, cosine_dist))  # clip до фізичного діапазону

    return {
        "verified":  cosine_dist < threshold,
        "distance":  round(cosine_dist, 4),
        "threshold": round(threshold, 4),
    }

# ── 6. Прогрів моделей при старті сервера ────────────────────────────────────
def warm_up() -> None:
    """
    Завантажує ArcFace і RetinaFace та проганяє пробні кадри обох розмірів
    (вхід — без збільшення, реєстрація — зі збільшенням), щоб перший вхід
    після запуску сервера не чекав завантаження й трасування моделей.
    """
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    _embed_face(np.zeros((112, 112, 3), dtype=np.float32))
    for no_upscale in (True, False):
        ctx = retinaface_no_upscale() if no_upscale else contextlib.nullcontext()
        with ctx:
            DeepFace.extract_faces(img_path=blank, detector_backend=DETECTOR, enforce_detection=False)
