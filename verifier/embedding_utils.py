"""
verifier/embedding_utils.py

Модуль Multi-Reference Embedding для покращення точності верифікації
без донавчання моделі.

Підходи що реалізовані:
  1. build_reference_embedding()  — усереднений вектор з N фото при реєстрації
  2. get_live_embedding()          — embedding живого фото (з align=True)
  3. compute_personal_threshold()  — адаптивний поріг під конкретну людину
  4. verify_with_embedding()       — пряме порівняння векторів через cosine
"""

import os
import cv2
import numpy as np
from deepface import DeepFace

# ── Константи ────────────────────────────────────────────────────────────────
MODEL_NAME      = "ArcFace"
DETECTOR        = "retinaface"
BASE_THRESHOLD  = 0.68   # м'якший за дефолтний 0.75 — краще для масок
MAX_THRESHOLD   = 0.82   # верхня межа безпеки
EMBEDDING_DIM   = 512    # ArcFace завжди 512


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


# ── 1. Побудова еталонного вектора з кількох фото ────────────────────────────
def build_reference_embedding(photo_paths: list[str]) -> tuple[np.ndarray, int]:
    """
    Будує усереднений L2-нормалізований embedding з кількох фото.

    Args:
        photo_paths: список шляхів до фото (мінімум 1, оптимально 3-5)

    Returns:
        (mean_embedding, кількість успішно оброблених фото)

    Raises:
        ValueError: якщо жодне фото не дало embedding
    """
    embeddings = []
    for path in photo_paths:
        vec = _get_embedding(path, enforce=True)
        if vec is not None:
            embeddings.append(vec)

    if not embeddings:
        raise ValueError("Жодне фото не містить виразного обличчя")

    mean_vec = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(mean_vec)
    if norm == 0:
        raise ValueError("Отриманий нульовий вектор")

    return mean_vec / norm, len(embeddings)


# ── 2. Embedding живого фото (з легкою TTA — 3 варіанти) ─────────────────────
def get_live_embedding(img_path: str) -> np.ndarray:
    """
    Отримує embedding живого фото.
    Робить 3 варіанти (оригінал + ±5°) і усереднює — компенсує кут голови.

    Args:
        img_path: шлях до збереженого кадру з браузера

    Returns:
        L2-нормалізований embedding

    Raises:
        ValueError: якщо обличчя не знайдено в жодному варіанті
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Не вдалося прочитати файл: {img_path}")

    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    variants = []

    # Оригінал
    variants.append(("orig", img))

    # Легкі повороти ±5° — компенсують нахил голови при зйомці з браузера
    for angle in [-5, 5]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
        variants.append((f"rot{angle}", rotated))

    embeddings = []
    tmp_paths = []

    for name, variant_img in variants:
        tmp_path = img_path.replace(".jpg", f"_tta_{name}.jpg")
        cv2.imwrite(tmp_path, variant_img)
        tmp_paths.append(tmp_path)

        vec = _get_embedding(tmp_path, enforce=False)
        if vec is not None:
            embeddings.append(vec)

    # Чистимо тимчасові файли
    for p in tmp_paths:
        if os.path.exists(p):
            os.remove(p)

    if not embeddings:
        raise ValueError("Обличчя не знайдено на живому фото")

    mean_vec = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(mean_vec)
    return mean_vec / norm


# ── 3. Адаптивний поріг під конкретну людину ─────────────────────────────────
def compute_personal_threshold(photo_paths: list[str]) -> float:
    """
    Обчислює персональний поріг верифікації на основі варіативності
    еталонних фото людини.

    Логіка: якщо між еталонними фото вже є природна варіація (різне
    освітлення, кут), поріг трохи збільшується — бо і жива верифікація
    матиме таку ж варіацію.

    Args:
        photo_paths: список шляхів до еталонних фото

    Returns:
        float поріг в діапазоні [BASE_THRESHOLD, MAX_THRESHOLD]
    """
    if len(photo_paths) < 2:
        return BASE_THRESHOLD

    embeddings = []
    for path in photo_paths:
        vec = _get_embedding(path, enforce=True)
        if vec is not None:
            embeddings.append(vec)

    if len(embeddings) < 2:
        return BASE_THRESHOLD

    # Cosine distance між всіма парами еталонів
    intra_distances = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            dist = float(1.0 - np.dot(embeddings[i], embeddings[j]))
            intra_distances.append(dist)

    mean_intra = np.mean(intra_distances)

    # Адаптивний поріг: base + половина внутрішньокласової варіації
    adaptive = BASE_THRESHOLD + (mean_intra * 0.5)
    return float(min(adaptive, MAX_THRESHOLD))


# ── 4. Верифікація через cosine similarity ────────────────────────────────────
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