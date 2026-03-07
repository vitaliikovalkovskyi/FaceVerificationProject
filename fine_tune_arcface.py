import os
# ФІКС для помилки 'KerasHistory' - має бути ПЕРЕД імпортом tensorflow/deepface
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import numpy as np
import tensorflow as tf
from deepface import DeepFace

# 10 людей для навчання за твоїми маркерами
TRAIN_MAP = {
    "AdrianDunbar": [2, 4],
    "AhmadAli": [2, 5],
    "ArifAlvi": [3, 4],
    "BellaHadid": [2],
    "BenAffleck": [3, 7],
    "BrianKemp": [1, 5],
    "BrodyJenner": [1],
    "CarrieLam": [4, 6],
    "CharlieBaker": [1, 2],
    "DennyTamaki": [7, 8]
}

def fine_tune():
    print("[SYS] Ініціалізація моделі ArcFace...")
    try:
        # Завантажуємо базову модель
        model_wrapper = DeepFace.build_model("ArcFace")
        base_model = model_wrapper.model
    except Exception as e:
        print(f"[ERROR] Не вдалося побудувати модель: {e}")
        return

    # Заморожуємо шари, крім останніх 4
    for layer in base_model.layers[:-4]:
        layer.trainable = False

    base_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
        loss='cosine_similarity'
    )

    x_ref, x_mask = [], []
    base_path = "mfr2" # Папка датасету

    print("[SYS] Підготовка зображень для 10 осіб...")
    for person, no_mask_list in TRAIN_MAP.items():
        p_path = os.path.join(base_path, person)
        if not os.path.exists(p_path):
            print(f"[WARN] Папка {person} не знайдена за шляхом {p_path}")
            continue
        
        files = os.listdir(p_path)
        # Знаходимо еталон (no-mask) за твоїми індексами
        ref_files = [f for f in files if int(f.split('_')[-1].split('.')[0]) in no_mask_list]
        # Всі інші вважаємо масками
        mask_files = [f for f in files if int(f.split('_')[-1].split('.')[0]) not in no_mask_list]

        if ref_files and mask_files:
            try:
                # Екстракція та підготовка
                img1 = DeepFace.extract_faces(os.path.join(p_path, ref_files[0]), detector_backend='retinaface', enforce_detection=False)[0]['face']
                img2 = DeepFace.extract_faces(os.path.join(p_path, mask_files[0]), detector_backend='retinaface', enforce_detection=False)[0]['face']
                
                x_ref.append(tf.image.resize(img1, (112, 112)))
                x_mask.append(tf.image.resize(img2, (112, 112)))
            except Exception as e:
                print(f"[SKIP] Помилка обробки {person}: {e}")
                continue

    if not x_ref:
        print("[ERROR] Немає даних для навчання. Перевір шляхи до папки mfr2.")
        return

    print(f"[TRAIN] Початок донавчання на {len(x_ref)} парах...")
    # Навчаємо модель зближувати вектори маски з еталоном
    base_model.fit(
        np.array(x_mask), 
        base_model.predict(np.array(x_ref)), 
        epochs=10, 
        batch_size=2,
        verbose=1
    )

    # Збереження результату
    base_model.save_weights("arcface_finetuned.h5")
    print("[SUCCESS] Нові ваги збережено у файл arcface_finetuned.h5")

if __name__ == "__main__":
    fine_tune()