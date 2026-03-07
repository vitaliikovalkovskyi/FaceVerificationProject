# verifier/utils.py
import os
import json
import random
from deepface import DeepFace
from django.conf import settings

# Твій словник маркерів
MFR2_NO_MASK_INDICES = {
    "AdrianDunbar": [2, 4], "AhmadAli": [2, 5], "ArifAlvi": [3, 4], "BellaHadid": [2],
    "BenAffleck": [3, 7], "BrianKemp": [1, 5], "BrodyJenner": [1], "CarrieLam": [4, 6],
    "CharlieBaker": [1, 2], "DennyTamaki": [7, 8], "DonnellRawlings": [1], "DougDucey": [2, 5],
    "EmmanuelMacron": [1, 4], "EricGarcetti": [2, 5], "FloydMayweather": [1, 3], "GaryHerbert": [7, 8],
    "GavinNewsom": [5, 6], "GretchenWhitmer": [1, 5], "IgorMatovic": [4, 5], "ImranKhan": [1, 5],
    "JaredPolis": [9, 10], "JoeBiden": [2, 5], "JonathanBennet": [2, 3], "JustinTrudeau": [3, 6],
    "KimKardashian": [2], "KyriakosMitsotakis": [3, 5], "LadyGaga": [4, 5], "LarryHogan": [5, 7],
    "MatthewMorrison": [2, 3], "MikeDewine": [3, 5], "MitchMcConnell": [2, 6], "MuhammadFahmi": [1, 4],
    "NancyPelosi": [4], "PharrellWilliams": [3, 4], "PhilMurphy": [4, 6], "RalphNortham": [2, 5],
    "ReneeMorrison": [2, 3], "RonDesantis": [3, 6], "RoyCooper": [1, 6], "SimonCowell": [3, 4],
    "SteveHarvey": [2], "StevenReed": [5, 6], "TakeOff": [2, 3], "TaroKono": [2],
    "TomWolf": [4], "TsaiIngwen": [1, 6], "UddhavThackery": [2, 3], "VassilisKikilias": [1, 3],
    "XiJinping": [2, 5], "YoshihideSuga": [5, 6], "YurikoKoike": [6, 7], "ZacEfron": [1, 3],
    "ZuzanaCaputova": [5, 7]
}

def run_accuracy_generator(limit=50, threshold=0.75, dataset='afdb'):
    # 1. Ініціалізація моделі
    model_wrapper = DeepFace.build_model("ArcFace")
    
    # 2. Завантаження донавчених ваг, якщо вони існують
    weights_path = os.path.join(settings.BASE_DIR, 'arcface_finetuned.h5')
    is_finetuned = False
    
    if os.path.exists(weights_path):
        try:
            model_wrapper.model.load_weights(weights_path)
            is_finetuned = True
        except Exception as e:
            print(f"Error loading weights: {e}")

    # 3. Підготовка пар для тестування (MFR2)
    base_path = os.path.join(settings.BASE_DIR, 'mfr2')
    all_test_pairs = []
    people = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    for person_id in people:
        p_path = os.path.join(base_path, person_id)
        files = [f for f in os.listdir(p_path) if f.lower().endswith(('.png', '.jpg'))]
        no_mask_list = MFR2_NO_MASK_INDICES.get(person_id, [])
        unmasked = [f for f in files if int(f.split('_')[-1].split('.')[0]) in no_mask_list]
        masked = [f for f in files if int(f.split('_')[-1].split('.')[0]) not in no_mask_list]
        
        for m_file in masked:
            if unmasked:
                all_test_pairs.append({
                    'id': person_id,
                    'ref': os.path.join(p_path, random.choice(unmasked)),
                    'test': os.path.join(p_path, m_file)
                })

    random.shuffle(all_test_pairs)
    test_batch = all_test_pairs[:limit]
    total, correct = len(test_batch), 0

    # Початкове повідомлення про статус моделі
    status_msg = "ВИКОРИСТОВУЄТЬСЯ ДОНАВЧЕНА МОДЕЛЬ" if is_finetuned else "СТАНДАРТНА МОДЕЛЬ"
    yield f"data: {json.dumps({'status': 'info', 'msg': status_msg})}\n\n"

    # 4. Процес верифікації
    for index, pair in enumerate(test_batch):
        try:
            res = DeepFace.verify(
                img1_path=pair['ref'], img2_path=pair['test'], 
                model_name='ArcFace', detector_backend='retinaface', enforce_detection=False
            )
            
            dist = res['distance']
            is_match = dist <= float(threshold)
            if is_match: correct += 1
            
            # Відносні шляхи для відображення фото
            rel_ref = "mfr2/" + pair['ref'].split('mfr2' + os.sep)[-1].replace(os.sep, '/')
            rel_test = "mfr2/" + pair['test'].split('mfr2' + os.sep)[-1].replace(os.sep, '/')

            progress_data = {
                'progress': int(((index + 1) / total) * 100),
                'person': pair['id'],
                'result': str(is_match).upper(),
                'dist': round(dist, 4),
                'img_ref': rel_ref,
                'img_test': rel_test,
                'status': 'processing'
            }
            yield f"data: {json.dumps(progress_data)}\n\n"
        except: continue

    accuracy = (correct / total * 100) if total > 0 else 0
    yield f"data: {json.dumps({'progress': 100, 'accuracy': f'{accuracy:.2f}%', 'status': 'completed'})}\n\n"