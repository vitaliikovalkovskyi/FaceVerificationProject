import cv2
from deepface import DeepFace
from django.http import JsonResponse
from django.shortcuts import render
from django.conf import settings
from django.templatetags.static import static
import os

def capture_image(file_name="captured_image.jpg"):
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        return None
    cap.release()
    file_path = f"{settings.STATICFILES_DIRS[0]}/{file_name}"
    cv2.imwrite(file_path, frame)
    return file_path

def verify_image(request):
    captured_image_path = capture_image()
    if captured_image_path:
        try:
            result = DeepFace.verify("static/testPhoto.jpg", captured_image_path, enforce_detection=False)

            # Імітація "accuracy" — тут очікується True, бо порівнюємо з testPhoto
            expected = True
            is_correct = result["verified"] == expected
            accuracy = int(is_correct)

            # Додатковий лог в консоль
            print("=== Перевірка облич ===")
            print(f"Verified: {result['verified']}")
            print(f"Distance: {result['distance']:.4f}")
            print(f"Threshold: {result['threshold']:.4f}")
            print(f"Accuracy (1 = правильно): {accuracy}")

            return JsonResponse({
                "verified": result["verified"],
                "distance": result["distance"],
                "threshold": result["threshold"],
                "accuracy": accuracy
            })

        except ValueError as e:
            return JsonResponse({"error": str(e)}, status=400)
    else:
        return JsonResponse({"error": "Не вдалося захопити зображення для перевірки"}, status=400)


def capture_test_photo(request):
    test_photo_path = capture_image("testPhoto.jpg")
    if test_photo_path:
        return JsonResponse({"success": True})
    else:
        return JsonResponse({"error": "Не вдалося захопити зображення для testPhoto"}, status=400)

def home(request):
    return render(request, 'verifier/home.html')



#face analize
from django.shortcuts import render
from django.http import HttpResponse
from deepface import DeepFace
import os
from django.conf import settings
import requests

def start(request):
    return render(request, 'verifier/start.html')

def faceAnalise(request):
    result = None
    relative_path = None  

    if request.method == 'POST':
        # Варіант 1: Завантаження файлу з комп'ютера
        if request.FILES.get('photo'):
            photo = request.FILES['photo']
            
            # Формуємо шляхи
            relative_path = os.path.join('uploads', photo.name)
            file_path = os.path.join(settings.MEDIA_ROOT, relative_path)

            # === ВИПРАВЛЕННЯ: Створюємо папку, якщо її немає ===
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            # ===================================================

            # Тепер безпечно зберігаємо файл
            with open(file_path, 'wb') as f:
                for chunk in photo.chunks():
                    f.write(chunk)

        # Варіант 2: Завантаження через URL (тут теж треба додати перевірку)
        elif request.POST.get('photo_url'):
            photo_url = request.POST.get('photo_url')
            try:
                response = requests.get(photo_url)
                if response.status_code == 200:
                    file_name = os.path.basename(photo_url.split("?")[0])
                    relative_path = os.path.join('uploads', file_name)
                    file_path = os.path.join(settings.MEDIA_ROOT, relative_path)

                    # === ТУТ ТЕЖ ДОДАЄМО СТВОРЕННЯ ПАПКИ ===
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    # =======================================

                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                else:
                    raise Exception("Не вдалося завантажити зображення.")
            except Exception as e:
                result = {"error": str(e)}
                return render(request, 'verifier/faceAnalise.html', {'result': result})

        # Аналіз через DeepFace (код залишається без змін)
        if relative_path:
            try:
                # Переконуємось, що файл записався
                if os.path.exists(file_path):
                    analysis_result = DeepFace.analyze(file_path, actions=['age', 'gender', 'emotion'])
                    result = analysis_result[0]
                    result['img_path'] = relative_path
                    
                    if 'gender' in result and isinstance(result['gender'], dict):
                        # Вибираємо стать з найбільшою ймовірністю
                        gender = max(result['gender'], key=result['gender'].get)
                        result['gender'] = gender
                else:
                    result = {"error": "Файл не знайдено після завантаження"}
            except Exception as e:
                result = {"error": f"Помилка аналізу: {str(e)}"}

    return render(request, 'verifier/faceAnalise.html', {'result': result, 'MEDIA_URL': settings.MEDIA_URL})