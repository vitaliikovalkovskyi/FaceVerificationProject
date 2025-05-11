import cv2
from deepface import DeepFace
from django.http import JsonResponse
from django.shortcuts import render
from django.conf import settings
from django.templatetags.static import static
import os

def capture_image(file_name="captured_image.jpg"):
    cap = cv2.VideoCapture(2)
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
            return JsonResponse({"verified": result["verified"]})
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
        # Обробка завантаженого файлу
        if request.FILES.get('photo'):
            photo = request.FILES['photo']
            relative_path = os.path.join('uploads', photo.name)  # Відносний шлях до файлу
            file_path = os.path.join(settings.MEDIA_ROOT, relative_path)  # Абсолютний шлях

            
            with open(file_path, 'wb') as f:
                for chunk in photo.chunks():
                    f.write(chunk)

        # Обробка URL
        elif request.POST.get('photo_url'):
            photo_url = request.POST.get('photo_url')  # Отримуємо посилання на зображення
            try:
                response = requests.get(photo_url)
                if response.status_code == 200:
                    file_name = os.path.basename(photo_url.split("?")[0])  # Ім'я файлу без параметрів у URL
                    relative_path = os.path.join('uploads', file_name)
                    file_path = os.path.join(settings.MEDIA_ROOT, relative_path)

                    # Зберігаємо файл у папку media/uploads
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                else:
                    raise Exception("Не вдалося завантажити зображення за посиланням.")
            except Exception as e:
                result = {"error": str(e)}  
                return render(request, 'verifier/faceAnalise.html', {'result': result, 'MEDIA_URL': settings.MEDIA_URL})

        # Аналізуємо фото за допомогою DeepFace
        if relative_path:
            try:
                analysis_result = DeepFace.analyze(file_path, actions=['age', 'gender', 'emotion'])
                result = analysis_result[0]  # Отримуємо результат аналізу для першого обличчя на фото
                result['img_path'] = relative_path  # Додаємо шлях до зображення в результат
                if 'gender' in result and isinstance(result['gender'], dict):
                    result['gender'] = {k: round(v, 2) for k, v in result['gender'].items()}
            except Exception as e:
                result = {"error": str(e)}  # Якщо сталася помилка

    return render(request, 'verifier/faceAnalise.html', {'result': result, 'MEDIA_URL': settings.MEDIA_URL})
