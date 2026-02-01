import os
import cv2
import base64
import requests
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from deepface import DeepFace
import shutil

# ============================================================
# ФУНКЦІЯ ЗБЕРЕЖЕННЯ BASE64 ФОТО
# ============================================================
def save_base64_image(data_url, file_name):
    """
    **ЦЕ ДЛЯ ЗБЕРЕЖЕННЯ ФОТО, НАДАНОГО З КАМЕРИ У ВИДІ BASE64**
    """
    try:
        header, encoded = data_url.split(",", 1)
        data = base64.b64decode(encoded)
        file_path = os.path.join(settings.STATICFILES_DIRS[0], file_name)
        with open(file_path, "wb") as f:
            f.write(data)
        return file_path
    except Exception as e:
        print(f"Помилка збереження: {e}")
        return None

# ============================================================
# ПЕРЕВІРКА ОБЛИЧЧЯ (VERIFY)
# ============================================================
def verify_image(request):
    """
    **ЦЕ ДЛЯ ПЕРЕВІРКИ ОБЛИЧЧЯ**
    Приймає POST з base64 зображення, порівнює з testPhoto
    """
    if request.method != 'POST':
        return JsonResponse({"error": "Method not allowed"}, status=405)

    image_data = request.POST.get('image_data')
    if not image_data:
        return JsonResponse({"error": "No data"}, status=400)

    # Зберігаємо фото з браузера
    captured_path = save_base64_image(image_data, "captured_image.jpg")
    test_photo_path = os.path.join(settings.STATICFILES_DIRS[0], "testPhoto.jpg")

    if not os.path.exists(test_photo_path):
        return JsonResponse({
            "error": "no_reference", 
            "message": "Спочатку зробіть еталонне фото (зліва)!"
        }, status=400)

    try:
        result = DeepFace.verify(
            img1_path=test_photo_path,
            img2_path=captured_path,
            enforce_detection=True
        )

        is_correct = result["verified"]
        accuracy = 1 if is_correct else 0

        return JsonResponse({
            "verified": result["verified"],
            "distance": result["distance"],
            "threshold": result["threshold"],
            "accuracy": accuracy
        })

    except Exception as e:
        error_msg = str(e)

        if "Face could not be detected" in error_msg:
            return JsonResponse({
                "error": "no_face_detected",
                "message": "На фото не видно обличчя. Будь ласка, станьте перед камерою."
            }, status=400)

        if "img2_path" in error_msg:
            return JsonResponse({
                "error": "no_face_detected",
                "message": "Не вдалося розпізнати обличчя на фото для перевірки. Спробуйте ще раз."
            }, status=400)

        if "img1_path" in error_msg:
            return JsonResponse({
                "error": "no_reference",
                "message": "Проблема з еталонним фото. Будь ласка, перезніміть еталон (зліва)."
            }, status=400)

        print(f"Server Error: {error_msg}")
        return JsonResponse({"error": "server_error", "message": f"Помилка сервера: {error_msg}"}, status=500)

# ============================================================
# ЗАХОПЛЕННЯ TEST PHOTO
# ============================================================
def capture_test_photo(request):
    """
    **ЦЕ ДЛЯ ЗАХОПЛЕННЯ ТЕСТОВОГО ФОТО**
    Перевіряє наявність обличчя перед збереженням як еталон
    """
    if request.method != 'POST':
        return JsonResponse({"error": "Method not allowed"}, status=405)

    image_data = request.POST.get('image_data')
    if not image_data:
        return JsonResponse({"error": "No data"}, status=400)

    temp_path = save_base64_image(image_data, "temp_test.jpg")

    try:
        DeepFace.extract_faces(img_path=temp_path, enforce_detection=True)
        save_base64_image(image_data, "testPhoto.jpg")
        return JsonResponse({"success": True, "message": "Еталонне фото успішно збережено."})
    except ValueError:
        return JsonResponse({
            "error": "no_face_detected",
            "message": "На еталонному фото не знайдено обличчя! Спробуйте ще раз."
        }, status=400)
    except Exception as e:
        return JsonResponse({"error": "server_error", "message": str(e)}, status=500)

# ============================================================
# FACE ANALYSIS
# ============================================================
def faceAnalise(request):
    """
    ЦЕ ДЛЯ АНАЛІЗУ ОБЛИЧЧЯ
    1. Очищає папку uploads.
    2. Зберігає нове фото.
    3. Аналізує.
    """
    result = None
    relative_path = None  
    upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')

    if request.method == 'POST':
        # --- КРОК 0: ОЧИЩЕННЯ ПАПКИ ---
        # Створюємо папку, якщо немає
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        else:
            # Видаляємо всі файли в папці uploads перед новим завантаженням
            for filename in os.listdir(upload_dir):
                file_path = os.path.join(upload_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path) # Видаляємо файл
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Не вдалося видалити {file_path}. Причина: {e}')

        # --- КРОК 1: ЗАВАНТАЖЕННЯ ---
        try:
            if request.FILES.get('photo'):
                photo = request.FILES['photo']
                relative_path = os.path.join('uploads', photo.name)
                file_path = os.path.join(settings.MEDIA_ROOT, relative_path)

                with open(file_path, 'wb') as f:
                    for chunk in photo.chunks():
                        f.write(chunk)

            elif request.POST.get('photo_url'):
                photo_url = request.POST.get('photo_url')
                response = requests.get(photo_url)
                if response.status_code == 200:
                    file_name = os.path.basename(photo_url.split("?")[0])
                    # Якщо ім'я файлу пусте або дивне, даємо дефолтне
                    if not file_name or len(file_name) < 3:
                        file_name = "url_image.jpg"
                        
                    relative_path = os.path.join('uploads', file_name)
                    file_path = os.path.join(settings.MEDIA_ROOT, relative_path)

                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                else:
                    raise Exception("Не вдалося завантажити зображення за посиланням.")
            else:
                 return render(request, 'verifier/faceAnalise.html', {'result': None})

            # --- КРОК 2: АНАЛІЗ ЧЕРЕЗ DeepFace ---
            if relative_path and os.path.exists(file_path):
                # Аналізуємо
                analysis_result = DeepFace.analyze(file_path, actions=['age', 'gender', 'emotion'])
                result = analysis_result[0]
                
                # Важливо: передаємо шлях до файлу для HTML
                result['img_path'] = relative_path

                # Виправлення для gender
                if 'gender' in result and isinstance(result['gender'], dict):
                    gender = max(result['gender'], key=result['gender'].get)
                    result['gender'] = gender
                
        except ValueError as e:
            error_msg = str(e)
            if "Face could not be detected" in error_msg:
                result = {"error": "Обличчя не знайдено! Переконайтеся, що на фото чітко видно обличчя."}
                # Якщо помилка, файл все одно залишається в папці, щоб ти бачив, ЩО саме не спрацювало.
                # Він видалиться при наступному запиті.
            else:
                result = {"error": f"Помилка даних: {error_msg}"}
        
        except Exception as e:
            result = {"error": f"Помилка: {str(e)}"}

    return render(request, 'verifier/faceAnalise.html', {'result': result, 'MEDIA_URL': settings.MEDIA_URL})
# ============================================================
# ДОДАТКОВІ VIEW (малі коментарі)
# ============================================================
def register(request):
    """
    Реєстрація користувача з ОБОВ'ЯЗКОВОЮ перевіркою наявності обличчя.
    """
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        
        # Отримуємо дані фото: або з камери (base64), або файл
        camera_data = request.POST.get('camera_data')
        file_upload = request.FILES.get('photo')

        temp_path = None
        # Формуємо ім'я файлу (наприклад: user_ivan_ref.jpg)
        file_name = f"user_{username}_ref.jpg"

        # --- 1. ЗБЕРЕЖЕННЯ ФОТО (Тимчасово або постійно) ---
        try:
            if camera_data:
                # Якщо фото з камери (base64)
                # Використовуємо твою функцію save_base64_image, якщо вона є, або пишемо тут:
                if "base64," in camera_data:
                    header, encoded = camera_data.split(",", 1)
                    data = base64.b64decode(encoded)
                else:
                    data = base64.b64decode(camera_data)

                temp_path = os.path.join(settings.STATICFILES_DIRS[0], file_name)
                with open(temp_path, "wb") as f:
                    f.write(data)

            elif file_upload:
                # Якщо завантажено файл через кнопку "Завантажити"
                temp_path = os.path.join(settings.STATICFILES_DIRS[0], file_name)
                with open(temp_path, 'wb') as f:
                    for chunk in file_upload.chunks():
                        f.write(chunk)
            else:
                return JsonResponse({"success": False, "error": "Фото не надано!"}, status=400)

        except Exception as e:
            return JsonResponse({"success": False, "error": f"Помилка збереження файлу: {str(e)}"}, status=500)

        # --- 2. ВАЛІДАЦІЯ ОБЛИЧЧЯ ЧЕРЕЗ DEEPFACE ---
        try:
            # enforce_detection=True - це головна перевірка!
            # Якщо обличчя немає, DeepFace викине ValueError
            DeepFace.extract_faces(img_path=temp_path, enforce_detection=True)

            # --- 3. ЯКЩО ОБЛИЧЧЯ ЗНАЙДЕНО ---
            # Тут мав би бути код збереження User в базу даних (Django Models).
            # Поки що ми просто залишаємо файл як еталон і повертаємо успіх.
            
            print(f"Успішна реєстрація: {username}. Обличчя знайдено.")
            return JsonResponse({
                "success": True, 
                "message": f"Користувач {username} успішно зареєстрований!"
            })

        except ValueError:
            # DeepFace не знайшов обличчя -> Видаляємо файл і вертаємо помилку
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return JsonResponse({
                "success": False, 
                "error": "no_face_detected", # Цей код ловить JS на фронтенді
                "message": "На фото не знайдено обличчя! Переконайтеся, що освітлення добре і обличчя в центрі."
            }, status=400)

        except Exception as e:
            # Будь-яка інша помилка DeepFace
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return JsonResponse({"success": False, "error": f"Помилка аналізу: {str(e)}"}, status=500)

    return JsonResponse({"success": False, "error": "Method not allowed"}, status=405)

# ============================================================
# VIEW З ПРОСТИМ РЕНДЕРОМ HTML
# ============================================================
def home(request):
    return render(request, 'verifier/home.html')

def startpage(request):
    return render(request, 'verifier/startpage.html')

def magisterjob(request):
    return render(request, 'verifier/magisterjob.html')