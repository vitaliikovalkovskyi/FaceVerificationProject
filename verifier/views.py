import cv2
from deepface import DeepFace
from django.http import JsonResponse
from django.shortcuts import render
from django.conf import settings
from django.templatetags.static import static

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
