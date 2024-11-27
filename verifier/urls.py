from django.urls import path
from .views import start, home, verify_image, capture_test_photo, faceAnalise
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('', start, name='start'),
    path('home/', home, name='home'),
    path('verify-image/', verify_image, name='verify_image'),
    path('capture-test-photo/', capture_test_photo, name='capture_test_photo'),
    path('faceAnalise/', faceAnalise, name='faceAnalise'),
    
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


