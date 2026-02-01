from django.urls import path
from .views import startpage, home, verify_image, capture_test_photo, faceAnalise,magisterjob
from django.conf import settings
from django.conf.urls.static import static
from . import views


urlpatterns = [
    path('', startpage, name='start'),
    path('startpage/', startpage, name='start'),
    path('home/', home, name='home'),
    path('verify-image/', verify_image, name='verify_image'),
    path('capture-test-photo/', capture_test_photo, name='capture_test_photo'),
    path('faceAnalise/', faceAnalise, name='faceAnalise'),
    path('magisterjob/', magisterjob, name='magisterjob'),
    path('register/', views.register, name='register'),
    
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


