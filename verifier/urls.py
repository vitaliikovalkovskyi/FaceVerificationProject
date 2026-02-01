from django.urls import path
from .views import startpage, home, verify_image, capture_test_photo, faceAnalise,magisterjob, krystyna_view
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
    path('register/step1/', views.register_step1, name='register_step1'),
    path('register/step2/', views.register_step2, name='register_step2'),
    path('users/', views.user_list, name='user_list'),
    path('users/delete/<int:user_id>/', views.delete_user, name='delete_user'),
    path('krystyna/', views.krystyna_view, name='krystyna'),
    
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


