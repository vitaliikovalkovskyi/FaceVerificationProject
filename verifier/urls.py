from django.urls import path
from .views import startpage, home, verify_image, capture_test_photo, faceAnalise,magisterjob, krystyna_view, login_face_verify, login_otp_verify, login_page
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
    path('login/<str:username>/', views.login_page, name='login_page'),
    path('api/login-verify/', views.login_face_verify, name='login_face_verify'),
    path('api/login-otp/', views.login_otp_verify, name='login_otp_verify'),
    path('login/success/<str:username>/', views.logined_page, name='logined_page'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


