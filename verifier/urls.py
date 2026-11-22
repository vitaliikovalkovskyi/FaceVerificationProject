from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views, views_tests


urlpatterns = [
    path('', views.startpage, name='start'),
    path('startpage/', views.startpage, name='start'),
    path('magisterjob/', views.magisterjob, name='magisterjob'),
    path('register/step1/', views.register_step1, name='register_step1'),
    path('register/step2/', views.register_step2, name='register_step2'),
    path('users/', views.user_list, name='user_list'),
    path('users/delete/<int:user_id>/', views.delete_user, name='delete_user'),
    path('login/<str:username>/', views.login_page, name='login_page'),
    path('api/login-verify/', views.login_face_verify, name='login_face_verify'),
    path('api/login-otp/', views.login_otp_verify, name='login_otp_verify'),
    path('api/liveness-detect/', views.liveness_detect, name='liveness_detect'),
    path('login/success/<str:username>/', views.logined_page, name='logined_page'),
    path('benchmark/', views.benchmark_page, name='benchmark_page'),
    path('benchmark-stream/', views.benchmark_stream, name='benchmark_stream'),

    # Demo use case: система проходження тестів
    path('invite/<str:token>/', views_tests.invite, name='invite'),
    path('invite/<str:token>/done/', views_tests.invite_done, name='invite_done'),
    path('tests/<int:test_id>/', views_tests.test_detail, name='test_detail'),
    path('tests/<int:test_id>/login/', views_tests.test_login, name='test_login'),
    path('tests/<int:test_id>/login/<str:username>/', views_tests.test_login_user, name='test_login_user'),
    path('tests/attempts/<int:attempt_id>/', views_tests.attempt_result, name='attempt_result'),
    path('logout/', views_tests.logout_view, name='logout'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
