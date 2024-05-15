from django.urls import path
from .views import home, verify_image, capture_test_photo

urlpatterns = [
    path('', home, name='home'),
    path('verify-image/', verify_image, name='verify_image'),
    path('capture-test-photo/', capture_test_photo, name='capture_test_photo'),
]
