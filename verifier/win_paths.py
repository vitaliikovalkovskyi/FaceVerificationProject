"""
Обхід не-ASCII шляхів на Windows.

Проєкт лежить у теці з кирилицею («Магістратура»), а нативні частини
OpenCV, DeepFace та MediaPipe не відкривають файли за такими шляхами.
Windows має для кожного шляху короткий 8.3-варіант, що складається лише
з ASCII, — його й підставляємо.
"""
import os


def ascii_path(path: str) -> str:
    """Повертає короткий 8.3-шлях, якщо `path` містить не-ASCII символи; інакше — без змін."""
    path = str(path)
    if os.name != "nt" or path.isascii():
        return path
    import ctypes
    buf = ctypes.create_unicode_buffer(1024)
    if ctypes.windll.kernel32.GetShortPathNameW(path, buf, len(buf)) and buf.value.isascii():
        return buf.value
    return path


def patch_cv2_data_path():
    """DeepFace шукає Haar-каскади детектора 'opencv' відносно cv2.__file__."""
    import cv2
    cv2.__file__ = ascii_path(cv2.__file__)
