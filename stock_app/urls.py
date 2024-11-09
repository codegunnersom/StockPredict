# stock_app/urls.py
from django.urls import path
from .views import predict_stock

urlpatterns = [
    path('', predict_stock, name='predict_stock'),  # The root URL will call predict_stock view
]