from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/predictions/', views.api_predictions, name='predictions'),
    path('api/predict/', views.api_predict, name='predict'),
]