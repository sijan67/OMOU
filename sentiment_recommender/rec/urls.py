from django.urls import path
from . import views

urlpatterns = [
  path('', views.home, name='home'),
  path('select', views.select, name='select'),
  path('recommendation', views.recommendation, name='recommendation')
]