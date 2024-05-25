from django.urls import path
from .views import StripAnalysisAPIView, index

urlpatterns = [
    path('analyze/', StripAnalysisAPIView.as_view(), name='strip-analysis'),
    path('', index, name='index'),
]
