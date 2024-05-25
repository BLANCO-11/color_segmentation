from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import ImageUploadSerializer
from django.shortcuts import render
from .strip_process import process_image
import numpy as np
import cv2 

class StripAnalysisAPIView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = ImageUploadSerializer(data=request.data)
        if serializer.is_valid():
            image = request.FILES['image']
            results = self.analyze_strip(image)
            return Response(results, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def analyze_strip(self, image):
        # Read the image using OpenCV
        image_array = np.frombuffer(image.read(), np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        final = process_image(image)
        return final
    
    
def index(request):
    return render(request, 'index.html')
