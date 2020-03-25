from rest_framework import serializers
from UnderstandableAi.settings import BASE_DIR
from .models import File
from PIL import Image
import os

class FileSerializer(serializers.ModelSerializer):

    class Meta():
        model = File
        fields = ['file']


    def save(self):
        file = self.validated_data['file']
        img = Image.open(file)
        path = "dataset/uploads/" + str(file).split('.')[0] + '.png'
        img.save(os.path.join(BASE_DIR, path), "PNG")
