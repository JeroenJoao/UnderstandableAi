from rest_framework import serializers
from UnderstandableAi.settings import MEDIA_ROOT
from .models import File
from PIL import Image

class FileSerializer(serializers.ModelSerializer):

    class Meta():
        model = File
        fields = ['file']


    def save(self):
        file = self.validated_data['file']
        img = Image.open(file)
        img.save(MEDIA_ROOT + str(file).split('.')[0] + '.png', "PNG")