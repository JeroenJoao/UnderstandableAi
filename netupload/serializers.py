from rest_framework import serializers
from UnderstandableAi.settings import BASE_DIR
from .models import File
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
import os
import h5py

class FileSerializer(serializers.ModelSerializer):

    class Meta():
        model = File
        fields = ['file1', 'file2', 'file3']


    def save(self):
        path = "netupload/uploads/"

        file = self.validated_data['file1']
        tempfile = file.read().decode("utf-8")
        with open(os.path.join(BASE_DIR, path + str(file)),"w+") as f:
            f.write(tempfile)

        file = self.validated_data['file2']
        tempfile = ContentFile(file.read())
        default_storage.save(str(file), tempfile)


        file = self.validated_data['file3']
        tempfile = file.read().decode("utf-8")
        with open(os.path.join(BASE_DIR, path + str(file)),"w+") as f:
            f.write(tempfile)

