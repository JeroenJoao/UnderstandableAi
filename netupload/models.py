from django.db import models

class File(models.Model):
    file1 = models.FileField()
    file2 = models.FileField()
    file3 = models.FileField()