"""UnderstandableAi URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
import dataset.views
from netupload.views import NetView
from upload.views import FileView
from rest_framework.views import APIView
from rest_framework_api_key.permissions import HasAPIKey



urlpatterns = [
    path('dataset/<dataset>/<upload>/<layer>/<picname>/<saliency>/<size>/<grad1>/<grad2>', dataset.views.index),
    path('netupload/', NetView.as_view()),
    path('upload/', FileView.as_view()),
    path('admin/', admin.site.urls),
]
