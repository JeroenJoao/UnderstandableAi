import os

from PIL import Image
from django.http import JsonResponse, HttpResponse, HttpResponseNotFound

from UnderstandableAi.settings import BASE_DIR
from dataset.Neural import getLayerPlot

def index(request, picnum, shapes, layer):
    if(layer == '0'):
        response = responseNormal(picnum, shapes)
    else:
        response = responseLayer(picnum, shapes, layer)

    return response


def responseNormal(picnum, shapes):
    response = HttpResponse(content_type="image/png")
    img = Image.open(os.path.join(BASE_DIR,'dataset/shapes/test_set/' + str(shapes) + '/drawing(' + str(picnum) + ').png'))
    img.save(response, 'png')
    return response

def responseLayer(picnum, shapes, layer):
    getLayerPlot(picnum, shapes)
    response = HttpResponse(content_type="image/png")
    img = Image.open(os.path.join(BASE_DIR, 'dataset\\image.png'))
    img.save(response, 'png')
    return response

def createJson(a):
    json = {}
    count = 0
    for i in a:
        json[str(count)] = i
        count+=1
    return json