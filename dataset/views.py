import os

from PIL import Image
from django.http import JsonResponse, HttpResponse, HttpResponseNotFound

from UnderstandableAi.settings import BASE_DIR
from dataset import ShapeSetNeural
from dataset import ResNetNeural
from dataset import saliency

def index(request,dataset, picnum, shapes, layer, saliency):
    if dataset == 'shapeset':
        response = shapeSet(picnum,shapes,layer, saliency)
    elif dataset == 'imagenet':
        response = ImageNet(picnum, layer, saliency)
    else:
        response = HttpResponseNotFound

    return response

def ImageNet(img, layer, saliency):
    if saliency == '1':
        response = responseSaliency(img)
    else:
        if (layer == '0'):
            response = responseNormalImageNet(img)
        else:
            response = responseLayerImageNet(img, layer)
    return response

def responseSaliency(img):
    saliency.getSaliency(img)
    response = HttpResponse(content_type="image/png")
    img = Image.open(
        os.path.join(BASE_DIR, 'dataset/ResNetSet/image2.png'))
    img.save(response, 'png')
    return response

def responseNormalImageNet(img):
    response = HttpResponse(content_type="image/png")
    img = Image.open(
        os.path.join(BASE_DIR, 'dataset/ResNetSet/' + str(img) + '.png'))
    img.save(response, 'png')
    return response

def responseLayerImageNet(img, layer):
    ResNetNeural.getLayerPlot(img,layer)
    response = HttpResponse(content_type="image/png")
    img = Image.open(os.path.join(BASE_DIR, 'dataset/ResNetSet/image.png'))
    img.save(response, 'png')
    return response

def shapeSet(picnum, shapes, layer, saliency):

    if (layer == '0'):
        response = responseNormal(picnum, shapes)
    else:
        response = responseLayer(picnum, shapes, layer)

    return response


def responseNormal(picnum, shapes):
    response = HttpResponse(content_type="image/png")
    img = Image.open(os.path.join(BASE_DIR,'dataset/ShapeSet/shapes/test_set/' + str(shapes) + '/drawing(' + str(picnum) + ').png'))
    img.save(response, 'png')
    return response


def responseLayer(picnum, shapes, layer):
    ShapeSetNeural.getLayerPlot(picnum, shapes,layer)
    response = HttpResponse(content_type="image/png")
    img = Image.open(os.path.join(BASE_DIR, 'dataset/ShapeSet/image.png'))
    img.save(response, 'png')
    return response


def createJson(a):
    json = {}
    count = 0
    for i in a:
        json[str(count)] = i
        count+=1
    return json