import os

from PIL import Image
from django.http import JsonResponse, HttpResponse, HttpResponseNotFound

from UnderstandableAi.settings import BASE_DIR
from dataset import ShapeSetNeural
from dataset import ResNetNeural
from dataset import saliency


def index(request,dataset, picname, upload, layer, saliency):
    if upload == '1':
        picname = 'uploads/' + picname
    if dataset == 'shapeset':
        response = shapeSet(picname, layer, saliency)
    elif dataset == 'imagenet':
        response = ImageNet(picname, layer, saliency)
    else:
        response = HttpResponseNotFound()
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



def shapeSet(picnum, layer, saliency):

    if (layer == '0'):
        response = responseNormal(picnum)
    else:
        response = responseLayer(picnum, layer)

    return response


def responseNormal(picname):
    response = HttpResponse(content_type="image/png")
    img = Image.open(os.path.join(BASE_DIR,'dataset/ShapeSet/shapes/' + picname + '.png' ))
    img.save(response, 'png')
    return response


def responseLayer(imgname, layer):
    ShapeSetNeural.getLayerPlot(imgname, layer)
    response = HttpResponse(content_type="image/png")
    img = Image.open(os.path.join(BASE_DIR, 'dataset/ShapeSet/image.png'))
    img.save(response, 'png')
    return response
