import os
import queue
import time
from PIL import Image
from django.http import JsonResponse, HttpResponse, HttpResponseNotFound
from UnderstandableAi.settings import AVAILABLE
from UnderstandableAi.settings import BASE_DIR
from dataset import ShapeSetNeural
from dataset import ResNetNeural
from dataset import saliency
from dataset import CustomSaliency
from dataset import CustomLayer

requestQueue = queue.Queue()
requestToResponse = {}

class requestObject():

    def __init__(self, request, dataset, picname, upload, layer, saliency, size, grad1, grad2):
        self.request = request
        self.dataset = dataset
        self.picname = picname
        self.upload = upload
        self.layer = layer
        self.saliency = saliency
        self.size = size
        self.grad1 = grad1
        self.grad2 = grad2
        self.response = HttpResponseNotFound

    def handle(self):
        self.response = solve(self.request,self.dataset, self.picname, self.upload, self.layer, self.saliency)

    def getResponse(self):
        return self.response



def index(request, dataset, picname, upload, layer, saliency, size, grad1, grad2):
    req = requestObject(request, dataset, picname, upload, layer, saliency, size, grad1, grad2)
    requestQueue.put(req)
    requestToResponse[request] = None
    while requestToResponse.get(request) == None:
        executeRequest()
        time.sleep(1)
    return requestToResponse.pop(request)


def executeRequest():
    global AVAILABLE
    if AVAILABLE:
        AVAILABLE = False
        request = requestQueue.get()
        try:
            response = solve(request.request, request.dataset, request.picname, request.upload, request.layer,
                             request.saliency, request.size, request.grad1, request.grad2)
        except:
            response = HttpResponseNotFound
        AVAILABLE = True
        requestToResponse[request.request] = response


def solve(request,dataset, picname, upload, layer, saliency, size, grad1, grad2):
    if upload == '1':
        picname = 'uploads/' + picname
    if dataset == 'shapeset':
        response = shapeSet(picname, layer, saliency)
    elif dataset == 'imagenet':
        response = ImageNet(picname, layer, saliency)
    elif dataset == 'custom':
        response = customNet(picname, layer, saliency, size, grad1, grad2)
    else:
        response = HttpResponseNotFound()
    return response


def customNet(img, layer, saliency, size, grad1, grad2):
    if saliency == '1':
        response = responseCustSal(img, size, grad1, grad2)
    else:
        if (layer == '0'):
            response = responseNormalImageNet(img)
        else:
            response = responseCustLayer(img, layer, size)
    return response


def responseCustSal(img, size, grad1, grad2):
    CustomSaliency.getSaliency(img, size, grad1, grad2)
    response = HttpResponse(content_type="image/png")
    img = Image.open(
        os.path.join(BASE_DIR, 'dataset/CusNet/image2.png'))
    img.save(response, 'png')
    return response


def responseCustLayer(img, layer, size):
    CustomLayer.getLayerPlot(img, layer, size)
    response = HttpResponse(content_type="image/png")
    img = Image.open(
        os.path.join(BASE_DIR, 'dataset/CusNet/image.png'))
    img.save(response, 'png')
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
        os.path.join(BASE_DIR, 'dataset/' + str(img) + '.png'))
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
