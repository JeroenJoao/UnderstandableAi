from django.http import JsonResponse

def index(request, num, vistype):
    data = startCalc(num, vistype)

    return JsonResponse(createJson(data))

def startCalc(a, b):
    int(a), int(b)
    num1 = int(a) * int(b)
    num2 = int(a) + int(b)
    return [a, b]

def createJson(a):
    json = {}
    count = 0
    for i in a:
        json[str(count)] = i
        count+=1
    return json