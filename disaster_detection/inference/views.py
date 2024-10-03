from django.http import HttpResponse
from django.shortcuts import render


# git subtree pull --prefix=disaster_detection/core https://github.com/GoldenMine0502/DiasterDetectionCore.git master --squash
def inference(request):
    return HttpResponse("Hello, World!")
