from django.http import HttpResponse, JsonResponse
from django.shortcuts import render

from disaster_detection.core.resnest_inference import inference


# git subtree pull --prefix=disaster_detection/core https://github.com/GoldenMine0502/DiasterDetectionCore.git master --squash
def inference_request(request):
    result = inference(request.number)
    return JsonResponse({
        'result': result
    })
