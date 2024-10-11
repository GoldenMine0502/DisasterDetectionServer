import numpy as np
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from PIL import Image

from core.vision_transformer_inference import inference
from django.views.decorators.csrf import csrf_exempt


def softmax(logits):
    exp_values = np.exp(logits - np.max(logits))  # Overflow 방지를 위해 최댓값을 빼줌
    return exp_values / np.sum(exp_values)


# ['collapsed_building', 'fire', 'flooded_areas', 'normal', 'traffic_incident']
# git subtree pull --prefix=disaster_detection/core https://github.com/GoldenMine0502/DiasterDetectionCore.git master --squash
@csrf_exempt
def inference_request(request):
    # POST일 때 입력받은 이미지를 통해 inference 결과를 보여줌
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        image = Image.open(image_file)
        output, label = inference(image)
        # print(output, label)
        # output = output.item()
        output = softmax(output)
        output = output.tolist()
        output = [round(num * 100, 2) for num in output]
        label = label.item()

        return JsonResponse({
            'output': output,
            'label': label,
        })

    return JsonResponse({
        'error': 'no image or error'
    })
