from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from PIL import Image

from core.vision_transformer_inference import inference


# git subtree pull --prefix=disaster_detection/core https://github.com/GoldenMine0502/DiasterDetectionCore.git master --squash
def inference_request(request):
    # POST일 때 입력받은 이미지를 통해 inference 결과를 보여줌
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        image = Image.open(image_file)
        output, label = inference(image)
        output = output.item()
        label = label.item()

        return JsonResponse({
            'output': output,
            'label': label,
        })

    return JsonResponse({
        'error': 'only supports post request'
    })
