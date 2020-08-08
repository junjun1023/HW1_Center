from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse

from PIL import Image
from Classifier import resnet

import base64
import torch
import torchvision
import torchvision.transforms as transforms

model, _ = resnet.load_checkpoint('Classifier/resnet_34.pth', resnet.BasicBlock, [3, 4, 6, 3], 1)
model.eval()

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

image_path = "image.png"

def classify(request):
    image = request.POST.get('image')

    image = base64.b64decode(image)

    # save image
    image_file = open(image_path, 'wb')
    image_file.write(image)
    image_file.close()

    # open image and resize
    image = Image.open(image_path)
    image = image.resize((32, 32), Image.BILINEAR)

    image = transform(image)
    image = image.expand(1, 3, 32, 32)

    outputs = model(image)
    index = torch.argmax(outputs)
    index = index.item()

    return JsonResponse({
        'image': "",
        'result': classes[index]
        })
