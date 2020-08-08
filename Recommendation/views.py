from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
import json


# Create your views here.
def recommend(request):
    return HttpResponse('Recommend')