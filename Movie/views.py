from django.shortcuts import render
from django.http import HttpResponse
from django.http import JsonResponse
import json
# from django.core import serializers
from Movie.models import Rating

# Create your views here.
def get_rating_list(request):
    if request.method == 'GET':
        get = request.GET
        user_id = get.get('user_id')
        # movie_id = get.get('movie_id')
        # rating = get.get('rating')

        models = Rating.objects.using('movie').filter(user_id=user_id)
        return render(request, 'home.html', 
            {'rating_list': models}
        )

    elif request.method == 'POST':
        post = request.POST
        user_id = post.get('user_id', 'default')
        # movie_id = post.get('movie_id')
        # rating = post.get('rating')
        
        models = Rating.objects.using('movie').filter(user_id=user_id)
        return render(request, 'home.html', 
            {'rating_list': models}
        )


def insert_rating_list(request):
    if request.method == 'GET':
        return HttpResponse('HTTP/GET is not support.')

    elif request.method == 'POST':
        
        post = request.POST
        user_id = post.get('user_id')
        movie_id = post.get('movie_id')
        rating = post.get('rating')

        records = Rating.objects.using('movie').filter(user_id=user_id, movie_id=movie_id)
        
        if records:
            for record in records:
                obj = Rating.objects.using('movie').get(id=record.id)
                obj.rating = rating
                obj.save()
        else:
            model = Rating(movie_id=movie_id, user_id=user_id, rating=rating)
            model.save()

        response = get_rating_list(request=request)
        return response


def delete_rating_record(request):
    if request.method == 'GET':
        return HttpResponse('HTTP/GET is not support.')

    elif request.method == 'POST':
        
        post = request.POST
        user_id = post.get('user_id')
        movie_id = post.get('movie_id')
        rating = post.get('rating')
        Rating.objects.using('movie').filter(user_id=user_id, movie_id=movie_id, rating=rating).delete()

    response = get_rating_list(request=request)
    return response