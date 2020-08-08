"""lab URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from Movie import views as movie_views
from Recommendation import views as rec_views
from Classifier import views as classifier_views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', movie_views.get_rating_list),
    path('insert/', movie_views.insert_rating_list),
    path('delete/', movie_views.delete_rating_record),
    path('recommendation/', rec_views.recommend),
    path('classifier/', classifier_views.classify)
]

