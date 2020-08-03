from django.db import models

# Create your models here.
class Movie(models.Model):
    movie_id = models.IntegerField(primary_key=True)
    title = models.CharField(max_length=256)
    genres = models.CharField(max_length=64)
    class Meta:
        db_table = 'movies'
        app_label = 'Movie'
            

class Rating(models.Model):
    id = models.AutoField(primary_key=True)
    user_id = models.IntegerField()
    movie_id = models.IntegerField()
    rating = models.DecimalField(decimal_places=3, max_digits=10)
    class Meta:
        db_table = 'ratings'
        app_label = 'Movie'
