from django.db import models

class Movie(models.Model):
    movieId = models.IntegerField(primary_key=True)
    title = models.CharField(max_length=255)
    genres = models.CharField(max_length=255)
    class Meta:
        db_table = 'movies'
        app_label = 'Recommendation'


class Rating(models.Model):
    id = models.AutoField(primary_key=True)
    userId = models.IntegerField()
    movieId = models.IntegerField()
    rating = models.DecimalField(decimal_places=3, max_digits=10)
    timestamp = models.DateTimeField(auto_now=True)
    class Meta:
        db_table = 'ratings'
        app_label = 'Recommendation'


class Tag(models.Model):
    userId = models.IntegerField()
    movieId = models.IntegerField()
    tag = models.CharField(max_length=255)
    timestamp = models.DateTimeField(auto_now=True)
    class Meta:
        db_table = 'tags'
        app_label = 'Recommendation'