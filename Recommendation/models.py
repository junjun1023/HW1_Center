from django.db import models

class Movie(models.Model):
    movie_id = models.IntegerField(primary_key=True)
    title = models.CharField(max_length=255)
    genres = models.CharField(max_length=64)
    class Meta:
        db_table = 'movies'
        app_label = 'Recommendation'


class Rating(models.Model):
    user_id = models.IntegerField()
    movie_id = models.IntegerField()
    rating = models.DecimalField(decimal_places=3, max_digits=10)
    timestamp = models.TimeField()
    class Meta:
        db_table = 'ratings'
        app_label = 'Recommendation'


class Tag(models.Model):
    user_id = models.IntegerField()
    movie_id = models.IntegerField()
    tag = models.CharField(max_length=255)
    timestamp = models.TimeField()
    class Meta:
        db_table = 'tags'
        app_label = 'Recommendation'