# Generated by Django 3.0.8 on 2020-08-03 12:27

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Recommendation', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='rating',
            name='id',
            field=models.AutoField(primary_key=True, serialize=False),
        ),
    ]