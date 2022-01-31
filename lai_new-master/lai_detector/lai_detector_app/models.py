from django.db import models
from django.contrib.auth.models import User

# Create your models here.


class Image(models.Model):
    path = models.CharField("path", max_length=255, null=True)
    detected_path = models.CharField("detected_path", max_length=255, null=True)
    output_path = models.CharField("output_path", max_length=255, null=True)
    
    LAI = models.FloatField("LAI", null=True, blank=True)
    FVC = models.FloatField("FVC", null=True, blank=True)
    X = models.FloatField("X", null=True, blank=True)
    Y = models.FloatField("Y", null=True, blank=True)
    Latitude = models.FloatField("Latitude", null=True, blank=True)
    Longitude = models.FloatField("Longitude", null=True, blank=True)

    image_file = models.ImageField(upload_to='images/%d_%m_%Y_%H_%M_%S/', blank=True, null=True)
    detected_image_file = models.ImageField(upload_to='images/%d_%m_%Y_%H_%M_%S/detected/', blank=True, null=True)
    detected_output_image_file = models.ImageField(upload_to='images/%d_%m_%Y_%H_%M_%S/detected_output/', blank=True, null=True)
