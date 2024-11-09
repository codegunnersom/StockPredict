from django.db import models

# Create your models here.

# stock_app/models.py
from django.db import models

class StockPrediction(models.Model):
    ticker = models.CharField(max_length=10)
    predicted_price = models.FloatField()
    prediction_date = models.DateField(auto_now_add=True)