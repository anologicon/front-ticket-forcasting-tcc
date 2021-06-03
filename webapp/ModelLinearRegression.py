from joblib import load
import os
import pandas as pd
import holidays
import numpy as np

class ModelLinearRegression:

    def __init__(self):
        self.model = load('webapp/linearModel.joblib')

    def getModel(self):
        return self.model
