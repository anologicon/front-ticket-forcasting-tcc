from joblib import load
import os
import pandas as pd
import holidays
import numpy as np
import streamlit as st


class ModelDecisiontree:

    def __init__(self):
        self.model = load('webapp/dtModel.joblib')

    def getModel(self):
        return self.model