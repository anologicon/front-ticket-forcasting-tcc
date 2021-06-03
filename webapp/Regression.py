import pandas as pd
import holidays
import numpy as np
from joblib import load

class Regression:

    def __init__(self, newModel):
        self.model = newModel.getModel()

    def predict(self, df):

        df['data'] = pd.to_datetime(df['data'])

        dfParse = self.newFeatures(df)

        dfEncoded = self.encoder(dfParse)

        dfEncoded.drop(['data'], axis=1, inplace=True)

        predict = self.model.predict(dfEncoded)

        outputDf = df.copy()

        outputDf['result'] = np.round(predict)

        dfFiltered = outputDf[['data',
                            'hora', 'salaCinema', 'result']]

        return dfFiltered

    def encoder(self, dataset):
        dataset_encoded = dataset.copy()

        labelEncoders = load('webapp/labelEncoders.joblib')

        categoricalCools = ('diretor', 'salaCinema', 'distribuidor',
                            'cast_1', 'cast_2', 'key_word_0', 'key_word_1', 'key_word_2')

        for col in categoricalCools:
            dataset_encoded[col] = labelEncoders[col].transform(
                dataset_encoded[col])

        return dataset_encoded

    def newFeatures(self, dataset):
        br_holidays = holidays.Brazil()

        # Separando o mes da sessão
        dataset['mes'] = dataset['data'].dt.month

        # Separando o dia da sessão
        dataset['dia'] = dataset['data'].dt.day

        # Dia do ano
        dataset['diaAno'] = dataset['data'].dt.dayofyear

        # Semana do ano
        dataset['semanaAno'] = dataset['data'].dt.weekofyear

        # Separando o dia da semana
        dataset['diaSemana'] = dataset['data'].dt.dayofweek

        # Separando se é feriado ou não
        dataset['isFeriado'] = dataset['data'].apply(
            lambda x: x in br_holidays)

        # Separando se é fim de semana ou não
        dataset['fimDeSemana'] = dataset['data'].apply(
            lambda x: 1 if(x.dayofweek == 5 or x.dayofweek == 6) else 0)

        return dataset
