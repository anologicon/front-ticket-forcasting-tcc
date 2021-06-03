import streamlit as st
import pandas as pd
from ModelDecisiontree import ModelDecisiontree
from ModelLinearRegression import ModelLinearRegression
from joblib import load
import plotly.offline as py
import plotly.graph_objs as go
from Regression import Regression

diretores = ['Sam Mendes', 'Michael Spierig, Peter Spierig', 'Michael Chaves',
             'James Gray', 'Jenny Gage', 'Robert Rodriguez',
             'Rawson Marshall Thurber', 'Elizabeth Banks', 'Lars Klevberg',
             'José Eduardo Belmonte', 'Dennis Widmyer, Kevin Kolsch',
             'Alex Ross Perry', 'James Foley', 'Greg Berlanti', 'Ben Stassen',
             'Steven Caple Jr.', 'Sean Anders', 'Eli Roth', 'Stephen Gaghan',
             'Mike Flanagan', 'Tatsuya Nagamine', 'James Mangold',
             'Michael Dougherty', 'Ari Sandel', 'Peter Farrelly',
             'David Gordon Green', 'Neil Marshall', 'André Ovredal',
             'Andy Muschietti', 'Chad Stahelski', 'David Kerr', 'Alex Kendrick',
             'Nicholas McCarthy', '104', 'Oz Perkins', 'Jennifer Yuh Nelson',
             'Simon Curtis', 'Roland Emmerich', 'Fede Alvarez',
             'Christian Rivers', 'Bradley Cooper', 'Chris Sanders',
             'Nicolas Pesce', 'Aleksey Tsitsilin', 'Rob Marshall',
             'Carlos Saldanha', 'Damien Chazelle', 'Julius Avery',
             'Charles E. Bastien', 'Andrew Hyatt', 'Rob Letterman',
             'Alexandre Aja', 'Dexter Fletcher', 'Adam Robitel', 'Scott Speer',
             'Rian Johnson', 'Nick Bruno, Troy Quane', 'Peter Segal',
             'Paul Feig', 'Baltasar Kormákur', 'M. Night Shyamalan',
             'Danny Boyle']

distribuidor = ['Universal', 'Paris Filmes - Sm Distribuidora',
                'Warner Bros. Pictures', 'Fox Filmes Do Brasil', 'Diamond Filmes',
                'Sony Pictures Home Entertainment do Brasil LTDA',
                'Imagem Filmes - Wmix', 'Paramount Pictures', 'Walt Disney',
                '360 Entreterimento e Distribuicao de Filmes LTDA - ME',
                'Galeria Distribuidora Audiovisual Ltda']

keywords_0 = ['world war i', 'drug addiction', 'the conjuring universe', 'moon',
              'based on novel or book', 'martial arts', 'china', 'spy',
              'artificial intelligence', 'prison', 'forest', 'sexual identity',
              'london, england', 'ukraine', 'adoption', 'robbery',
              'space battle', 'mexico', 'halloween', 'southern usa', 'illinois',
              'new york city', 'christianity', 'pennsylvania, usa', 'greece',
              'witch', 'world war ii', 'country music', 'japan', 'magic',
              'spain', 'france', 'computer animation', 'roman empire',
              'detective', 'florida', 'medium', 'seattle', 'bunker', 'mother',
              'holiday', 'sailboat', 'sequel', 'pop singer']

keywords_1 = ['british army', 'haunted house', 'lanetli gözyasları',
              'loss of loved one', 'love', 'bounty hunter', 'fire',
              'high technology', 'remake', 'gang war', 'cat', 'children',
              'eroticism', 'sexuality', 'self-discovery', 'sport',
              'social worker', 'detective', 'island', 'transformation',
              'biography', 'boston, massachusetts', 'sequel', 'friendship',
              'halloween', 'secret society', 'small town', 'clown',
              'martial arts', 'slapstick', 'musical', 'fairy tale', 'dystopia',
              'reincarnation', 'war', 'sweden', 'post-apocalyptic future',
              'waitress', 'gold rush', 'kingdom', 'based on novel or book',
              'madrid, spain', 'world war ii', 'dog', 'bible', 'amnesia',
              'hurricane', 'drug abuse', 'key', 'sunlight', 'space battle',
              'inventor', 'career woman', 'london, england', 'boat', 'superhero',
              'the beatles']

keywords_2 = ['race against time', 'earthquake', 'la llorona', 'planet mars',
              'teenage crush', 'extreme sports', 'skyscraper', 'remake',
              'killer doll', 'prison break', 'husband wife relationship',
              'family', 'sequel', 'based on novel or book', 'palace',
              'based on a true story', 'ship', 'supernatural', 'resurrection',
              'sport', 'giant monster', "based on children's book", 'racism',
              'knife', 'based on comic', 'scarecrow', 'carnival', 'casablanca',
              'cross country', 'reincarnation', 'woods', 'super power',
              'battle of midway', 'hacker', 'dystopia', 'self-destruction',
              'dog', 'children', 'snow', 'nanny', 'europe', 'nasa', 'nazi',
              'nero', 'pokémon', 'alligator', '1970s', 'haunted house',
              'terminal illness', 'failure', 'save the planet',
              'forty something', 'homeless shelter', 'marriage proposal',
              'psychiatric hospital', 'alternate reality']

cast_1 = ['Dean-Charles Chapman', 'Jason Clarke', 'Raymond Cruz',
          'Tommy Lee Jones', 'Hero Fiennes Tiffin', 'Christoph Waltz',
          'Neve Campbell', 'Naomi Scott', 'Aubrey Plaza', 'Jackson Antunes',
          'Amy Seimetz', 'Hayley Atwell', 'Jamie Dornan', 'Josh Duhamel',
          'Jo Wyatt', 'Sylvester Stallone', 'Rose Byrne',
          "Vincent D'Onofrio", 'Antonio Banderas', 'Rebecca Ferguson',
          'Aya Hisakawa', 'Matt Damon', 'Vera Farmiga', 'Madison Iseman',
          'Mahershala Ali', 'Judy Greer', 'Milla Jovovich', 'Michael Garza',
          'James McAvoy', 'Halle Berry', 'Emma Thompson', 'Alex Kendrick',
          'Jackson Robert Scott', 'Lily James', 'Samuel Leakey',
          'Harris Dickinson', 'Kevin Costner', 'Patrick Wilson',
          'Beau Gadsdon', 'Robert Sheehan', 'Lady Gaga', 'Dan Stevens',
          'Demián Bichir', 'Vladimir Zaytsev', 'Lin-Manuel Miranda',
          'Kate McKinnon', 'Claire Foy', 'Wyatt Russell', 'Devan Cohen',
          'James Faulkner', 'Justice Smith', 'Barry Pepper', 'Jamie Bell',
          'Angus Sampson', 'Patrick Schwarzenegger', 'Carrie Fisher',
          'Tom Holland', 'Vanessa Hudgens', 'Henry Golding', 'Sam Claflin',
          'Bruce Willis']

cast_2 = ['Mark Strong', 'Sarah Snook', 'Marisol Ramirez', 'Ruth Negga',
          'Khadijha Red Thunder', 'Jennifer Connelly', 'Chin Han',
          'Ella Balinska', 'Brian Tyree Henry', 'Bianca Muller',
          'Jeté Laurence', 'Bronte Carmichael', 'Eric Johnson',
          'Jennifer Garner', 'Mari Devon', 'Dolph Lundgren',
          'Isabela Merced', 'Dean Norris', 'Michael Sheen', 'Kyliegh Curran',
          'Ryou Horikawa', 'Jon Bernthal', 'Millie Bobby Brown',
          'Caleel Harris', 'Linda Cardellini', 'Andi Matichak',
          'Ian McShane', 'Gabriel Rush', 'Bill Hader', 'Olga Kurylenko',
          'Ben Davies', 'Colm Feore', 'Christine Baranski',
          'Charles Babalola', 'Patrick Gibson', 'Amanda Seyfried',
          'Luke Kleintank', 'Sverrir Gudnason', 'Hugo Weaving',
          'Sam Elliott', 'Colin Woodell', 'John Cho', 'Olga Zubkova',
          'Ben Whishaw', 'Anthony Anderson', 'Jason Clarke',
          'Mathilde Ollivier', 'Drew Davis', 'Olivier Martinez',
          'Kathryn Newton', 'Morfydd Clark', 'Richard Madden',
          'Leigh Whannell', 'Rob Riggle', 'Adam Driver', 'Rashida Jones',
          'Leah Remini', 'Emma Thompson', 'Jeffrey Thomas',
          'Anya Taylor-Joy', 'Ed Sheeran']

classificacao = [12, 14, 16,  0, 10]


generos = ['Animação', 'Aventura', 'Ação', 'Comédia',
           'Crime', 'Drama', 'Família', 'Fantasia', 'Suspense', 'Terror',
           'Thriller', 'Biografia', 'Documentário', 'Esporte', 'Faroeste',
           'Ficção científica', 'Histórico', 'Mistério', 'Musical', 'Romance']

filme = {}


for genero in generos:
    filme[genero] = 0


st.sidebar.markdown('# Filme')

st.sidebar.markdown('---')

st.sidebar.markdown(r"""## Diretor """)

filme['diretor'] = st.sidebar.selectbox('Selecionado', diretores)

st.sidebar.markdown(r"""##  Distribuidor """)

filme['distribuidor'] = st.sidebar.selectbox('Selecionado', distribuidor)

st.sidebar.markdown(r"""##  Classificação """)

filme['classificacao'] = st.sidebar.selectbox('Selecionado', classificacao)

st.sidebar.markdown(r"""##  Tempo do filme (em minutos) """)

filme['tempoFilme'] = st.sidebar.number_input('Tempo')

st.sidebar.markdown(r"""##  Palavras chaves """)

filme['key_word_0'] = st.sidebar.selectbox('Palavara chave 1', keywords_0)

filme['key_word_1'] = st.sidebar.selectbox('Palavara chave 2', keywords_1)

filme['key_word_2'] = st.sidebar.selectbox('Palavara chave 3', keywords_2)

st.sidebar.markdown(r"""## Atores""")

filme['cast_1'] = st.sidebar.selectbox('Elenco 1', cast_1)

filme['cast_2'] = st.sidebar.selectbox('Elenco 2', cast_2)

st.sidebar.markdown(r"""## Genero """)

generoSelecionadoFilme = st.sidebar.multiselect('Selecionados', generos)

for generoFilme in generoSelecionadoFilme:
    filme[generoFilme] = 1

st.title('Previsão de vendas de ingressos')

cols1, cols2 = st.beta_columns(2)

cols1.markdown(r"""

## Sessão

""")

predictThis = False

if cols2.button('Gerar Projeção'):
    predictThis = True

st.markdown("---")

col1, col2, col3, col4 = st.beta_columns(4)

data_inicio = col1.date_input('De')

data_fim = col2.date_input('Até')

data_range = pd.date_range(start=data_inicio, end=data_fim)

filme['hora'] = col3.time_input('Horáriro', help="Somente a hora sera utilizada").strftime("%H")

filme['salaCinema'] = col4.selectbox('Sala', ['Sala 01', 'Sala 02', 'Sala 03', 'Sala 04'])


sessoes = []
for date in data_range:
    filmeCopy = filme.copy()

    filmeCopy['data'] = date.date().strftime('%Y-%m-%d')

    sessoes.append(filmeCopy)

df = pd.DataFrame(sessoes)


dfPredict = pd.DataFrame([{'data': 0,
                          'hora': 0,
                          'salaCinema': 0,
                          'result': 0}])

if predictThis:

    modelPredictor = ModelDecisiontree()

    decisionTreeRegression = Regression(modelPredictor)

    dfPredict = decisionTreeRegression.predict(df)

    trace = go.Scatter(x = dfPredict['data'], y=dfPredict['result'])

    data = [trace]

    layout = go.Layout(yaxis={'title': 'Vendas de ingressos'},
                       xaxis={'title': 'Dias'},
                       title="Árvore de Decisão")

    fig = go.Figure(data=data, layout=layout)

    st.plotly_chart(fig)

    modelLinear = ModelLinearRegression()

    regressionLinear = Regression(modelLinear)

    dfPredict = regressionLinear.predict(df)

    trace = go.Scatter(x=dfPredict['data'], y=dfPredict['result'])

    data = [trace]

    layout = go.Layout(yaxis={'title': 'Vendas de ingressos'},
                       xaxis={'title': 'Dias'},
                       title="Regressão Linear")

    fig = go.Figure(data=data, layout=layout)

    st.plotly_chart(fig)
