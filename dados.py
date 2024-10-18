import pandas as pd
import pyautogui as pg
import plotly.express as px
import time
from selenium import webdriver
import seaborn as sns
import matplotlib as plt
from sklearn.model_selection import train_test_split

# DADOS QUE USEI NESSE CÓDIGO FOI DO ARQUIVO "Compras.csv" QUE ESTÁ UPADO NO MEU GOOGLE DRIVE
tabela = pd.read_csv(r"CAMINHO_DO_ARQUIVO", sep=';', encoding="utf-8")
display(tabela)
print(tabela.info())

display(tabela.describe().round(2))
grafico = px.histogram(tabela, x='Produto', y='Quantidade', text_auto=True)
print('Histórico de Compras: Itens e Quantidade Vendida:')
grafico.show()
print("Histórico de Compras: Total de Vendas de Cada Fornecedor")
px.histogram(tabela, x='Fornecedor', y='Quantidade', text_auto=True).show()

# pg.hotkey('win', 'r')
# pg.typewrite('cmd')
# pg.hotkey('enter')
# pg.typewrite('pip freeze')
# pg.hotkey('enter')
# time.sleep(1)
# pg.hotkey('ctrl', 'a')
# pg.hotkey('ctrl', 'c')
# time.sleep(1)
# pg.hotkey('alt', 'f4')

# navegador = webdriver.Chrome()
# navegador.get('https://www.google.com')

display(tabela)

def mapear_produtos(produto):
  if produto in produto_codigo:
    return produto_codigo[produto]
  else:
    codigo = len(produto_codigo) + 1
    produto_codigo[produto] = codigo
    return codigo
  
def mapear_fornecedor(fornecedor):
  if fornecedor in fornecedor_codigo:
    return fornecedor_codigo[fornecedor]
  else:
    codigo = len(fornecedor_codigo) + 1
    fornecedor_codigo[fornecedor] = codigo
    return codigo

fornecedor_codigo = {}
produto_codigo = {}

# Substituindo produtos e fornecedores por códigos na Tabela
tabela["Produto"] = tabela["Produto"].apply(mapear_produtos)
tabela["Fornecedor"] = tabela["Fornecedor"].apply(mapear_fornecedor)
tabela["Data"] = pd.to_datetime(tabela["Data"], format="mixed")

display(tabela)

display(tabela.corr())
sns.heatmap(tabela.corr(), cmap="Blues", annot=True)

y = tabela["ValorFinal"] # O que quero Prever
columns_to_drop = ["Data", "CódigoCompra", "ValorFinal", "Fornecedor", "Produto"]
x = tabela.drop(columns_to_drop, axis=1) # Baseado nessas Informações

# Train Test Split
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3)

from sklearn.linear_model import LinearRegression # Regressão Linear
from sklearn.ensemble import RandomForestRegressor # Árvore de Decisão

modelo_regressaolinear = LinearRegression()
modelo_arvoredecisao = RandomForestRegressor()

modelo_regressaolinear.fit(x_treino, y_treino)
modelo_arvoredecisao.fit(x_treino, y_treino)

from sklearn.metrics import r2_score

previsao_regressaolinear = modelo_regressaolinear.predict(x_teste)
previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)

print(r2_score(y_teste, previsao_regressaolinear))
print(r2_score(y_teste, previsao_arvoredecisao))

tabela_aux = pd.DataFrame()
tabela_aux["y_teste"] = y_teste
tabela_aux["ArvoreDecisao"] = previsao_arvoredecisao
tabela_aux["RegressaoLinear"] = previsao_regressaolinear

sns.lineplot(data=tabela_aux)

# Para fazer novas previsões
tabela_nova = pd.read_excel(r"CAMINHO_DO_ARQUIVO_QUE_VOCÊ_QUER_PREVER")
display(tabela_nova)

previsao = modelo_arvoredecisao.predict(tabela_nova)
print(previsao)
