# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 11:51:27 2020

O objetivo deste projeto é fornecer um serviço automatizado que recomenda leads para um
usuário dado sua atual lista de clientes (Portfólio).

Vamos então utilizar os 3 portfolios que temos para recomendar outras empresas do MERCADO.

Para cada empresa/portfolio, após o modelo treinado, vamos ler os respectivos dados e identificar na tabela MERCADO novas oportunidades.

@author: Armstrong
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 20:51:41 2020

@author: Armstrong
"""

#importações----------------------------------------------------------------------------------

print ("--------------------------------------------------------------------------")
print ("Importações - OK")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')

#analise dos dados
from pandas_profiling import ProfileReport

#from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors



#leitura dos dados brutos----------------------------------------------------------------------------------

print ("--------------------------------------------------------------------------")
print ("Leitura dos dados brutos - OK")

portfolio1_codenation = pd.read_csv('E:/codenation/dados/estaticos_portfolio1.csv')
portfolio2_codenation = pd.read_csv('E:/codenation/dados/estaticos_portfolio2.csv')
portfolio3_codenation = pd.read_csv('E:/codenation/dados/estaticos_portfolio3.csv')
mercado_codenation = pd.read_csv('E:/codenation/dados/estaticos_market.csv/estaticos_market.csv')

#guardando os ids para utilizarmos no final (excluir sugestões que já existem)
portfolio1_codenation = portfolio1_codenation.filter(['id'])
portfolio2_codenation = portfolio2_codenation.filter(['id'])
portfolio3_codenation = portfolio3_codenation.filter(['id'])


#MERGE
print ("--------------------------------------------------------------------------")
print ("Recupera o ID de cada linha das tabelas porfolios para realizarmos em seguida o MERGE - OK")

portfolio_empresa1 = portfolio1_codenation['id']
portfolio_empresa2 = portfolio2_codenation['id']
portfolio_empresa3 = portfolio3_codenation['id']


#retirei primeiro as colunas que não tem a menor lógica com o que buscamos. Fiz manualmente analisando as tabelas e com o panda profile
#retiramos algumas colunas onde já existem outras colunas que foram desmembradas em categorias como: nu_meses_rescencia','idade_empresa_anos

#analise dos dados - feita uma vez apenas
#profile = ProfileReport(mercado_codenation, title='Leads')
#profile.to_file("your_report2geral.html")


mercado_codenation.drop(columns=[
'idade_maxima_socios','qt_socios', 'qt_socios','qt_socios_coligados','qt_socios_coligados',
'qt_socios_pep', 'qt_socios_pep', 'qt_socios_pf',	'qt_socios_pf', 'qt_socios_pj',	'qt_socios_pj',
'qt_socios_pj_ativos',	'qt_socios_pj_ativos', 'qt_socios_pj_baixados',	'qt_socios_pj_baixados',
'qt_socios_pj_inaptos',	'qt_socios_pj_inaptos', 'qt_socios_pj_nulos',	'qt_socios_pj_nulos',
'qt_socios_pj_suspensos',	'qt_socios_pj_suspensos', 'qt_socios_st_regular',	'qt_socios_st_regular',
'qt_socios_st_suspensa',	'qt_socios_st_suspensa','vl_total_tancagem','vl_total_tancagem',
'vl_total_tancagem_grupo','vl_total_tancagem_grupo', 'vl_total_veiculos_antt',	'vl_total_veiculos_antt',
'vl_total_veiculos_antt_grupo',	'vl_total_veiculos_antt_grupo', 'vl_total_veiculos_leves',	'vl_total_veiculos_leves',
'vl_total_veiculos_leves_grupo','vl_total_veiculos_leves_grupo', 'vl_total_veiculos_pesados',	'vl_total_veiculos_pesados',
'vl_total_veiculos_pesados_grupo',     'idade_media_socios','idade_media_socios', 'idade_minima_coligadas','idade_minima_coligadas',
'idade_minima_socios','idade_minima_socios','vl_total_veiculos_antt','nu_meses_rescencia','idade_empresa_anos','dt_situacao',
'fl_email','fl_telefone','Unnamed: 0','idade_ate_18','idade_acima_de_58', 'idade_de_54_a_58',
'idade_de_49_a_53','idade_de_44_a_48','idade_de_19_a_23','idade_de_39_a_43','idade_de_34_a_38',
'idade_de_24_a_28','idade_de_29_a_33','qt_ex_funcionarios','qt_socios_feminino','qt_socios_masculino',
'qt_admitidos','tx_rotatividade','qt_desligados_12meses','qt_desligados','qt_admitidos_12meses',
'qt_funcionarios_12meses','meses_ultima_contratacaco','qt_funcionarios_24meses',
'qt_alteracao_socio_total','qt_alteracao_socio_365d','qt_alteracao_socio_90d',
'qt_alteracao_socio_180d'],axis=1,inplace=True)


#removendo colunas que estão completamente sem valores. < 1% de valores faltantes 
print ("--------------------------------------------------------------------------")
print ("Filtra dados das planilhas que tenha colunas com menos de 1% de miss - OK")

missing_data_summary = pd.DataFrame({'columns': mercado_codenation.columns, 'types': mercado_codenation.dtypes, 
                                     'missing(%)': mercado_codenation.isnull().mean().round(4) * 100,   'unicos' : mercado_codenation.nunique()})
missing_data_summary.sort_values(by=['missing(%)'], inplace=True)

missing_data_summary_zerado  = missing_data_summary.loc[missing_data_summary['missing(%)'] < 1]['columns']

#filtra as colunas com menos de 1% de dados faltantes
mercado_codenation_filtro1 = mercado_codenation.filter(missing_data_summary_zerado.values)

#drop as linhas sem dados ( como é menos de 1%, optei pro dropar e não inputar dados)
mercado_codenation_filtro1 = mercado_codenation_filtro1.dropna()

#verificar se restou alguma coisa sem dados, mas ta OK
missing_data_summary_filtro1 = pd.DataFrame({'columns': mercado_codenation_filtro1.columns, 'types': mercado_codenation_filtro1.dtypes, 
                                     'missing(%)': mercado_codenation_filtro1.isnull().mean().round(4) * 100,   'unicos' : mercado_codenation_filtro1.nunique()})


#Análise dos dados

print ("--------------------------------------------------------------------------")
print ('Dados únicos - Analisando todas as empresas - OK')
print ('de_natureza_juridica')
print (mercado_codenation_filtro1['de_natureza_juridica'].nunique())
print (mercado_codenation_filtro1.groupby('de_natureza_juridica').sum().sort_values(by=['qt_filiais']))
print ('natureza_juridica_macro')
print (mercado_codenation_filtro1['natureza_juridica_macro'].nunique())
print (mercado_codenation_filtro1.groupby('natureza_juridica_macro').sum().sort_values(by=['qt_filiais']))
print ('de_ramo')
print (mercado_codenation_filtro1['de_ramo'].nunique())
print (mercado_codenation_filtro1.groupby('de_ramo').sum().sort_values(by=['qt_filiais']))
print ('setor')
print (mercado_codenation_filtro1['setor'].nunique())
print (mercado_codenation_filtro1.groupby('setor').sum().sort_values(by=['qt_filiais']))
print ('nm_divisao')
print (mercado_codenation_filtro1['nm_divisao'].nunique())
print (mercado_codenation_filtro1.groupby('nm_divisao').sum().sort_values(by=['qt_filiais']))
print ('nm_segmento')
print (mercado_codenation_filtro1['nm_segmento'].nunique())
print (mercado_codenation_filtro1.groupby('nm_segmento').sum().sort_values(by=['qt_filiais']))

print ("--------------------------------------------------------------------------")
print ('Tipos de dados da tabela da Codenation - Temos valores booleanos/INT/FLOAT/Strings(objets) OK')
print (mercado_codenation_filtro1.dtypes.unique().size)
print (mercado_codenation_filtro1.dtypes.unique())
print (mercado_codenation_filtro1.dtypes)



#transformando com getdummies os dados texto em numerico
print ("--------------------------------------------------------------------------")
print ("Transformando dados booleanos e texto para númerico com dummies - OK")

texto_data_summary = pd.DataFrame({'columns': mercado_codenation_filtro1.columns,'TEXTO': mercado_codenation_filtro1.dtypes})

#filtro so as colunas com object e bool
texto_data_summary = texto_data_summary.loc[(texto_data_summary['TEXTO'] == 'object') | (texto_data_summary['TEXTO'] == 'bool')]['columns']
#texto_data_summary = texto_data_summary.loc[(texto_data_summary['TEXTO'] == 'bool')]['columns']

texto_data_summary = texto_data_summary.to_frame().reset_index()

#transforma em uma lista. Vamos usar estes argumentos para repassar diretament o get_dummies. 
texto_data_summary = texto_data_summary['columns'].astype(str).values.flatten().tolist()

#tem que retirar o ID da lista, já que não tem logica rodar ele pro getdummies
texto_data_summary.remove('id')

#executa get_dummies para todas as variaveis booleanas e de objetos
mercado_codenation_filtro1 = pd.concat([pd.get_dummies(mercado_codenation_filtro1, columns=texto_data_summary)],axis=1)


#merge já com dummies
#complementa dados das tabelas de portfolio a partir da tabela de mercado usando para isso o campo ID
print ("--------------------------------------------------------------------------")
print ("Complementa os dados das empresas de portfolio - Faz um merge pelo ID - OK")
portfolio_empresa1 = pd.merge(portfolio_empresa1, mercado_codenation_filtro1, on='id')
portfolio_empresa2 = pd.merge(portfolio_empresa2, mercado_codenation_filtro1, on='id')
portfolio_empresa3 = pd.merge(portfolio_empresa3, mercado_codenation_filtro1, on='id')


#confere se ficaram idêntica no número de colunas
print ("--------------------------------------------------------------------------")
print ('Número de linhas e colunas - OK')
print (portfolio_empresa1.shape)
print (portfolio_empresa2.shape)
print (portfolio_empresa3.shape)
print (mercado_codenation_filtro1.shape)


print ("--------------------------------------------------------------------------")
print ("Executar modelo - OK")

#dropar o id, já que é texto e nao é pra influenciar no modelo
mercado_codenation_filtro1.drop('id', axis=1, inplace=True)
portfolio_empresa1.drop('id', axis=1, inplace=True)
portfolio_empresa2.drop('id', axis=1, inplace=True)
portfolio_empresa3.drop('id', axis=1, inplace=True)


qtd_neighbors = 5
nearest = NearestNeighbors(n_neighbors=qtd_neighbors,metric = 'cosine')

nearest.fit(mercado_codenation_filtro1)

print(nearest)

#procurando por sugestões
print ("--------------------------------------------------------------------------")
print ("Procurando por sugestoes para o portfolio 3 - OK")


neighbors_list = {}

for row in range(portfolio_empresa3.shape[0]):
    print(row)
    neighbors_list[row] = nearest.kneighbors(portfolio_empresa3.iloc[[row]].values)
    
list_size = len(neighbors_list)
num_neighbors = len(neighbors_list[0][1][0])

neighbors_idx_array = neighbors_list[0][1][0]
neighbors_distance_array = neighbors_list[0][0][0]
np.delete(neighbors_idx_array,[0,1])
np.delete(neighbors_distance_array,[0,1])

for line in range(1,list_size):
    neighbors_idx_array = np.concatenate((neighbors_idx_array,neighbors_list[line][1][0]),axis=None)
    neighbors_distance_array = np.concatenate((neighbors_distance_array,neighbors_list[line][0][0]),axis = None)
                                                                                       
                                                                                       
                                                                                       
if len(neighbors_idx_array) != list_size*num_neighbors:
    print ("ERROR: Check array size.")                                                                                       
    
#temos agora um array unidimensional com os indices dos clientes recomendados


print (neighbors_distance_array[0])

dicio = {}

for idx,ind in zip(neighbors_idx_array, range(len(neighbors_idx_array))):
    dicio[ind] = (portfolio_empresa3.iloc[int(ind/qtd_neighbors)].name, mercado_codenation_filtro1.iloc[idx].name, (neighbors_distance_array[ind]))
    

#recupera os indices das sugestões    
neig_df = pd.DataFrame.from_dict(dicio,orient='index')   
neig_df.rename(columns={0:'id_origin',1:'id',2:'distance'},inplace=True)
neig_df.set_index('id',inplace=True)


#faz o merge com a tabela inicial para montar os dados a serem entregues
sugestoes = neig_df.merge(mercado_codenation,how='inner',left_index=True,right_index=True)

#remove as duplicadas
sugestoes.drop_duplicates(subset ="id",keep = False, inplace = True)

#remove as que já são clientes e deixa só as novas indicações
mergedStuff = pd.merge(sugestoes, portfolio3_codenation , on=['id'], how='inner')
sugestoes_finais = sugestoes.merge(mergedStuff['id'],left_on='id', right_on='id',how='left',indicator='i').query('i == "left_only"')
sugestoes_finais.drop(columns=['i'],inplace=True)


print ("--------------------------------------------------------------------------")
print ("Exportar csv - OK")
sugestoes_finais.to_csv('sugestoes_finais.csv')
#mercado_codenation.to_csv('mercado_codenation.csv')

print ("--------------------------------------------------------------------------")
print ("Finalizado com sucesso - OK")
