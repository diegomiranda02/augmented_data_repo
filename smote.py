from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

# Arquivo com os dados em formato csv
# Exemplo:
# Content,Label
# "CONTEÚDO DO PROCESSO 1",aprovada
# "CONTEÚDO DO PROCESSO 2",aprovada_com_ressalvas
# "CONTEÚDO DO PROCESSO 3",desaprovada
DATASET = "SUBSTITUIR PELO DATASET EM FORMATO CSV"

df = pd.read_csv(DATASET, sep=',', header=0)

# Função para mapear os tipos de classificação para inteiro (case-insensitive)
def map_tipo_to_int(tipo):
    if 'aprovada' == tipo:
        return 0
    elif 'aprovada_com_ressalvas' == tipo:
        return 1
    elif 'desaprovada' == tipo:
        return 2
    else:
        return tipo
    
# Aplica a função map_tipo_to_int na coluna "Label" e cria uma coluna "label" com os valores em inteiro
df['label'] = df["Label"].apply(map_tipo_to_int)

# Renomeando as colunas "Label" para "label_text" e "Content" para "text"
df = df.rename(columns={'Label': 'label_text', 'Content': 'text'})

# Convertendo as colunas 'text' e 'label' para lista
X = df['text'].tolist() 
y = df['label'].tolist()

print("Número de observações em X (Antes de aplicar SMOTE):", len(X))
print("Número de observações em y (Antes de aplicar SMOTE):", len(y))

# Convertendo os texto para vetores TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(X)

#sampling_strategy = {0: 127, 1: 127, 2: 127}

# Aplicando SMOTE para gerar mais dados (oversampling) com a estratégia de amostragem especificada
# A estratégia de amostragem é gerar sintenticamente observações até chegar à quantidade de 127 para o três labels (0, 1 e 2)
smote = SMOTE(sampling_strategy = {0: 127, 1: 127, 2: 127})
X_resampled, y_resampled = smote.fit_resample(X_tfidf, y)

# Contagem do número de observações em X_resampled e em y_resampled
num_samples = X_resampled.shape[0]
num_labels = len(y_resampled)

print("Número de observações em X_resampled:", num_samples)
print("Número de observações em y_resampled:", num_labels)

# Convertendo os vetores TF-IDF de volta para texto
X_resampled_text = tfidf_vectorizer.inverse_transform(X_resampled)

# Convertendo os texto para uma lista de strings
X_resampled_text_list = [' '.join(doc) for doc in X_resampled_text]

# Criando um dataframe com os dados rebalanceados
df_resampled = pd.DataFrame({'text': X_resampled_text_list, 'label': y_resampled})

# Gravando o dataset em um arquivo .csv sem os índices
df_resampled.to_csv('dataset_rebalanceado.csv', index=False)

# Contagem da quantidade de observações por label
column = 'label'
value_counts = df_resampled[column].value_counts()
print(f"Value counts for '{column}':")
print(value_counts)
