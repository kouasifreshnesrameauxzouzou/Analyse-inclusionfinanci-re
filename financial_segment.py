pandas==1.3.3
streamlit==1.10.0
scikit-learn==0.24.2
plotly==4.14.3

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px
data= pd.read_csv('Financial_inclusion_dataset.csv')
data.head()
data.describe()

fig1 = px.histogram(data, x="age_of_respondent", color="gender_of_respondent", title="Distribution des âges par genre")

fig2 = px.pie(data, names="gender_of_respondent", title="Répartition des genres")
fig3 = px.bar(data, x="relationship_with_head", color="marital_status", title="Relation avec le chef du ménage par statut matrimonial")

fig4 = px.bar(data, x="marital_status", color="gender_of_respondent", title="Statut matrimonial par genre")
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

categorical_columns = ['country', 'uniqueid', 'bank_account', 'location_type', 'cellphone_access',
                        'gender_of_respondent', 'relationship_with_head', 'marital_status', 'education_level', 'job_type']

for column in categorical_columns:
    data[column] = label_encoder.fit_transform(data[column])
# Séparation des données en ensembles d'entraînement et de test
X = data.drop("bank_account", axis=1)
y = data["bank_account"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement du modèle de régression logistique
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
# Prédictions avec le modèle de régression logistique
logistic_predictions = logistic_model.predict(X_test)
# Évaluation du modèle de régression logistique
print("Régression Logistique:")
accuracy_logistic = accuracy_score(y_test, logistic_predictions)
print(f"Précision : {accuracy_logistic:.2f}")
print("Rapport de classification :\n", classification_report(y_test, logistic_predictions))

# Styles personnalisés
st.markdown(
    """
    <style>
        .st-eb {
            background-color: #3498db !important;
            color: #ffffff !important;
            border-radius: 5px;
        }
        .st-df {
            background-color: #2ecc71 !important;
            color: #ffffff !important;
            border-radius: 5px;
        }
        .st-df:hover {
            background-color: #27ae60 !important;
        }
        .st-dg {
            background-color: #e74c3c !important;
            color: #ffffff !important;
            border-radius: 5px;
        }
        .st-dg:hover {
            background-color: #c0392b !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit
st.title("Analyse de l'inclusion financière")

# Boutons stylisés
if st.button("Afficher l'histogramme des âges par genre", key="btn_hist"):
    st.plotly_chart(fig1)

if st.button("Afficher la répartition des genres", key="btn_pie"):
    st.plotly_chart(fig2)

if st.button("Afficher la relation avec le chef du ménage par statut matrimonial", key="btn_bar1"):
    st.plotly_chart(fig3)

if st.button("Afficher le statut matrimonial par genre", key="btn_bar2"):
    st.plotly_chart(fig4)

# Transformation des données
label_encoder = LabelEncoder()
for column in categorical_columns:
    data[column] = label_encoder.fit_transform(data[column])

# Affichage des données
st.write("Données transformées:", data.head())

# Séparation des données en ensembles d'entraînement et de test
# (Votre code existant)

# Entraînement du modèle
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Prédictions
logistic_predictions = logistic_model.predict(X_test)

# Évaluation du modèle
st.subheader("Évaluation du modèle de régression logistique")
st.write("Régression Logistique:")
accuracy_logistic = accuracy_score(y_test, logistic_predictions)
st.write(f"Précision : {accuracy_logistic:.2f}")
st.write("Rapport de classification :\n", classification_report(y_test, logistic_predictions))
