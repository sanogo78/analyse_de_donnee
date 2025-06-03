import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import prince
import numpy as np

# netoyage des données
credit = pd.read_csv("credit.csv", sep=";")
# Affichage des premières lignes
print("\tAperçus du jour\n")
print(credit.head())


#type de données
print("\n\tTypes de données\n")
print(credit.dtypes)
# Analyse des valeurs manquantes
credit.isnull().sum()
# Description de la base de données
print("\t\n Description de la base de données\n")
credit.describe(include='all')
# Analyse des doublons
# Vérification des doublons
duplicates = credit.duplicated().sum()
print("\n\tNombre de ligne dupliquée est : ", duplicates)
# La variable age est de type quantitative
# Transformation de la variable catégorielles age en qualitative

# Definir les bornes pour les classes d'âge
bins = [19, 29, 39, 49, 59, 69, 79, 100]
# Definir les étiquettes pour les classes d'âge
labels = ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']
# Créer une nouvelle colonne 'age_group' avec les classes d'âge
credit['ages_group'] = pd.cut(credit['Age'], bins=bins, labels=labels, right=False)
# Afficher les premières lignes pour vérifier la nouvelle colonne
print("\n\tAperçus du jour avec la nouvelle colonne 'ages_group'\n")
print(credit.head())
# Affichage des variables catégorielles
# Liste des variables catégorielles
categorical_columns = ['Marche', 'Apport', 'Impaye', 'Assurance', 'Endettement', 'Famille', 'Enfants', 'Logement', 'Profession', 'Intitule']

# Création des graphiques pour chaque variable catégorielle en terme de proportion
for column in categorical_columns:
    plt.figure(figsize=(10, 5))
    proportions = credit[column].value_counts(normalize=True) * 100
    sns.barplot(x=proportions.index, y=proportions.values)
    plt.title(f"Diagramme en barres pour la variable {column} (proportions)")
    plt.ylabel('Proportion (%)')
    plt.show()
    # Liste des variables catégorielles
categorical_columsns = ['Marche', 'Apport', 'Impaye', 'Assurance', 'Endettement', 'Famille', 'Enfants', 'Logement', 'Profession', 'Intitule']
for columns in categorical_columns:
    print(f"\nFréquences de la variable {columns}:")
    print(credit[columns].value_counts())
    # Créer des graphiques pour les variables catégorielles
for columns in categorical_columns:
    plt.figure(figsize=(10, 5))
    sns.countplot(x=columns, data=credit)
    plt.title(f"Diagramme en barres pour la variable {columns}")
    plt.show()
    # Créer des graphiques camemberts pour les variables catégorielles
for columns in categorical_columns:
    plt.figure(figsize=(10, 6))
    credit[columns].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
    plt.title(f"Diagramme camembert pour la variable : {columns}")
    plt.ylabel('')
    plt.show()
    # Liste des variables qualitatives
categorical_columns = ['Marche', 'Apport', 'Impaye', 'Assurance', 'Endettement', 'Famille', 'Enfants', 'Logement', 'Profession', 'Intitule']
# Créer des graphiques bivariés pour chaque paire de variables qualitatives
for col1 in categorical_columns:
    for col2 in categorical_columns:
        if col1 != col2:
            # Calculer les proportions pour le graphique bivarié
            proportions = credit.groupby([col1, col2]).size().reset_index(name='Count')
            proportions['Proportion'] = proportions['Count'] / proportions.groupby(col1)['Count'].transform('sum') * 100

            # Créer un graphique en barres empilées pour la paire de variables
            plt.figure(figsize=(10, 5))
            sns.barplot(x=col1, y='Proportion', hue=col2, data=proportions)
            plt.title(f"Graphique bivarié entre {col1} et {col2} (proportions)")
            plt.ylabel('Proportion (%)')
            plt.show()
            
            # Initialiser les DataFrame pour les coefficients de Cramér et les p-values
cramer_v_df = pd.DataFrame(index=categorical_columns, columns=categorical_columns)
p_value_df = pd.DataFrame(index=categorical_columns, columns=categorical_columns)
tschuprow_t_df = pd.DataFrame(index=categorical_columns, columns=categorical_columns)

# Calculer le test de chi-deux pour chaque paire de variables qualitatives
for i, column1 in enumerate(categorical_columns):
    for j, column2 in enumerate(categorical_columns):
        if column1 != column2:
            contingency_table = pd.crosstab(credit[column1], credit[column2])
            chi2, p, dof, expected = chi2_contingency(contingency_table)
            cramer_v = np.sqrt(chi2 / (credit.shape[0] * (min(contingency_table.shape) - 1)))
            tschuprow_t = cramer_v * np.sqrt((contingency_table.shape[0] - 1) * (contingency_table.shape[1] - 1) / (credit.shape[0] - 1))
            cramer_v_df.loc[column1, column2] = cramer_v
            tschuprow_t_df.loc[column1, column2] = tschuprow_t
            p_value_df.loc[column1, column2] = p

# Afficher la DataFrame des p-values
print("\nDataFrame des p-values :")
p_value_df.style.set_properties(**{'border-color': 'black', 'border-width': '2px', 'border-style': 'solid'})

# Afficher la DataFrame des coefficients de Cramér
print("DataFrame des coefficients de Cramér :")
cramer_v_df.style.background_gradient(cmap='Greens', high=0.4, low=0).set_properties(**{'border-color': 'black', 'border-width': '1px', 'border-style': 'solid'})
# Afficher la DataFrame des coefficients de Tchuprow
print("\nDataFrame des coefficients de Tchuprow :")
tschuprow_t_df.style.background_gradient(cmap='Greens', high=0.4, low=0).set_properties(**{'border-color': 'black', 'border-width': '1px', 'border-style': 'solid'})

#  Créer et ajuster l'ACM
acm = prince.MCA(
    n_components=2,
    n_iter=10,
    copy=True,
    check_input=True,
    engine='sklearn', 
    random_state=42
)
acm = acm.fit(tableau_disjonctif)

#  Affichage des coordonnées des colonnes (modalités)
print("\n\tCoordonnées des colonnes (modalités)\n")
coords_col = acm.column_coordinates(tableau_disjonctif)
print(coords_col)

#  Affichage des coordonnées des lignes (individus)
print("\n\tCoordonnées des lignes (individus)\n")
coords_row = acm.row_coordinates(tableau_disjonctif)
print(coords_row)

#  Visualisation des modalités
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_title("Nuage des modalités - ACM")
for i, label in enumerate(coords_col.index):
    x, y = coords_col.iloc[i, 0], coords_col.iloc[i, 1]
    ax.scatter(x, y, color='blue')
    ax.text(x, y, label, fontsize=9)
ax.axhline(0, color='gray', linestyle='--')
ax.axvline(0, color='gray', linestyle='--')
plt.xlabel("Axe 1")
plt.ylabel("Axe 2")
plt.grid(True)
plt.show()

# Visualisation des profils lignes (individus)
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_title("Graphique - Profils lignes")

for i, label in enumerate(coords_row.index):
    x, y = coords_row.iloc[i, 0], coords_row.iloc[i, 1]
    ax.scatter(x, y, color='orange', s=10)
    ax.text(x, y, str(label), fontsize=7) 

ax.axhline(0, color='gray', linestyle='--')
ax.axvline(0, color='gray', linestyle='--')
plt.xlabel("Axe 1")
plt.ylabel("Axe 2")
plt.grid(True)
plt.show()

# Visualisation des profils colonnes (modalités)
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_title("Graphique - Profils colonnes")

for i, label in enumerate(coords_col.index):
    x, y = coords_col.iloc[i, 0], coords_col.iloc[i, 1]
    ax.scatter(x, y, color='red', s=10)
    ax.text(x, y, str(label), fontsize=7) 

ax.axhline(0, color='gray', linestyle='--')
ax.axvline(0, color='gray', linestyle='--')
plt.xlabel("Axe 1")
plt.ylabel("Axe 2")
plt.grid(True)
plt.show()