import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import prince
from scipy.stats import chi2_contingency

# Chargement et nettoyage des données
credit = pd.read_csv("credit.csv", sep=";")
print("\nAperçu des données :")
print(credit.head())

# Types de données
print("\nTypes de données :")
print(credit.dtypes)

# Valeurs manquantes
print("\nValeurs manquantes :")
print(credit.isnull().sum())

# Doublons
duplicates = credit.duplicated().sum()
print("\nNombre de lignes dupliquées :", duplicates)

# Transformation de l'âge en classe
bins = [19, 29, 39, 49, 59, 69, 79, 100]
labels = ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']
credit['ages_group'] = pd.cut(credit['Age'], bins=bins, labels=labels, right=False)
print("\nDonnées avec 'ages_group' :")
print(credit[['Age', 'ages_group']].head())

# Variables catégorielles
categorical_columns = ['Marche', 'Apport', 'Impaye', 'Assurance', 'Endettement', 'Famille', 'Enfants', 'Logement', 'Profession', 'Intitule']

# Diagrammes en barres (proportions)
for col in categorical_columns:
    plt.figure(figsize=(10, 5))
    proportions = credit[col].value_counts(normalize=True) * 100
    sns.barplot(x=proportions.index, y=proportions.values)
    plt.title(f"Proportions pour la variable {col}")
    plt.ylabel("Proportion (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Fréquences simples
for col in categorical_columns:
    print(f"\nFréquences de {col} :")
    print(credit[col].value_counts())

# Diagrammes camembert
for col in categorical_columns:
    plt.figure(figsize=(8, 6))
    credit[col].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
    plt.title(f"Répartition de {col}")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()

# Graphiques bivariés
for i, col1 in enumerate(categorical_columns):
    for j, col2 in enumerate(categorical_columns):
        if i < j:  # éviter les doublons
            grouped = credit.groupby([col1, col2]).size().reset_index(name='Count')
            grouped['Proportion'] = grouped['Count'] / grouped.groupby(col1)['Count'].transform('sum') * 100
            plt.figure(figsize=(10, 5))
            sns.barplot(x=col1, y='Proportion', hue=col2, data=grouped)
            plt.title(f"Relation entre {col1} et {col2}")
            plt.ylabel("Proportion (%)")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

# Matrices pour les coefficients d'association
cramer_v_df = pd.DataFrame(index=categorical_columns, columns=categorical_columns)
tschuprow_t_df = pd.DataFrame(index=categorical_columns, columns=categorical_columns)
p_value_df = pd.DataFrame(index=categorical_columns, columns=categorical_columns)

for i in categorical_columns:
    for j in categorical_columns:
        if i != j:
            table = pd.crosstab(credit[i], credit[j])
            chi2, p, dof, expected = chi2_contingency(table)
            n = credit.shape[0]
            cramer_v = np.sqrt(chi2 / (n * (min(table.shape)-1)))
            tschuprow_t = cramer_v * np.sqrt((table.shape[0] - 1) * (table.shape[1] - 1) / (n - 1))
            cramer_v_df.loc[i, j] = round(cramer_v, 3)
            tschuprow_t_df.loc[i, j] = round(tschuprow_t, 3)
            p_value_df.loc[i, j] = round(p, 4)

print("\nP-values :")
print(p_value_df)
print("\nCramér's V :")
print(cramer_v_df)
print("\nTschuprow's T :")
print(tschuprow_t_df)

# Analyse des correspondances multiples (ACM)
# Création du tableau disjonctif
tableau_disjonctif = pd.get_dummies(credit[categorical_columns], drop_first=False)

# Ajustement de l'ACM
acm = prince.MCA(n_components=2, n_iter=10, copy=True, engine='sklearn', random_state=42)
acm = acm.fit(tableau_disjonctif)

# Coordonnées colonnes (modalités)
coords_col = acm.column_coordinates(tableau_disjonctif)
print("\nCoordonnées des modalités :")
print(coords_col)

# Coordonnées lignes (individus)
coords_row = acm.row_coordinates(tableau_disjonctif)
print("\nCoordonnées des individus :")
print(coords_row)

# Graphique modalités
plt.figure(figsize=(10, 8))
plt.title("Nuage des modalités - ACM")
plt.scatter(coords_col[0], coords_col[1], c='blue')
for i, label in enumerate(coords_col.index):
    plt.text(coords_col.iloc[i, 0], coords_col.iloc[i, 1], str(label), fontsize=8)
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.xlabel("Axe 1")
plt.ylabel("Axe 2")
plt.grid(True)
plt.show()

# Graphique individus
plt.figure(figsize=(10, 8))
plt.title("Profils lignes - ACM")
plt.scatter(coords_row[0], coords_row[1], c='orange', s=10)
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.xlabel("Axe 1")
plt.ylabel("Axe 2")
plt.grid(True)
plt.show()

# Graphique modalités (profils colonnes)
plt.figure(figsize=(10, 8))
plt.title("Profils colonnes - ACM")
plt.scatter(coords_col[0], coords_col[1], c='red', s=10)
for i, label in enumerate(coords_col.index):
    plt.text(coords_col.iloc[i, 0], coords_col.iloc[i, 1], str(label), fontsize=7)
plt.axhline(0, color='gray', linestyle='--')
plt.axvline(0, color='gray', linestyle='--')
plt.xlabel("Axe 1")
plt.ylabel("Axe 2")
plt.grid(True)
plt.show()
