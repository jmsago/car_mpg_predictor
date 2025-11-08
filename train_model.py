import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pickle

# 🔹 Carrega dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
df = pd.read_csv(url).dropna()

# 🔹 Features e alvo
features = ["horsepower", "weight", "acceleration", "model_year"]
X = df[features]
y = df["mpg"]

# 🔹 Divide os dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Treina o modelo
model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# 🔹 Avaliação
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ Modelo treinado com sucesso!")
print(f"MAE: {mae:.2f} | R²: {r2:.2f}")

# 🔹 Salva o modelo
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(features, open("model_features.pkl", "wb"))

# 🔹 Gera gráfico de importância das features
importances = pd.Series(model.feature_importances_, index=features)
plt.figure(figsize=(6, 4))
importances.sort_values().plot(kind='barh', color="#00b4d8")
plt.title("Importância das Variáveis no Modelo")
plt.xlabel("Importância")
plt.tight_layout()
plt.savefig("feature_importance.png")
