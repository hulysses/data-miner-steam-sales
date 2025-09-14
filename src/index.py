import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# 1. Carregar dados
df = pd.read_csv('steam_sales.csv')

# 2. Pré-processar
df['#Reviews'] = pd.to_numeric(df['#Reviews'], errors='coerce').fillna(0)
df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce')
df = df.dropna(subset=['Price (€)', 'Discount%', 'Rating'])

# 3. Features e target
X = df[['Rating', '#Reviews', 'Discount%', 'Original Price (€)', 'Windows','Linux','MacOS']]
y = df['Price (€)']

# 4. Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5. Modelo baseline
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Avaliação
preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE:", rmse)
