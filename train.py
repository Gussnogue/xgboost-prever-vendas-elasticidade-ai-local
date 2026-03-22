import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
import joblib
import os
import kagglehub

# Baixar dataset (BigMart Sales)
path = kagglehub.dataset_download("brijbhushannanda1979/bigmart-sales-data")
df = pd.read_csv(os.path.join(path, "Train.csv"))

# Pré-processamento
# Selecionar features relevantes
features = ['Item_Weight', 'Item_Fat_Content', 'Item_Visibility',
            'Item_Type', 'Item_MRP', 'Outlet_Identifier',
            'Outlet_Establishment_Year', 'Outlet_Size',
            'Outlet_Location_Type', 'Outlet_Type']
target = 'Item_Outlet_Sales'

df = df[features + [target]].copy()

# Tratar missing values
df['Item_Weight'] = df['Item_Weight'].fillna(df['Item_Weight'].median())
df['Outlet_Size'] = df['Outlet_Size'].fillna('Medium')

# Codificar variáveis categóricas
le_dict = {}
for col in ['Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 'Outlet_Size',
            'Outlet_Location_Type', 'Outlet_Type']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Separar X e y
X = df.drop(columns=[target])
y = df[target]

# Dividir treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar XGBoost
model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# Avaliação
print(f"Train R2: {model.score(X_train, y_train):.3f}")
print(f"Test R2: {model.score(X_test, y_test):.3f}")

# Salvar modelo e pré-processadores
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/xgboost_model.pkl")
joblib.dump(le_dict, "models/label_encoders.pkl")

# Salvar nomes das features e valores de referência (para interface)
feature_names = X.columns.tolist()
joblib.dump(feature_names, "models/feature_names.pkl")
print("Modelo salvo em models/xgboost_model.pkl")

