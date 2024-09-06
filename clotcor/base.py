import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class Prediccion:
    def __init__(self):
        self.data = pd.read_csv("./data/Estadisticas.csv")
        self.le_dict = {}
        self.rf_model = None
        self.features = ['SubDelito', 'Hora', 'Victima', 'SubVictima', 'Edad', 'Sexo', 'Nacionalidad', 'Provincia', 'Canton', 'Distrito', 'DiaSemana', 'Mes']
        
    def preprocess_data(self):
        self.data['Fecha'] = pd.to_datetime(self.data['Fecha'])
        self.data['DiaSemana'] = self.data['Fecha'].dt.dayofweek
        self.data['Mes'] = self.data['Fecha'].dt.month

        self.data['Hora'] = self.data['Hora'].apply(lambda x: int(x.split(':')[0]))
        categorical_columns = ['Delito', 'SubDelito', 'Victima', 'SubVictima', 'Edad', 'Sexo', 'Nacionalidad', 'Provincia', 'Canton', 'Distrito']
        for col in categorical_columns:
            self.le_dict[col] = LabelEncoder()
            self.data[col] = self.le_dict[col].fit_transform(self.data[col].astype(str))
        return self.data
    
    def split_data(self):
        X = self.data[self.features]
        y = self.data['Delito']
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self):
        X_train, X_test, y_train, y_test = self.split_data()
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rf_model.fit(X_train, y_train)
        return X_test, y_test
    
    def evaluate_model(self, X_test, y_test):
        y_pred = self.rf_model.predict(X_test)
        print(classification_report(y_test, y_pred))
        self.plot_confusion_matrix(y_test, y_pred)
        self.plot_feature_importance()

    def plot_confusion_matrix(self, y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10,8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Matriz de Confusión')
        plt.xlabel('Predicción')
        plt.ylabel('Valor Real')
        plt.show()

    def plot_feature_importance(self):
        feature_importance = pd.DataFrame({'feature': self.features, 'importance': self.rf_model.feature_importances_})
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        plt.figure(figsize=(10,6))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title('Importancia de las Características')
        plt.show()

    def predict_crime(self, input_data):
        prediction = self.rf_model.predict(input_data)
        probabilities = self.rf_model.predict_proba(input_data)
        return prediction, probabilities

    def run(self):
        self.preprocess_data()
        X_test, y_test = self.train_model()
        self.evaluate_model(X_test, y_test)

    def safe_transform(self, le, values):
        # Función para manejar valores no vistos en el LabelEncoder
        new_values = np.array(values)
        for unique_item in np.unique(values):
            if unique_item not in le.classes_:
                le.classes_ = np.append(le.classes_, unique_item)
        return le.transform(new_values)

    def predict_new_data(self, new_data):
        if isinstance(new_data, dict):
            new_data = pd.DataFrame([new_data])
        
        encoded_data = pd.DataFrame(columns=self.features)
        for col in self.features:
            if col in new_data.columns:
                if col in self.le_dict:
                    # Usar el método safe_transform para manejar nuevas categorías
                    encoded_data[col] = self.safe_transform(self.le_dict[col], new_data[col].astype(str))
                else:
                    encoded_data[col] = new_data[col]
            else:
                encoded_data[col] = 0  # O algún valor por defecto apropiado

        prediction, probabilities = self.predict_crime(encoded_data)
        print(f"Delito predicho: {self.le_dict['Delito'].inverse_transform(prediction)[0]}")
        print("Probabilidades de cada tipo de delito:")
        for crime, prob in zip(self.le_dict['Delito'].classes_, probabilities[0]):
            print(f"{crime}: {prob:.2f}")