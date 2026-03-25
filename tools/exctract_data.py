import pandas as pd
import json

df = pd.read_csv('./data/Estadisticas.csv')
data = pd.read_csv('./data/Unique_values.csv')

with open('./data/Unique_values_dict.json', 'r') as file:
    unique_dict = json.load(file)

interest_columns = ['Delito', 'SubDelito', 'Victima', 'SubVictima', 'Edad', 'Sexo', 'Nacionalidad', 'Provincia', 'Canton', 'Distrito']
values_dict = {}
unique_types = {}

def extraction_of_columns():

    for column in interest_columns:
        unique_types[column] = df[column].dropna().unique()

    unique_values_list = []

    for column, values in unique_types.items():
        for value in values:
            unique_values_list.append({'Column': column, 'Unique Value': value})

    unique_values_df = pd.DataFrame(unique_values_list)
    unique_values_df.to_csv('unique_values.csv', index=False)

    for column, types in unique_types.items():
        print(f"Tipos en la columna: '{column}': ")
        print(types)
        print()

def extraction_of_values():

    for column in data['Column'].unique():
        values = data[data['Column'] == column]['Unique Value'].tolist()
        values_dict[column] = values

    for key, value in values_dict.items():
        print(f"{key}: {value}")

    with open('../../data/Unique_values_dict.json', 'w') as file:
        json.dump(values_dict, file, indent=4)

