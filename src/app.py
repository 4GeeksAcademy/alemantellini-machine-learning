from utils import db_connect
engine = db_connect()

def main():

# your code here
# IMPORTAR LAS LIBRERÍAS
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.feature_selection import SelectKBest, f_regression
    from sklearn.preprocessing import MinMaxScaler

# 1. Carga del dataset
    total_data = pd.read_csv('/workspaces/alemantellini-machine-learning/data/raw/AB_NYC_2019.csv')

# 2. Limpieza
    total_data = total_data.drop_duplicates(subset=total_data.columns.difference(['id']))

# 3. Elimino las columnas que decidí eliminar
    total_data = total_data.drop(columns=[
        'last_review','reviews_per_month',
        'host_name','name', 'latitude', 'longitude'
        ])
    total_data = total_data.dropna().reset_index(drop=True)

# 4. Factorización
    total_data['neighbourhood_group_factor'] = pd.factorize(total_data['neighbourhood_group'])[0]
    total_data['neighbourhood_factor'] = pd.factorize(total_data['neighbourhood'])[0]
    total_data['room_type_factor'] = pd.factorize(total_data['room_type'])[0]

# 5. Escalado
    variables = [
        'minimum_nights', 
        'number_of_reviews', 
        'calculated_host_listings_count',
        'neighbourhood_group_factor', 
        'neighbourhood_factor', 
        'room_type_factor', 
        'availability_365'
        ]

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(total_data[variables])
    df_scaler = pd.DataFrame(scaled, columns = variables, index = total_data.index)
    df_scaler["price"] = total_data["price"] 

# 6. Train / Test split
    mi_x = df_scaler.drop(columns=["price"])
    mi_y = df_scaler["price"]

    X_train, X_test, y_train, y_test = train_test_split(mi_x, mi_y, test_size=0.2, random_state=42)

# 6.1 Feature selection
    k_best = SelectKBest(score_func=f_regression, k=4)
    k_best.fit(X_train, y_train)

    selected_columns = X_train.columns[k_best.get_support()]

    X_train_sel = X_train[selected_columns]
    X_test_sel = X_test[selected_columns]

# 7. Guardar datos procesados
    train_processed = pd.concat([X_train_sel, y_train.reset_index(drop=True)], axis=1)
    test_processed = pd.concat([X_test_sel, y_test.reset_index(drop=True)], axis=1)

    train_processed.to_csv("data/processed/train_processed.csv", index=False)
    test_processed.to_csv("data/processed/test_processed.csv", index=False)

print("Procesamiento completado correctamente.")

if __name__ == "__main__":
    main()
