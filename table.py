import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from lime.lime_tabular import LimeTabularExplainer

data = pd.read_csv('Air.csv')

print("Classes únicas de 'Air Quality':", data['Air Quality'].unique())

label_encoder = LabelEncoder()
data['Air Quality'] = label_encoder.fit_transform(data['Air Quality'])

X = data.drop('Air Quality', axis=1)
y = data['Air Quality']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

num_classes = len(label_encoder.classes_)

model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

accuracy = model.evaluate(X_test, y_test)
print(f'Acurácia do modelo: {accuracy[1] * 100:.2f}%')

explainer = LimeTabularExplainer(
    training_data=X_train,                 
    feature_names=X.columns.tolist(),    
    class_names=label_encoder.classes_,   
    mode='classification'                 
)

results = []

for i in range(len(X_test)):
    instance = X_test[i].reshape(1, -1) 

    explanation = explainer.explain_instance(
        data_row=instance.flatten(),  
        predict_fn=model.predict,     
        num_features=5              
    )

    contributions = explanation.as_list() 

    true_label = y_test.iloc[i]  
    true_label_name = label_encoder.inverse_transform([true_label])[0]  

    predicted_class = np.argmax(model.predict(instance))  
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]  

    row = {
        "Instância": i + 1,
        "Dados Iniciais": {col: scaler.inverse_transform([X_test[i]])[0][j] for j, col in enumerate(X.columns)},
        "Diagnóstico Inicial": true_label_name,
        "Diagnóstico do Modelo": predicted_label,
        "Contribuições": {feature: weight for feature, weight in contributions},
    }
    results.append(row)

results_df = pd.DataFrame(results)
results_df.to_csv('lime_results.csv', index=False)
print("Tabela salva como 'lime_results.csv'")
