!pip install lime

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
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

model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

accuracy = model.evaluate(X_test, y_test)
print(f'Acurácia do modelo: {accuracy[1]*100:.2f}%')

explainer = LimeTabularExplainer(
    X_train,  
    training_labels=y_train,  
    feature_names=X.columns.tolist(),  
    class_names=label_encoder.classes_.tolist(), 
    mode="classification"
)

instance = X_test[0].reshape(1, -1)  

explanation = explainer.explain_instance(
    data_row=instance.flatten(),  
    predict_fn=model.predict,     
    num_features=9              
)

fig = explanation.as_pyplot_figure()  
fig.savefig('lime_explanation1.png', bbox_inches='tight') 
print("Gráfico LIME salvo como 'lime_explanation.png'")
