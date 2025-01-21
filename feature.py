for i in range(len(X_test)):
    instance = X_test[i].reshape(1, -1)  

    explanation = explainer.explain_instance(
        data_row=instance.flatten(),  
        predict_fn=model.predict,     
        num_features=5               
    )

    contributions = explanation.as_list() 

    predicted_class = np.argmax(model.predict(instance)) 
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]  

    print(f"Instância {i + 1}: Classe prevista -> {predicted_label}")
    print("Explicação:")
    for feature, weight in contributions:
        influence = "positiva" if weight > 0 else "negativa"
        print(f"  - {feature}: influência {influence} de {weight:.2f}")

    print("\n" + "-" * 50 + "\n")
