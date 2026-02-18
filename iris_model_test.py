import pickle
import numpy as np
from sklearn.datasets import load_iris

with open("model_and_scalar.pkl", "rb") as f:
    saved_data = pickle.load(f)
    loaded_model = saved_data["model"]  
    loaded_scalar = saved_data["scalar"]


iris = load_iris()

def predict_iris_flower():
    try:
        sepal_length = float(input("Enter sepal length (cm): "))
        sepal_width = float(input("Enter sepal width (cm): "))
        petal_length = float(input("Enter petal length (cm): "))
        petal_width = float(input("Enter petal width (cm): "))

        new_sample=np.array([sepal_length,sepal_width,petal_length,petal_width])
        new_sample = new_sample.reshape(1,-1)
        new_sample_scaled = loaded_scalar.transform(new_sample)

        predicted_class = loaded_model.predict(new_sample_scaled)
        predicted_proba = loaded_model.predict_proba(new_sample_scaled)


        print(f"Predicted class: {predicted_class[0]}")
        print(f"Predicted Probability: {predicted_proba[0]}")

        print(f"Predicted class name: {iris.target_names[predicted_class[0]]}")
    
    except ValueError:
        print("Invalid input, please enter numeric values")



while True:
    predict_iris_flower()

    continue_input = input("Do you want to enter another sample? (y/n): ").strip().lower()

    if continue_input != 'y':
        print("Exiting App")
        break