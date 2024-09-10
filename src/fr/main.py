from typing import Union
from fastapi import FastAPI
from fr.manager import get_model_path
import pickle

def run_prediction(length:float):
    model_path = get_model_path()
    with open(model_path, 'rb') as f:
        fish_model = pickle.load(f)
    prediction = fish_model.predict([[length ** 2, length]])
    return float(prediction[0])

@app.get("/")
def read_root():
    return {"Hello": "fish world"}


@app.get("/fish_ml_regression")
def lr_api(length: float):
    """
    물고기의 무게를 예측하는 함수

    Args:
        length(float): 물고기의 길이(cm)

    Returns:
        dict:
            weight(float): 물고기 무게(g)
            length(flozt): 물고기 길이(cm)
    """
 ### 예측해서 결과 return
    prediction = run_prediction(length)
    return {
                "length":length,
                "prediction":prediction
            }
