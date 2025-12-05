from fastapi import FastAPI
from pydantic import BaseModel, conlist
from typing import List, Optional
import pandas as pd

from model import recommend, output_recommended_recipes


# Load dataset once at startup (optimized)
dataset = pd.read_csv("../Data/dataset.csv", compression="gzip")

app = FastAPI()


class Params(BaseModel):
    n_neighbors: int = 5
    return_distance: bool = False


class PredictionIn(BaseModel):
    nutrition_input: conlist(float, min_items=9, max_items=9) # type: ignore
    ingredients: List[str] = []
    params: Optional[Params] = Params()


class Recipe(BaseModel):
    Name: str
    CookTime: str
    PrepTime: str
    TotalTime: str
    RecipeIngredientParts: List[str]
    Calories: float
    FatContent: float
    SaturatedFatContent: float
    CholesterolContent: float
    SodiumContent: float
    CarbohydrateContent: float
    FiberContent: float
    SugarContent: float
    ProteinContent: float
    RecipeInstructions: List[str]


class PredictionOut(BaseModel):
    output: Optional[List[Recipe]] = None


@app.get("/")
def home():
    return {"health_check": "OK"}


@app.post("/predict/", response_model=PredictionOut)
def predict(prediction_input: PredictionIn):
    try:
        params_dict = prediction_input.params.dict() if prediction_input.params else {}

        recommendation_df = recommend(
            dataset,
            prediction_input.nutrition_input,
            prediction_input.ingredients,
            params_dict
        )

        output = output_recommended_recipes(recommendation_df)
        return {"output": output or None}

    except Exception:
        # In production, replace with proper logging
        return {"output": None}
