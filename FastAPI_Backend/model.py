import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


def scaling(dataframe):
    """
    Scales the nutrition columns (6:15) using StandardScaler.
    Returns the scaled data and the scaler object.
    """
    scaler = StandardScaler()
    prep_data = scaler.fit_transform(dataframe.iloc[:, 6:15].to_numpy())
    return prep_data, scaler


def nn_predictor(prep_data):
    """
    Fits a NearestNeighbors model with cosine metric and brute algorithm.
    """
    neigh = NearestNeighbors(metric='cosine', algorithm='brute')
    neigh.fit(prep_data)
    return neigh


def build_pipeline(neigh, scaler, params):
    """
    Builds a pipeline with the scaler and a transformer for kneighbors.
    """
    transformer = FunctionTransformer(neigh.kneighbors, kw_args=params)
    pipeline = Pipeline([('std_scaler', scaler), ('NN', transformer)])
    return pipeline


def extract_data(dataframe, ingredients):
    """
    Extracts and filters data based on ingredients.
    """
    extracted_data = dataframe.copy()
    extracted_data = extract_ingredient_filtered_data(extracted_data, ingredients)
    return extracted_data


def extract_ingredient_filtered_data(dataframe, ingredients):
    """
    Filters the dataframe to include only recipes that contain all specified ingredients.
    Uses regex with positive lookaheads for efficient AND matching.
    """
    if not ingredients:
        return dataframe  # No filtering if no ingredients provided
    extracted_data = dataframe.copy()
    regex_string = ''.join(f'(?=.*{re.escape(ing)})' for ing in ingredients)
    extracted_data = extracted_data[
        extracted_data['RecipeIngredientParts'].str.contains(
            regex_string, regex=True, flags=re.IGNORECASE
        )
    ]
    return extracted_data


def apply_pipeline(pipeline, _input, extracted_data):
    """
    Applies the pipeline to the input and returns the recommended recipes.
    """
    _input = np.array(_input).reshape(1, -1)
    indices = pipeline.transform(_input)[0]
    return extracted_data.iloc[indices]


def recommend(dataframe, _input, ingredients=[], params={'n_neighbors': 5, 'return_distance': False}):
    """
    Main recommendation function.
    Filters data by ingredients, scales, fits NN, and predicts recommendations.
    Returns None if insufficient data.
    """
    extracted_data = extract_data(dataframe, ingredients)
    if extracted_data.shape[0] >= params['n_neighbors']:
        prep_data, scaler = scaling(extracted_data)
        neigh = nn_predictor(prep_data)
        pipeline = build_pipeline(neigh, scaler, params)
        return apply_pipeline(pipeline, _input, extracted_data)
    else:
        return None


def extract_quoted_strings(s):
    """
    Extracts strings enclosed in double quotes from the input string.
    Assumes the string is a list-like representation with quoted items.
    """
    # Find all strings inside double quotes
    strings = re.findall(r'"([^"]*)"', s)
    return strings


def output_recommended_recipes(dataframe):
    """
    Formats the recommended recipes dataframe into a list of dictionaries.
    Parses RecipeIngredientParts and RecipeInstructions from quoted strings.
    Returns None if dataframe is None.
    """
    if dataframe is not None:
        output = dataframe.copy()
        output = output.to_dict("records")
        for recipe in output:
            recipe['RecipeIngredientParts'] = extract_quoted_strings(recipe['RecipeIngredientParts'])
            recipe['RecipeInstructions'] = extract_quoted_strings(recipe['RecipeInstructions'])
        return output
    else:
        return None
