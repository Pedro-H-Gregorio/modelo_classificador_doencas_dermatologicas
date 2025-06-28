from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

class DataPipeline():
    
    def __init__(self):
        self.tabular_pipe = Pipeline([
            ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore')),
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=10)),
        ], memory='./pipeline_cache')
        self.image_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=1000)),
        ], memory='./pipeline_cache')
    
    def pipeline_fit_transform_tabular(self, dataframe: pd.DataFrame) -> np.ndarray:
        return self.tabular_pipe.fit_transform(dataframe)

    def pipeline_fit_transform_image(self, array: np.asarray)-> np.ndarray:
        return self.image_pipe.fit_transform(array)
    
    def pipeline_transform_tabular(self, dataframe: pd.DataFrame) -> np.ndarray:
        return self.tabular_pipe.transform(dataframe)

    def pipeline_transform_image(self, array: np.asarray)-> np.ndarray:
        return self.image_pipe.transform(array)