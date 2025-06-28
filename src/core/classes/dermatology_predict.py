import joblib
import os
import json
import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.svm import SVC, OneClassSVM
from core.classes.logger import logger
from core.pipelines.pipelines import DataPipeline
from core.utils.load_data_utils import preprocess_input_data
from core.interfaces.input_interface import InputInterfaceDict
from core.interfaces.metadate_interface import ModelMetadataInterface
from core.enums.lession_type_enum import LesionType
from core.enums.type_model import TypeModel
from core.config.config import config
from typing import Union, Optional
import warnings
warnings.filterwarnings('ignore')

class DermatologyPredictor:
    
    def __init__(self, model_metadata_path: str):
        self.metadata_model: Optional[ModelMetadataInterface] = None
        self.model: Union[SVC, OneClassSVM]  = None
        self.model_type: Optional[str] = None
        self.input_combined: np.ndarray = None
        self.pipeline: DataPipeline = None
        self.lesion_type = LesionType
        logger.info("Carregando modelos...")
        try:
            self.metadata_model = self._from_json_file(model_metadata_path)
            self.model_type = self.metadata_model['model_type']
            self.model = joblib.load(self.metadata_model["model_path"])
            logger.info(f"Modelo carregado com sucesso! Tipo: {self.model_type}")
            if self.metadata_model["pipeline_path"]:
                self.pipeline = joblib.load(self.metadata_model["pipeline_path"])
                logger.info(f"Pipeline carregada com sucesso! Tipo: {self.model_type}")

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Arquivo de modelo não encontrado: {e}")
    
    def _from_json_file(self, json_path: str) -> ModelMetadataInterface:                 
        with open(json_path, 'r', encoding='utf-8') as f:             
            data = json.load(f)         
        
            return data

    def to_percentage(self,value: float):
        return f"{value * 100:.2f}%"

    def _predict_svc(self) -> tuple:
        logger.info("Executando predição SVC")
        prediction = self.model.predict_proba(self.input_combined)
        predicted_class = self.model.predict(self.input_combined)
    
        return (self.lesion_type.get_all_value()[predicted_class[0]], self.to_percentage(prediction[0][predicted_class[0]]))
    
    def _predict_oneclass(self) -> tuple:
        logger.info("Executando predição OneClassSVM")
        
        prediction = self.model.predict(self.input_combined)[0]
        
        is_normal = prediction == 1
        
        try:
            decision_score = self.model.decision_function(self.input_combined)[0]
            
            probability_normal = 1 / (1 + np.exp(-decision_score))
            probability_anomaly = 1 - probability_normal
            
            if is_normal:
                return ("normal", self.to_percentage(probability_normal))
            else:
                return ("anormal", self.to_percentage(probability_anomaly))

        except Exception as e:
            logger.warning(f"Não foi possível calcular probabilidades: {e}")

    def process_data(self, input: InputInterfaceDict):
        logger.info("Iniciando processamento de input")
        tabular_data, image_data = preprocess_input_data(input)
        inter_tabular_processed = self.pipeline.pipeline_transform_tabular(tabular_data)
        inter_image_processed = self.pipeline.pipeline_transform_image(image_data)
        self.input_combined = np.hstack([inter_tabular_processed, inter_image_processed])
        logger.info(f"Dados combinados para input: {self.input_combined.shape}")
        logger.info("Finalizada normalização e transformação de dados")
         
    
    def predict_single(self):
        if self.input_combined is None:
            raise ValueError("Dados não foram processados. Execute process_data() primeiro.")

        try:
            if self.model_type == TypeModel.one_class_svm.value:
                return self._predict_oneclass()
            elif self.model_type == TypeModel.svc.value:
                return self._predict_svc()
                
        except NotFittedError as e:
            raise NotFittedError(f"Erro na predição: {e}")
        