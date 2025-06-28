import joblib
import os
import numpy as np
from sklearn.exceptions import NotFittedError
from core.enums.type_model import TypeModel
from sklearn.svm import SVC, OneClassSVM
from core.classes.logger import logger
from core.pipelines.pipelines import DataPipeline
from core.utils.load_data_utils import preprocess_input_data
from core.interfaces.input_interface import InputInterfaceDict
from core.enums.lession_type_enum import LesionType
from core.config.config import config
from typing import Union, Optional
import warnings
warnings.filterwarnings('ignore')

class DermatologyPredictor:
    
    def __init__(self, model_path):
        self.model: Union[SVC, OneClassSVM]  = None
        self.model_type: Optional[str] = None
        self.input_combined: np.ndarray = None
        self.pipeline: DataPipeline = None
        self.lesion_type = LesionType
        logger.info("Carregando modelos...")
        try:
            self.model = joblib.load(model_path)
            self._detect_model_type()
            logger.info(f"Modelo carregado com sucesso! Tipo: {self.model_type}")

            pipeline_path = self._find_corresponding_pipeline()
            if pipeline_path:
                self.pipeline = joblib.load(pipeline_path)
                logger.info(f"Pipeline carregada com sucesso! Tipo: {self.model_type}")

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Arquivo de modelo não encontrado: {e}")
    
    def _find_corresponding_pipeline(self):
        
        try:
            pipelines_dir = os.path.join(config.PROJECT_ROOT_PATH, "pipelines")
            pipeline_filename = f"pipeline_{self.model_type}.pkl"
            pipeline_path = os.path.join(pipelines_dir, pipeline_filename)
            
            if os.path.exists(pipeline_path):
                logger.info(f"Pipeline encontrada: {pipeline_path}")
                return pipeline_path
            else:
                logger.warning(f"Pipeline não encontrada em: {pipeline_path}")
                return None
                
        except Exception as e:
            logger.error(f"Erro ao buscar pipeline correspondente: {e}")
            return None

    def _detect_model_type(self):
        
        if isinstance(self.model, OneClassSVM):
            self.model_type = TypeModel.one_class_svm.value
            logger.info("Tipo detectado: OneClassSVM (Detecção de anomalias)")
        elif isinstance(self.model, SVC):
            self.model_type = TypeModel.svc.value
            logger.info("Tipo detectado: SVC")

    def _predict_svc(self) -> tuple:
        logger.info("Executando predição SVC")
        prediction = self.model.predict_proba(self.input_combined)
        predicted_class = self.model.predict(self.input_combined)
    
        return (prediction[0], self.lesion_type.get_all_value()[predicted_class])
    
    def _predict_oneclass(self) -> tuple:
        logger.info("Executando predição OneClassSVM")
        
        prediction = self.model.predict(self.input_combined)[0]
        
        is_normal = prediction == 1
        
        try:
            decision_score = self.model.decision_function(self.input_combined)[0]
            
            # Converte para "probabilidade" usando sigmoid
            # Valores positivos = normal, negativos = anomalia
            probability_normal = 1 / (1 + np.exp(-decision_score))
            probability_anomaly = 1 - probability_normal
            
            if is_normal:
                return (probability_normal, "normal")
            else:
                return (probability_anomaly, "anormal")

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
            if self.model_type == "oneclass":
                return self._predict_oneclass()
            else:
                return self._predict_svc()
                
        except NotFittedError as e:
            raise NotFittedError(f"Erro na predição: {e}")
        