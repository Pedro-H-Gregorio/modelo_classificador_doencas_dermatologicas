import os
import datetime
from core.interfaces.config_interface import ConfigInterface
from core.enums.type_model import TypeModel
from core.utils.load_data_utils import preprosses_data_training
from core.classes.logger import logger
from core.config.config import config
import numpy as np
import pandas as pd
import joblib
from sklearn.svm import SVC, OneClassSVM
from sklearn.model_selection import GridSearchCV
from core.pipelines.pipelines import DataPipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class GenerateMedicalModel:
    def __init__(self, config: ConfigInterface):
        self.config: ConfigInterface = config
        self.pipeline: DataPipeline = DataPipeline()
        self.x_tabular_train: 'pd.DataFrame | None' = None
        self.x_tabular_test: 'pd.DataFrame | None' = None
        self.x_image_train: 'np.ndarray | None' = None
        self.x_image_test: 'np.ndarray | None' = None
        self.x_train_combined: 'np.ndarray | None' = None
        self.x_test_combined: 'np.ndarray | None' = None
        self.y_train: 'np.ndarray | None' = None
        self.y_test: 'np.ndarray | None' = None
        self.model: 'GridSearchCV | None' = None

    def generate_model(self):
        logger.info(f"Gerando model do tipo: {self.config['model']}")
        param_grid = None
        svm = None
        
        if self.config['model'] == TypeModel.one_class_svm.value:
            param_grid = {
                'nu': [0.01, 0.05, 0.1],
                'gamma': [1e-4, 1e-3, 1e-2],
                'kernel': ['rbf']
            }
            svm = OneClassSVM()

        else:
            param_grid = {
                'C': [0.1, 1, 10],
                'gamma': [1e-4, 1e-3, 1e-2],
                'kernel': ['rbf']
            }
            svm = SVC(probability=True)
        
        self.model = GridSearchCV(
                svm, 
                param_grid, 
                cv=4, 
                verbose=2, 
                n_jobs=-1,
                scoring="f1_weighted"
            )
            

    def process(self):
        logger.info("Iniciando o processamento de dados")
        X_TABULAR, X_IMAGE, Y = preprosses_data_training(self.config['dir_path'])
        logger.info("finalizando o processamento de dados")
        logger.info("Dividindo dados de treino e teste")
        indices = np.arange(len(X_TABULAR))
        train_idx, test_idx = train_test_split(
            indices, test_size=0.1, stratify=Y, random_state=42
        )

        self.x_tabular_train = X_TABULAR.iloc[train_idx]
        self.x_tabular_test = X_TABULAR.iloc[test_idx]
        self.x_image_train = X_IMAGE[train_idx]
        self.x_image_test = X_IMAGE[test_idx]
        self.y_train = Y[train_idx]
        self.y_test = Y[test_idx]

        logger.info(f"Dados de treino - Tabular: {self.x_tabular_train.shape}, Imagem: {self.x_image_train.shape}")
        logger.info(f"Dados de teste - Tabular: {self.x_tabular_test.shape}, Imagem: {self.x_image_test.shape}")

    def transformation_data(self):
        logger.info("Iniciando normalização e transformação de dados")

        logger.info("Ajustando pipeline tabular...")
        x_tabular_processed = self.pipeline.pipeline_fit_transform_tabular(self.x_tabular_train)
        logger.info(f"Dados tabulares processados: {x_tabular_processed.shape}")

        logger.info("Ajustando pipeline de imagens...")
        x_image_processed = self.pipeline.pipeline_fit_transform_image(self.x_image_train)
        logger.info(f"Dados de imagem processados: {x_image_processed.shape}")

        self.x_train_combined = np.hstack([x_tabular_processed, x_image_processed])
        logger.info(f"Dados combinados para treino: {self.x_train_combined.shape}")

        x_tabular_test_processed = self.pipeline.pipeline_transform_tabular(self.x_tabular_test)
        x_image_test_processed = self.pipeline.pipeline_transform_image(self.x_image_test)
        self.x_test_combined = np.hstack([x_tabular_test_processed, x_image_test_processed])
        logger.info(f"Dados combinados para teste: {self.x_test_combined.shape}")
        logger.info("Finalizada normalização e transformação de dados")

    def training_model(self):
        logger.info("Iniciando treinamento do modelo")
        if self.config['model'] == TypeModel.one_class_svm.value:
            self.model.fit(self.x_train_combined)
        else: 
            self.model.fit(self.x_train_combined, self.y_train)
        logger.info("Treinamento Finalizado")

    def evaluating_parameters(self):
        logger.info("Avaliando parametros")
        logger.info(f'Melhores parâmetros: {self.model.best_params_}')
        logger.info(f'Melhor validação de pontuação: {self.model.best_score_}')

        logger.info("Avaliando no conjunto de teste")
        y_pred = self.model.predict(self.x_test_combined)
        logger.info(f"Teste de acuracia: {accuracy_score(self.y_test, y_pred)}")
        logger.info("relatório de classificação")
        logger.info(classification_report(self.y_test, y_pred, zero_division=0))
    
    def saving_model(self):
        logger.info("Salvando modelo e pipeline...")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"dermatology_model_{self.config['model']}"
        pipeline_name = f"pipeline_{self.config['model']}.pkl"
        
        base_path = config.PROJECT_ROOT_PATH
        models_dir = os.path.join(base_path, "models")
        pipelines_dir = os.path.join(base_path, "pipelines")
        
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(pipelines_dir, exist_ok=True)
        
        self._remove_existing_model_files(models_dir, pipelines_dir, self.config['model'])
        
        model_path = os.path.join(models_dir, f"{model_name}.pkl")
        pipeline_path = os.path.join(pipelines_dir, pipeline_name)
        
        try:
            logger.info(f"Salvando modelo em: {model_path}")
            joblib.dump(self.model, model_path)
            
            logger.info(f"Salvando pipeline em: {pipeline_path}")
            joblib.dump(self.pipeline, pipeline_path)
            
            metadata = {
                'model_type': self.config['model'],
                'timestamp': timestamp,
                'model_path': model_path,
                'pipeline_path': pipeline_path,
                'model_filename': f"{model_name}.pkl",
                'pipeline_filename': pipeline_name,
                'best_params': getattr(self.model, 'best_params_', None),
                'best_score': getattr(self.model, 'best_score_', None),
                'train_data_shape': self.x_train_combined.shape if hasattr(self, 'x_train_combined') else None,
                'test_data_shape': self.x_test_combined.shape if hasattr(self, 'x_test_combined') else None
            }
            
            metadata_path = os.path.join(models_dir, f"metadata_{model_name}.json")
            
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info("Modelo, pipeline e metadados salvos com sucesso!")
            logger.info("Estrutura de arquivos criada:")
            logger.info(f"  {models_dir}/")
            logger.info(f"    {model_name}.pkl")
            logger.info(f"    metadata_{model_name}.json")
            logger.info(f"  {pipelines_dir}/")
            logger.info(f"    {pipeline_name}")
            
        except Exception as e:
            logger.error(f"Erro ao salvar modelo/pipeline: {e}")
            raise

    def _remove_existing_model_files(self, models_dir, pipelines_dir, model_type):
        try:
            model_pattern = f"dermatology_model_{model_type}.pkl"
            pipeline_pattern = f"pipeline_{model_type}.pkl"
            metadata_pattern = f"metadata_dermatology_model_{model_type}.json"
            
            model_file = os.path.join(models_dir, model_pattern)
            if os.path.exists(model_file):
                os.remove(model_file)
                logger.info(f"Arquivo de modelo existente removido: {model_file}")
            
            metadata_file = os.path.join(models_dir, metadata_pattern)
            if os.path.exists(metadata_file):
                os.remove(metadata_file)
                logger.info(f"Arquivo de metadados existente removido: {metadata_file}")
            
            pipeline_file = os.path.join(pipelines_dir, pipeline_pattern)
            if os.path.exists(pipeline_file):
                os.remove(pipeline_file)
                logger.info(f"Arquivo de pipeline existente removido: {pipeline_file}")
                
        except Exception as e:
            logger.warning(f"Erro ao remover arquivos existentes: {e}")

    def run(self):
        self.process()
        self.transformation_data()
        self.generate_model()
        self.training_model()
        self.evaluating_parameters()
        self.saving_model()