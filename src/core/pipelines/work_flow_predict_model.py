import os
from core.classes.dermatology_predict import DermatologyPredictor
from core.interfaces.input_interface import InputInterfaceDict
from core.config.config import config
from core.classes.logger import logger

class PredictMedicalModel():

    def __init__(self, path_metadata_model, input):
        self.model_predict = DermatologyPredictor(os.path.join(config.PROJECT_ROOT_PATH, path_metadata_model))
        self.input:InputInterfaceDict  = input

    def run(self):
        logger.info("Iniciando fluxo de predição")
        self.model_predict.process_data(self.input)
        result = self.model_predict.predict_single()
        logger.info("RESULTADO DA PREDIÇÃO")
        logger.info(f'A predição é que seja {result[0]}')
        logger.info(f'Probabilidade da predição {result[1]}')

