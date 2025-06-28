import os
from glob import glob
import pandas as pd
import numpy as np
from PIL import Image
from core.enums.lession_type_enum import LesionType
from core.interfaces.input_interface import InputInterfaceDict
from core.config.config import config
from core.classes.logger import logger

def process_image(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).resize((150,150))).flatten()

def preprocess_input_data(input_dict: InputInterfaceDict) -> tuple:
    image_array = process_image(input_dict['image_path'])
    tabular_data = pd.DataFrame({
        'age': [float(input_dict['age'])],
        'sex': [input_dict['sex']],
        'localization': [input_dict['localization']]
    })
    
    return tabular_data, image_array.reshape(1, -1)

def etl_data_training(dir_path: str) -> pd.DataFrame:
    BASE_DIR_ARCHIVE = os.path.join(config.PROJECT_ROOT_PATH, dir_path)

    logger.info(f"Lendo arquivos do diretório: {BASE_DIR_ARCHIVE}")

    # Merging the images into one dictonary
    image_id_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join(BASE_DIR_ARCHIVE, '*', '*.jpg'))}
    
    medical_df = pd.read_csv(os.path.join(BASE_DIR_ARCHIVE, 'HAM10000_metadata.csv'))
    
    medical_df['path'] = medical_df['image_id'].map(image_id_path_dict.get)
    medical_df['cell_type'] = medical_df['dx'].map(LesionType.get) 
    medical_df['cell_type_idx'] = pd.Categorical(medical_df['cell_type']).codes

    medical_df = medical_df.dropna(subset=['path'])

    medical_df['age'] = pd.to_numeric(medical_df['age'], errors='coerce').fillna(medical_df['age'].median())

    return medical_df

def batch_process_images(dataframe: pd.DataFrame) -> np.ndarray:
    logger.info("Iniciando processamento de Imagens")
    image_arrays = []
    for i, path in enumerate(dataframe['path']):
        if i % 1000 == 0:
            logger.info(f"Processadas {i}/{len(dataframe)} imagens")
        image_arrays.append(process_image(path))
    logger.info("Finalizado processamento de Imagens")
    return np.array(image_arrays)

def preprosses_data_training(dir_path:str) -> tuple:
    dataframe = etl_data_training(dir_path)
    X_TABULAR = dataframe[["age", "sex", "localization"]].copy()
    X_IMAGE = batch_process_images(dataframe)
    Y = dataframe["cell_type_idx"].values

    return X_TABULAR, X_IMAGE, Y