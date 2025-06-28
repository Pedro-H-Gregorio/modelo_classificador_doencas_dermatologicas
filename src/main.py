import sys
from core.pipelines.work_flow_predict_model import PredictMedicalModel
from core.classes.logger import logger

if __name__ == "__main__":
    if len(sys.argv) != 6:
        logger.error("Uso: python main.py <metadate_model_path> <age> <sex> <localization> <image_path>")
        sys.exit(1)

    metadata_model_path = sys.argv[1]
    age = sys.argv[2]
    sex = sys.argv[3]
    localization = sys.argv[4]
    image_path = sys.argv[5]

    config = {
        "model_path": metadata_model_path,
        "age": age,
        "sex": sex,
        "localization": localization,
        "image_path": image_path
    }

    predict_model = PredictMedicalModel(
        metadata_model_path, 
        {"age": age,
        "sex": sex,
        "localization": localization,
        "image_path": image_path}
        )
    
    predict_model.run()