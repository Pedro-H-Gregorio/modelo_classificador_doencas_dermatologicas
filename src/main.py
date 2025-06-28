import sys
from core.pipelines.work_flow_predict_model import PredictMedicalModel

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Uso: python main.py <model_path> <age> <sex> <localization> <image_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    age = sys.argv[2]
    sex = sys.argv[3]
    localization = sys.argv[4]
    image_path = sys.argv[5]

    config = {
        "model_path": model_path,
        "age": age,
        "sex": sex,
        "localization": localization,
        "image_path": image_path
    }

    predict_model = PredictMedicalModel(
        model_path, 
        {"age": age,
        "sex": sex,
        "localization": localization,
        "image_path": image_path}
        )
    
    predict_model.run()