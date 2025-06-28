from core.pipelines.work_flow_generate_model import GenerateMedicalModel
from core.enums.type_model import TypeModel
from core.config.config import config

def type_model_is_valid(model: str) -> str:
    if model in TypeModel.get_all_values():
        return TypeModel.get(model)
    else: raise ValueError("Invalid model type provided.")


def generate_dermatology_model(dir_path: str, model: str):
    model_type = type_model_is_valid(model)
    config = {"dir_path": dir_path, "model" : model_type}
    generate_model = GenerateMedicalModel(config)
    generate_model.run()

generate_dermatology_model(config.DATA_PATH, config.MODEL)