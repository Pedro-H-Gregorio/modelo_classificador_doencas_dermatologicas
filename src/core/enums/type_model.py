from enum import Enum

class TypeModel(Enum):
    svc = "svc"
    one_class_svm = "one_class_svm"

    @classmethod
    def get(cls, code: str) -> str:
        try:
            return cls[code].value
        except KeyError:
            return 'Código inválido'
    
    @classmethod
    def get_all_values(cls) -> list:
        return [i.value for i in cls]