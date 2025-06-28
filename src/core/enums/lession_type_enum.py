from enum import Enum

class LesionType(Enum):
    nv = 'Melanocytic nevi'
    mel = 'Melanoma'
    bkl = 'Benign keratosis-like lesions'
    bcc = 'Basal cell carcinoma'
    akiec = 'Actinic keratoses'
    vasc = 'Vascular lesions'
    df = 'Dermatofibroma'

    @classmethod
    def get(cls, code: str) -> str:
        try:
            return cls[code].value
        except KeyError:
            return 'Código inválido'
    
    @classmethod
    def get_all_value(cls) -> list:
        return [i.value for i in cls]