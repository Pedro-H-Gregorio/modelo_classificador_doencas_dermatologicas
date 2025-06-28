from typing import Dict, Optional, Any, Tuple, TypedDict

class ModelMetadataInterface(TypedDict):
    
    model_type: str
    timestamp: str
    model_path: str
    pipeline_path: str
    model_filename: str
    pipeline_filename: str
    best_params: Optional[Dict[str, Any]]
    best_score: Optional[float]
    train_data_shape: Optional[Tuple[int, int]]
    test_data_shape: Optional[Tuple[int, int]]