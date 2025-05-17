from typing import Dict, Any
import torch.nn as nn
from .lstm_model import LSTMPredictor
from .gru_model import GRUPredictor
from .transformer_model import TransformerPredictor

class ModelFactory:
    """모델 생성을 담당하는 팩토리 클래스"""
    
    @staticmethod
    def create_model(model_type: str, model_config: Dict[str, Any]) -> nn.Module:
        """
        모델 타입과 설정에 따라 적절한 모델을 생성합니다.
        
        Args:
            model_type: 모델 타입 ('lstm', 'gru', 'transformer')
            model_config: 모델 설정 딕셔너리
            
        Returns:
            생성된 모델 인스턴스
        """
        if model_type == 'lstm':
            return LSTMPredictor(
                input_size=model_config.get('input_size', 1),
                hidden_size=model_config.get('hidden_size', 64),
                num_layers=model_config.get('num_layers', 2),
                dropout=model_config.get('dropout', 0.2),
                output_size=model_config.get('output_size', 1)
            )
        elif model_type == 'gru':
            return GRUPredictor(
                input_size=model_config.get('input_size', 1),
                hidden_size=model_config.get('hidden_size', 64),
                num_layers=model_config.get('num_layers', 2),
                dropout=model_config.get('dropout', 0.2),
                output_size=model_config.get('output_size', 1)
            )
        elif model_type == 'transformer':
            return TransformerPredictor(
                input_size=model_config.get('input_size', 1),
                d_model=model_config.get('d_model', 64),
                nhead=model_config.get('nhead', 8),
                num_layers=model_config.get('num_layers', 2),
                dropout=model_config.get('dropout', 0.2),
                output_size=model_config.get('output_size', 1)
            )
        else:
            raise ValueError(f"지원하지 않는 모델 타입입니다: {model_type}") 