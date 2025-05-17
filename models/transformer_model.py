import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Transformer의 위치 인코딩"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(1, max_len, d_model)  # (1, max_len, d_model) 형태로 변경
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 입력 텐서, shape (batch_size, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1), :]

class TransformerPredictor(nn.Module):
    """Transformer 기반 시계열 예측 모델"""
    
    def __init__(self, input_size: int, d_model: int, nhead: int, num_layers: int,
                 dropout: float, output_size: int, target_length: int = 24):
        """
        Args:
            input_size: 입력 특성의 차원
            d_model: 모델의 차원
            nhead: 멀티헤드 어텐션의 헤드 수
            num_layers: 트랜스포머 레이어의 수
            dropout: 드롭아웃 비율
            output_size: 출력 차원
            target_length: 예측할 시간 길이
        """
        super().__init__()
        
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_proj = nn.Linear(d_model, output_size)
        self.d_model = d_model
        self.target_length = target_length
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 입력 텐서, shape (batch_size, sequence_length, input_size)
            
        Returns:
            예측값, shape (batch_size, target_length, output_size)
        """
        # 입력 프로젝션
        x = self.input_proj(x)
        
        # 위치 인코딩 추가
        x = self.pos_encoder(x)
        
        # target_length 만큼의 시퀀스 예측
        predictions = []
        current_input = x
        
        for _ in range(self.target_length):
            # Transformer 인코더 통과
            encoded = self.transformer_encoder(current_input)
            
            # 마지막 시점의 출력으로 다음 값 예측
            current_pred = self.output_proj(encoded[:, -1:, :])
            predictions.append(current_pred)
            
            # 다음 예측을 위해 현재 예측값을 입력에 추가
            pred_proj = self.input_proj(current_pred)
            current_input = torch.cat([current_input[:, 1:, :], pred_proj], dim=1)
        
        # 모든 예측값을 시간 순서대로 결합
        predictions = torch.cat(predictions, dim=1)
        
        return predictions 