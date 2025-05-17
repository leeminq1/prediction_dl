import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict

class LSTMPredictor(nn.Module):
    """LSTM 기반 시계열 예측 모델"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 dropout: float, output_size: int, target_length: int = 24):
        """
        Args:
            input_size: 입력 특성의 차원
            hidden_size: LSTM 히든 레이어의 차원
            num_layers: LSTM 레이어의 수
            dropout: 드롭아웃 비율
            output_size: 출력 차원
            target_length: 예측할 시간 길이
        """
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
        self.target_length = target_length
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 입력 텐서, shape (batch_size, sequence_length, input_size)
            
        Returns:
            예측값, shape (batch_size, target_length, output_size)
        """
        batch_size = x.size(0)
        
        # 초기 은닉 상태
        h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)
        
        # LSTM 레이어 통과
        _, (hidden, cell) = self.lstm(x, (h0, c0))
        
        # target_length 만큼의 시퀀스 예측
        predictions = []
        current_input = x[:, -1:, :]  # 마지막 입력값, shape: (batch_size, 1, input_size)
        
        for _ in range(self.target_length):
            # 현재 입력으로 다음 값 예측
            lstm_out, (hidden, cell) = self.lstm(current_input, (hidden, cell))
            current_pred = self.fc(lstm_out.squeeze(1))  # shape: (batch_size, output_size)
            predictions.append(current_pred.unsqueeze(1))  # shape: (batch_size, 1, output_size)
            
            # 다음 예측을 위해 현재 예측값을 입력으로 사용
            current_input = current_pred.unsqueeze(1)  # shape: (batch_size, 1, input_size)
        
        # 모든 예측값을 시간 순서대로 결합
        predictions = torch.cat(predictions, dim=1)  # shape: (batch_size, target_length, output_size)
        
        return predictions
    
    def predict(self, x: torch.Tensor) -> np.ndarray:
        """예측을 수행하는 메서드"""
        self.eval()
        with torch.no_grad():
            predictions = self(x)
        return predictions.cpu().numpy()

class LSTMTrainer:
    def __init__(self,
                 model: LSTMPredictor,
                 learning_rate: float = 0.001,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """한 배치에 대한 학습을 수행합니다."""
        self.model.train()
        x, y = x.to(self.device), y.to(self.device)
        
        # Forward pass
        outputs = self.model(x)
        loss = self.criterion(outputs, y)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {'loss': loss.item()}
    
    def validate_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """한 배치에 대한 검증을 수행합니다."""
        self.model.eval()
        with torch.no_grad():
            x, y = x.to(self.device), y.to(self.device)
            outputs = self.model(x)
            loss = self.criterion(outputs, y)
            
        return {'val_loss': loss.item()} 