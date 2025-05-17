import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from typing import Tuple, Dict, Any

class TimeSeriesDataLoader:
    """시계열 데이터 로딩을 담당하는 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 데이터 로더 설정
        """
        self.sequence_length = config['sequence_length']
        self.target_length = config['target_length']
        self.batch_size = config['batch_size']
        self.val_ratio = config['val_ratio']
        self.train_stride = config.get('train_stride', 1)
        self.test_stride = config.get('test_stride', 24)
        
    def create_sequences(self, data: np.ndarray, stride: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """시계열 데이터를 입력 시퀀스와 타겟으로 변환합니다.
        
        Args:
            data: 원본 시계열 데이터
            stride: 시퀀스 생성 시 사용할 stride (기본값: None, None이면 train_stride 사용)
            
        Returns:
            sequences: 입력 시퀀스 배열
            targets: 타겟 배열
        """
        if stride is None:
            stride = self.train_stride
            
        sequences = []
        targets = []
        
        # 입력 시퀀스와 타겟 시퀀스의 총 길이
        total_length = self.sequence_length + self.target_length
        
        # 가능한 시작점들을 stride를 고려하여 생성
        for i in range(0, len(data) - total_length + 1, stride):
            # 입력 시퀀스
            seq = data[i:i + self.sequence_length]
            # 타겟 시퀀스 (다음 target_length 시간의 값들)
            target = data[i + self.sequence_length:i + total_length]
            
            sequences.append(seq)
            targets.append(target)
            
        return np.array(sequences), np.array(targets)
    
    def prepare_data(self, data: np.ndarray) -> Tuple[DataLoader, DataLoader]:
        """데이터를 학습과 검증 데이터로더로 변환합니다."""
        # 시퀀스 생성 (학습용 - train_stride 사용)
        X, y = self.create_sequences(data, self.train_stride)
        
        # 텐서 변환
        X_tensor = torch.FloatTensor(X).unsqueeze(-1)  # shape: [N, seq_len, 1]
        y_tensor = torch.FloatTensor(y).unsqueeze(-1)  # shape: [N, target_len, 1]
        
        # 데이터셋 생성
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # 학습/검증 분할
        train_size = int((1 - self.val_ratio) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # 데이터로더 생성
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        return train_loader, val_loader
    
    def prepare_test_data(self, data: np.ndarray) -> DataLoader:
        """테스트 데이터를 데이터로더로 변환합니다."""
        # 시퀀스 생성 (테스트용 - test_stride 사용)
        X, y = self.create_sequences(data, self.test_stride)
        
        X_tensor = torch.FloatTensor(X).unsqueeze(-1)
        y_tensor = torch.FloatTensor(y).unsqueeze(-1)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.batch_size) 