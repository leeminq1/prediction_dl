import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from datetime import datetime

class MetricsLogger:
    """학습 메트릭을 로깅하고 저장하는 클래스"""
    
    def __init__(self, log_dir: str):
        """
        Args:
            log_dir: 로그를 저장할 디렉토리 경로
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_times': []
        }
        self.best_metrics = {
            'val_loss': float('inf'),
            'epoch': 0
        }
        self.config = None  # 설정 저장을 위한 변수 추가
        
    def set_config(self, args):
        """설정 정보를 저장합니다."""
        self.config = {
            # 데이터 관련 설정
            'data_config': {
                'data_dir': str(args.data_dir),
                'processed_dir': str(args.processed_dir),
                'train_years': args.train_years,
                'test_years': args.test_years,
                'sequence_length': args.sequence_length,
                'target_length': args.target_length,
                'train_stride': args.train_stride,
                'test_stride': args.test_stride
            },
            # 모델 관련 설정
            'model_config': {
                'model_type': args.model_type,
                'hidden_size': args.hidden_size,
                'num_layers': args.num_layers,
                'dropout': args.dropout
            },
            # 학습 관련 설정
            'training_config': {
                'batch_size': args.batch_size,
                'epochs': args.epochs,
                'learning_rate': args.learning_rate,
                'early_stopping': args.early_stopping,
                'val_ratio': args.val_ratio
            },
            # 저장 관련 설정
            'save_config': {
                'weight_dir': str(args.weight_dir),
                'experiment_name': args.experiment_name,
                'save_best_only': args.save_best_only
            },
            # 기타 설정
            'other_config': {
                'device': str(args.device),
                'seed': args.seed
            }
        }
        
    def log_metrics(self, metrics: Dict[str, float], epoch: int):
        """한 에포크의 메트릭을 기록합니다."""
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
                
        # 최고 성능 갱신 확인
        if metrics.get('val_loss', float('inf')) < self.best_metrics['val_loss']:
            self.best_metrics['val_loss'] = metrics['val_loss']
            self.best_metrics['epoch'] = epoch
            return True
        return False
    
    def save_metrics(self):
        """현재까지의 메트릭을 JSON 파일로 저장합니다."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        metrics_file = self.log_dir / f'metrics_{timestamp}.json'
        
        metrics_data = {
            'metrics_history': {
                k: [float(v) for v in vals] for k, vals in self.metrics_history.items()
            },
            'best_metrics': self.best_metrics,
            'final_metrics': {
                'train_loss': float(np.mean(self.metrics_history['train_loss'][-5:])),
                'val_loss': float(np.mean(self.metrics_history['val_loss'][-5:]))
            }
        }
        
        # 설정 정보 추가
        if self.config is not None:
            metrics_data['config'] = self.config
            
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=4)
            
    def get_summary(self) -> Dict[str, Any]:
        """현재까지의 학습 요약을 반환합니다."""
        return {
            'best_val_loss': self.best_metrics['val_loss'],
            'best_epoch': self.best_metrics['epoch'],
            'latest_train_loss': self.metrics_history['train_loss'][-1],
            'latest_val_loss': self.metrics_history['val_loss'][-1],
            'total_epochs': len(self.metrics_history['train_loss'])
        } 