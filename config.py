import argparse
from pathlib import Path
import datetime
import torch

def get_args():
    parser = argparse.ArgumentParser(description='SMP 예측 모델 학습을 위한 설정')
    
    # 데이터 관련 설정
    parser.add_argument('--data_dir', type=str, default='data',
                        help='원본 데이터가 있는 디렉토리')
    parser.add_argument('--processed_dir', type=str, default='data_processed',
                        help='전처리된 데이터를 저장할 디렉토리')
    parser.add_argument('--train_years', type=int, nargs='+', default=[2023, 2024],
                        help='학습에 사용할 연도들')
    parser.add_argument('--test_years', type=int, nargs='+', default=[2025],
                        help='테스트에 사용할 연도들')
    parser.add_argument('--sequence_length', type=int, default=24*5,
                        help='입력 시퀀스 길이 (시간)')
    parser.add_argument('--target_length', type=int, default=24,
                        help='예측할 시간 길이')
    parser.add_argument('--train_stride', type=int, default=1,
                        help='학습 데이터 생성시 사용할 stride (시간 단위)')
    parser.add_argument('--test_stride', type=int, default=24,
                        help='테스트 데이터 생성시 사용할 stride (시간 단위)')
    
    # 모델 관련 설정
    parser.add_argument('--model_type', type=str, default='gru',
                        choices=['lstm', 'gru', 'transformer'],
                        help='사용할 모델 타입')
    parser.add_argument('--hidden_size', type=int, default=32,
                        help='히든 레이어 크기')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='레이어 수')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='드롭아웃 비율')
    # Transformer 모델 전용 설정 추가
    parser.add_argument('--d_model', type=int, default=32,
                        help='Transformer 모델 차원')
    parser.add_argument('--nhead', type=int, default=4,
                        help='Transformer 멀티헤드 어텐션의 헤드 수')
    
    # 학습 관련 설정
    parser.add_argument('--batch_size', type=int, default=64,
                        help='배치 크기')
    parser.add_argument('--epochs', type=int, default=5,
                        help='전체 에포크 수')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='학습률')
    parser.add_argument('--early_stopping', type=int, default=10,
                        help='조기 종료를 위한 에포크 수')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='검증 데이터 비율')
    
    # 저장 관련 설정
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    parser.add_argument('--weight_dir', type=str, default='weights',
                        help='가중치를 저장할 디렉토리')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='실험 이름')
    parser.add_argument('--save_best_only', type=bool, default=True,
                        help='최고 성능 모델만 저장')
    
    # 기타 설정
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='학습에 사용할 디바이스')
    parser.add_argument('--seed', type=int, default=42,
                        help='랜덤 시드')
    
    args = parser.parse_args()
    
    # experiment_name이 지정되지 않은 경우 model_type을 사용하여 설정
    if args.experiment_name is None:
        args.experiment_name = f'{current_time}_{args.model_type}'
    
    # 실험 디렉토리 생성
    args.experiment_dir = Path(args.weight_dir) / args.experiment_name
    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    
    return args 