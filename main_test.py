import torch
import numpy as np
from pathlib import Path
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from models.model_factory import ModelFactory
from utils.data_preprocessor import SMPDataPreprocessor
from utils.data_loader import TimeSeriesDataLoader
from config import get_args

def set_seed(seed):
    """재현성을 위한 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def save_predictions(predictions: np.ndarray, actual_values: np.ndarray, 
                    start_date: datetime, save_dir: Path,
                    scaler_mean: float, scaler_std: float):
    """예측 결과를 CSV로 저장하고 시각화합니다."""
    print("\n예측 결과 저장 및 시각화 시작...")
    
    # 정규화 복원
    predictions_original = predictions * scaler_std + scaler_mean
    actual_values_original = actual_values * scaler_std + scaler_mean
    
    print(f"정규화 복원 완료:")
    print(f"- 평균값: {scaler_mean:.2f}")
    print(f"- 표준편차: {scaler_std:.2f}")
    print(f"- 예측값 범위: {predictions_original.min():.2f} ~ {predictions_original.max():.2f}")
    print(f"- 실제값 범위: {actual_values_original.min():.2f} ~ {actual_values_original.max():.2f}")
    
    # 날짜 인덱스 생성
    dates = [start_date + timedelta(hours=i) for i in range(len(predictions))]
    
    # DataFrame 생성
    df = pd.DataFrame({
        'datetime': dates,
        'predicted': predictions_original.flatten(),
        'actual': actual_values_original.flatten()
    })
    df.set_index('datetime', inplace=True)
    
    # CSV 저장
    csv_path = save_dir / 'predictions.csv'
    df.to_csv(csv_path)
    print(f"예측 결과가 {csv_path}에 저장되었습니다.")
    
    # 시각화
    plt.figure(figsize=(15, 7))
    sns.lineplot(data=df, markers=True, dashes=False)
    plt.title('SMP 예측 결과 비교')
    plt.xlabel('날짜')
    plt.ylabel('SMP 가격 (원/kWh)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # 그래프 저장
    plot_path = save_dir / 'predictions_plot.png'
    plt.savefig(plot_path)
    plt.close()
    print(f"시각화 결과가 {plot_path}에 저장되었습니다.")

def main():
    print("\n=== SMP 예측 모델 추론 시작 ===")
    
    # 설정 로드
    args = get_args()
    print("\n설정 로드 완료")
    
    # 시드 설정
    set_seed(args.seed)
    print(f"시드 설정: {args.seed}")
    
    # 데이터 전처리
    print("\n데이터 전처리 시작...")
    preprocessor = SMPDataPreprocessor(data_dir=args.data_dir, save_dir=args.processed_dir)
    
    # 테스트 데이터 처리
    processed_data = preprocessor.process_all_years(args.test_years)
    test_data = processed_data[args.test_years[0]]  # 첫 번째 테스트 연도 사용
    
    print(f"\n테스트 데이터 정보:")
    print(f"데이터 기간: {test_data.index[0]} ~ {test_data.index[-1]}")
    print(f"데이터 개수: {len(test_data)}")
    
    # 모델 설정
    print("\n모델 생성 중...")
    model_config = {
        'input_size': 1,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'output_size': 1,
        'target_length': args.target_length,
        'd_model': args.d_model,
        'nhead': args.nhead
    }

    # 모델 로드
    model_path = Path('weights_best_model') / args.model_type / 'best_model.pth'
    if not model_path.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

    model = ModelFactory.create_model(args.model_type, model_config)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    scaler_params = checkpoint['scaler_params']
    print(f"모델 로드 완료: {model_path}")
    
    # 테스트 데이터 정규화
    test_data_normalized = (test_data['price'].values - scaler_params['mean']) / scaler_params['std']
    
    # 데이터 로더 초기화
    print("\n데이터 로더 초기화...")
    data_loader = TimeSeriesDataLoader({
        'sequence_length': args.sequence_length,
        'target_length': args.target_length,
        'batch_size': 1,  # 추론시에는 배치 크기 1 사용
        'val_ratio': 0.0,  # 검증 데이터 사용하지 않음
        'train_stride': args.test_stride,
        'test_stride': args.test_stride
    })
    
    # 테스트 데이터 로더 생성
    test_loader = data_loader.prepare_test_data(test_data_normalized)
    
    print(f"\n데이터셋 정보:")
    print(f"입력 시퀀스 길이: {args.sequence_length} 시간")
    print(f"예측 시퀀스 길이: {args.target_length} 시간")
    print(f"테스트 stride: {args.test_stride} 시간")
    
    # 추론 설정
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()
    
    print(f"\n추론 설정:")
    print(f"모델: {args.model_type}")
    print(f"디바이스: {device}")
    
    # 추론 수행
    print("\n=== 추론 시작 ===")
    all_predictions = []
    all_actuals = []
    
    with torch.no_grad():
        for batch_idx, (batch_x, batch_y) in enumerate(test_loader):
            if batch_idx % 10 == 0:
                print(f"- 배치 {batch_idx}/{len(test_loader)} 처리 중")
            
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            
            predictions = outputs.cpu().numpy()
            actuals = batch_y.numpy()
            
            all_predictions.append(predictions[0])
            all_actuals.append(actuals[0])
    
    # 결과 저장
    predictions = np.concatenate(all_predictions)
    actuals = np.concatenate(all_actuals)
    
    # 결과 저장 디렉토리 생성
    results_dir = Path('test_results') / f"{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 예측 결과 저장 및 시각화
    start_date = test_data.index[args.sequence_length]
    save_predictions(predictions, actuals, start_date, results_dir,
                    scaler_params['mean'], scaler_params['std'])
    
    # 모델 저장 시
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'scaler_params': scaler_params
    }
    torch.save(checkpoint, model_path)
    
    print("\n=== 추론 완료 ===")

if __name__ == "__main__":
    main() 