import torch
import numpy as np
from pathlib import Path
import random
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from models.model_factory import ModelFactory
from utils.data_preprocessor import SMPDataPreprocessor
from utils.data_loader import TimeSeriesDataLoader
from utils.metrics_logger import MetricsLogger
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
    print("\n=== SMP 예측 모델 학습 시작 ===")
    
    # 설정 로드
    args = get_args()
    print("\n설정 로드 완료")
    
    # 시드 설정
    set_seed(args.seed)
    print(f"시드 설정: {args.seed}")
    
    # 데이터 전처리
    print("\n데이터 전처리 시작...")
    preprocessor = SMPDataPreprocessor(data_dir=args.data_dir, save_dir=args.processed_dir)
    
    # 학습 데이터와 테스트 데이터 처리
    all_years = args.train_years + args.test_years
    processed_data = preprocessor.process_all_years(all_years)
    
    # 학습 데이터 준비
    print("\n학습 데이터 준비 중...")
    train_years_data = [processed_data[year] for year in args.train_years]
    
    # 전체 학습 기간의 정규화 파라미터 계산
    all_train_prices = np.concatenate([data['price'].values for data in train_years_data])
    scaler_mean = all_train_prices.mean()
    scaler_std = all_train_prices.std()
    print(f"\n정규화 파라미터 (전체 학습 기간 기준):")
    print(f"- 평균: {scaler_mean:.2f}")
    print(f"- 표준편차: {scaler_std:.2f}")
    
    # 스케일링 파라미터 저장
    scaler_params = {
        'mean': scaler_mean,
        'std': scaler_std
    }

    # 정규화된 데이터 준비
    train_data = np.concatenate([
        (data['price'].values - scaler_mean) / scaler_std 
        for data in train_years_data
    ])
    print(f"학습 데이터 준비 완료: {len(train_data)} 샘플")
    
    # 데이터 로더 초기화
    print("\n데이터 로더 초기화...")
    data_loader = TimeSeriesDataLoader({
        'sequence_length': args.sequence_length,
        'target_length': args.target_length,
        'batch_size': args.batch_size,
        'val_ratio': args.val_ratio,
        'train_stride': args.train_stride,
        'test_stride': args.test_stride
    })
    
    # 학습/검증 데이터 로더 생성
    train_loader, val_loader = data_loader.prepare_data(train_data)
    
    print(f"\n데이터셋 정보:")
    print(f"입력 시퀀스 길이: {args.sequence_length} 시간")
    print(f"예측 시퀀스 길이: {args.target_length} 시간")
    print(f"학습 데이터: {len(train_loader.dataset)} 샘플")
    print(f"검증 데이터: {len(val_loader.dataset)} 샘플")
    print(f"학습 stride: {args.train_stride} 시간")
    print(f"테스트 stride: {args.test_stride} 시간")
    
    # 모델 생성
    print("\n모델 생성 중...")
    model_config = {
        'input_size': 1,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'output_size': 1,
        'target_length': args.target_length,
        # Transformer 모델 전용 파라미터 추가
        'd_model': args.d_model,
        'nhead': args.nhead
    }
    model = ModelFactory.create_model(args.model_type, model_config)
    print(f"모델 생성 완료 (타입: {args.model_type})")
    
    # 학습 설정
    device = torch.device(args.device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.MSELoss()
    
    # 메트릭 로거 초기화
    metrics_logger = MetricsLogger(args.experiment_dir)
    metrics_logger.set_config(args)  # 설정 정보 전달
    
    print(f"\n학습 설정:")
    print(f"모델: {args.model_type}")
    print(f"디바이스: {device}")
    print(f"학습률: {args.learning_rate}")
    print(f"배치 크기: {args.batch_size}")
    print(f"에포크: {args.epochs}")
    
    # 학습 루프
    print("\n=== 학습 시작 ===")
    early_stopping_counter = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs} 시작...")
        epoch_start_time = time.time()
        model.train()
        train_losses = []
        
        # 학습
        print("학습 단계...")
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            if batch_idx % 100 == 0:
                print(f"- 배치 {batch_idx}/{len(train_loader)} 처리 중")
            
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # 검증
        print("검증 단계...")
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_idx, (batch_x, batch_y) in enumerate(val_loader):
                if batch_idx % 50 == 0:
                    print(f"- 검증 배치 {batch_idx}/{len(val_loader)} 처리 중")
                
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                val_loss = criterion(outputs, batch_y)
                val_losses.append(val_loss.item())
        
        # 메트릭 계산 및 로깅
        epoch_time = time.time() - epoch_start_time
        metrics = {
            'train_loss': np.mean(train_losses),
            'val_loss': np.mean(val_losses),
            'learning_rate': args.learning_rate,
            'epoch_times': epoch_time
        }
        
        is_best = metrics_logger.log_metrics(metrics, epoch)
        
        print(f"\nEpoch {epoch+1}/{args.epochs} 결과:")
        print(f"Train Loss: {metrics['train_loss']:.4f}")
        print(f"Val Loss: {metrics['val_loss']:.4f}")
        print(f"Epoch Time: {epoch_time:.2f}s")
        
        # 최고 성능 모델 저장
        if is_best:
            print("\n새로운 최고 성능 달성! 모델 저장 중...")
            model_save_path = args.experiment_dir / 'best_model.pth'
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': metrics['train_loss'],
                'val_loss': metrics['val_loss'],
                'scaler_params': scaler_params,  # 스케일링 파라미터 추가
                # 모델 설정 저장
                'model_config': {
                    'model_type': args.model_type,
                    'input_size': model_config['input_size'],
                    'hidden_size': args.hidden_size,
                    'num_layers': args.num_layers,
                    'dropout': args.dropout,
                    'output_size': model_config['output_size'],
                    'target_length': args.target_length,
                    'd_model': args.d_model,  # Transformer 파라미터 추가
                    'nhead': args.nhead      # Transformer 파라미터 추가
                },
                # 학습 설정 저장
                'training_config': {
                    'learning_rate': args.learning_rate,
                    'batch_size': args.batch_size,
                    'sequence_length': args.sequence_length,
                    'target_length': args.target_length,
                    'train_stride': args.train_stride,
                    'test_stride': args.test_stride,
                    'val_ratio': args.val_ratio,
                    'early_stopping': args.early_stopping,
                    'device': str(args.device),
                    'seed': args.seed
                }
            }
            torch.save(save_dict, model_save_path)
            
            # weights_best_model 디렉토리에 복사
            best_model_dir = Path('weights_best_model') / args.model_type
            best_model_dir.mkdir(parents=True, exist_ok=True)
            best_model_path = best_model_dir / 'best_model.pth'
            torch.save(save_dict, best_model_path)
            
            print(f"모델 저장 완료:")
            print(f"- 실험 디렉토리: {model_save_path}")
            print(f"- 최고 성능 디렉토리: {best_model_path}")
            
            # 예측 결과 저장 및 시각화
            print("\n예측 결과 생성 중...")
            model.eval()
            with torch.no_grad():
                # 검증 세트의 첫 번째 배치에 대해 예측 수행
                val_batch_x, val_batch_y = next(iter(val_loader))
                val_batch_x, val_batch_y = val_batch_x.to(device), val_batch_y.to(device)
                predictions = model(val_batch_x)
                
                # 첫 번째 샘플의 예측값과 실제값 추출
                pred_sample = predictions[0].cpu().numpy()
                actual_sample = val_batch_y[0].cpu().numpy()
                
                # 현재 시간을 기준으로 시작 날짜 설정
                start_date = datetime.now()
                
                # 예측 결과 저장 및 시각화
                save_predictions(pred_sample, actual_sample, start_date, args.experiment_dir,
                               scaler_mean, scaler_std)
            
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            print(f"\n성능 향상 없음. Early stopping counter: {early_stopping_counter}/{args.early_stopping}")
        
        # 조기 종료 체크
        if early_stopping_counter >= args.early_stopping:
            print(f"\n{args.early_stopping}번 연속으로 성능 향상이 없어 학습을 종료합니다.")
            break
        
        print("-" * 50)
    
    # 최종 메트릭 저장
    print("\n최종 메트릭 저장 중...")
    metrics_logger.save_metrics()
    
    # 학습 요약 출력
    summary = metrics_logger.get_summary()
    print("\n=== 학습 요약 ===")
    print(f"최고 검증 손실: {summary['best_val_loss']:.4f} (에포크 {summary['best_epoch']})")
    print(f"최종 학습 손실: {summary['latest_train_loss']:.4f}")
    print(f"최종 검증 손실: {summary['latest_val_loss']:.4f}")
    print(f"총 에포크 수: {summary['total_epochs']}")
    print("\n=== 학습 완료 ===")

if __name__ == "__main__":
    main() 