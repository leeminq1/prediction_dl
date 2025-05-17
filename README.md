# SMP (계통한계가격) 예측 모델

제주 지역 SMP(계통한계가격) 예측을 위한 딥러닝 모델 구현 프로젝트입니다.

## 프로젝트 구조

```
├── data/                   # 원본 데이터 파일 (2023-2025 SMP 데이터)
├── data_processed/         # 전처리된 데이터 파일
├── models/                 # 딥러닝 모델 구현
│   ├── gru_model.py       # GRU 모델
│   ├── lstm_model.py      # LSTM 모델
│   ├── transformer_model.py# Transformer 모델
│   └── model_factory.py   # 모델 생성 팩토리
├── utils/                  # 유틸리티 함수
│   ├── data_loader.py     # 데이터 로더
│   ├── data_preprocessor.py# 데이터 전처리기
│   └── metrics_logger.py  # 학습 지표 로거
├── weights_best_model/     # 최고 성능 모델 가중치
├── main.py                # 학습 실행 파일
├── main_test.py          # 추론 실행 파일
├── config.py             # 설정 파일
└── requirements.txt      # 의존성 패키지
```

## 주요 기능

### 1. 데이터 전처리
- Excel 파일의 시간별 SMP 데이터 처리
- 이상치 처리 (IQR 방식)
- 데이터 정규화
- 전처리된 데이터를 pkl 형식으로 저장

### 2. 모델 구현
- LSTM, GRU, Transformer 모델 지원
- ModelFactory 패턴을 통한 모델 생성
- 시계열 예측을 위한 구조 설계

### 3. 학습 설정
- Sequence Length: 120 (5일)
- Target Length: 24 (24시간)
- Train Stride: 1
- Test Stride: 24
- 학습 데이터: 2023-2024년
- 테스트 데이터: 2025년

### 4. 추론 기능
- 저장된 최고 성능 모델 로드
- 테스트 데이터에 대한 예측 수행
- 예측 결과 CSV 저장 및 시각화

## 설치 방법

1. 저장소 클론
```bash
git clone https://github.com/leeminq1/prediction_dl.git
cd prediction_dl
```

2. 의존성 패키지 설치
```bash
pip install -r requirements.txt
```

## 사용 방법

### 1. 학습
```bash
python main.py --model_type [lstm/gru/transformer]
```

### 2. 추론
```bash
python main_test.py --model_type [lstm/gru/transformer]
```

## 주요 파라미터

`config.py`에서 다음 파라미터를 설정할 수 있습니다:

- `hidden_size`: 히든 레이어 크기
- `num_layers`: 레이어 수
- `dropout`: 드롭아웃 비율
- `d_model`: Transformer 모델 차원
- `nhead`: Transformer 어텐션 헤드 수
- `sequence_length`: 입력 시퀀스 길이
- `target_length`: 예측 시퀀스 길이
- `train_stride`: 학습 데이터 스트라이드
- `test_stride`: 테스트 데이터 스트라이드

## 결과

예측 결과는 다음 위치에 저장됩니다:
- CSV 파일: `test_results/{model_type}_{timestamp}/predictions.csv`
- 시각화: `test_results/{model_type}_{timestamp}/predictions_plot.png` 