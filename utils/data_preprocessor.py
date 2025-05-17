import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import os

class SMPDataPreprocessor:
    def __init__(self, data_dir='data', save_dir='data_processed'):
        self.data_dir = Path(data_dir)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
    def load_and_preprocess(self, year):
        """연도별 SMP 데이터를 로드하고 전처리합니다."""
        try:
            file_path = self.data_dir / f"smp_jeju_{year}.xlsx"
            
            if not file_path.exists():
                raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
            
            print(f"Loading file: {file_path}")
            
            # Excel 파일 읽기
            df = pd.read_excel(file_path)
            print("\n데이터 미리보기:")
            print(df.head())
            print("\n컬럼 목록:", df.columns.tolist())
            
            # 첫 번째 행(헤더)과 불필요한 컬럼 제거
            df = df.iloc[1:].reset_index(drop=True)  # 첫 번째 행 제거
            
            # 날짜 컬럼 처리
            df['datetime'] = pd.to_datetime(df['년도별 계통한계가격 리스트 (원/kWh)'], format='%Y%m%d')
            
            # 시간별 데이터를 long 형식으로 변환
            hour_columns = [f'Unnamed: {i}' for i in range(1, 25)]  # 1h부터 24h까지
            df_melted = df.melt(
                id_vars=['datetime'],
                value_vars=hour_columns,
                var_name='hour',
                value_name='price'
            )
            
            # 시간 정보 추가
            df_melted['hour'] = df_melted['hour'].apply(lambda x: int(x.split(': ')[1]))
            df_melted['datetime'] = df_melted.apply(
                lambda row: row['datetime'] + pd.Timedelta(hours=row['hour']-1),
                axis=1
            )
            
            # price를 숫자형으로 변환
            df_melted['price'] = pd.to_numeric(df_melted['price'], errors='coerce')
            
            # 시간순으로 정렬
            df_melted = df_melted.sort_values('datetime').set_index('datetime')
            
            # hour 컬럼 제거 (더 이상 필요없음)
            df_melted = df_melted.drop('hour', axis=1)
            
            print("\n전처리된 데이터 형태:")
            print(df_melted.head())
            print("\n데이터 정보:")
            print(df_melted.info())
            
            # 결측치 처리
            if df_melted['price'].isna().any():
                print(f"\n결측치 발견: {df_melted['price'].isna().sum()}개")
                df_melted = df_melted.interpolate(method='linear')
            
            # 이상치 처리 (IQR 방식)
            Q1 = df_melted['price'].quantile(0.25)
            Q3 = df_melted['price'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = (df_melted['price'] < lower_bound) | (df_melted['price'] > upper_bound)
            if outliers.any():
                print(f"\n이상치 발견: {outliers.sum()}개")
                df_melted['price'] = df_melted['price'].clip(lower_bound, upper_bound)
            
            print("\n최종 데이터 통계:")
            print(df_melted.describe())
            
            return df_melted
            
        except Exception as e:
            print(f"Error processing data for year {year}: {str(e)}")
            print(f"현재 작업 디렉토리: {os.getcwd()}")
            print(f"데이터 디렉토리 경로: {self.data_dir.absolute()}")
            raise
            
    def save_processed_data(self, df, year):
        """전처리된 데이터를 pickle 형태로 저장합니다."""
        save_path = self.save_dir / f"smp_jeju_{year}_processed.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(df, f)
        print(f"Processed data saved to {save_path}")
    
    def process_all_years(self, years):
        """여러 연도의 데이터를 한번에 처리합니다."""
        processed_data = {}
        for year in years:
            print(f"\nProcessing data for year {year}")
            try:
                df = self.load_and_preprocess(year)
                self.save_processed_data(df, year)
                processed_data[year] = df
            except Exception as e:
                print(f"Warning: {year}년도 데이터 처리 실패 - {str(e)}")
                continue
        
        if not processed_data:
            raise ValueError("처리된 데이터가 없습니다. 데이터 파일을 확인해주세요.")
        
        return processed_data 