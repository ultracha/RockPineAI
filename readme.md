# Rock Pine 최적 생육 환경 추천 시스템

Rock Pine의 최적 생육 환경을 추천하기 위한 머신러닝 모델입니다.

## 기능

- CSV 파일 업로드 및 데이터 분석
- 동적 컬럼 선택 (입력 변수 및 타겟 변수)
- 높이 예측 회귀 모델 및 건강 상태 분류 모델 학습
- 최적 환경 조합 추천
- 결과 시각화 및 다운로드

## 설치

```bash
pip install -r requirements.txt
```

## 사용 방법

### 웹 앱 실행 (권장)

```bash
streamlit run app.py
```

브라우저에서 자동으로 열리며, 다음 단계를 따라주세요:

1. 사이드바에서 CSV 파일 업로드
2. "컬럼 선택" 탭에서 입력 변수와 타겟 변수 선택
3. "모델 학습" 탭에서 모델 학습 시작
4. "결과 확인" 탭에서 추천 결과 확인

### CLI 실행

```bash
python rock_pine_recommender.py --data-path /path/to/data.csv --top-k 5 --output-json output.json --plot-dir plots/
```

## 데이터 형식

CSV 파일은 다음 컬럼을 포함해야 합니다:

- **입력 변수**: Soil_Type, Fertilizer, Plant_Age, Dripper_Count, Water_Daily_cc, Ventilation, Temp_Low, Temp_High, Humidity_Low, Humidity_High, CO2_Low, CO2_High
- **타겟 변수**: Height_cm (높이), Health_Status (건강 상태)

## 모델

- **높이 예측**: RandomForestRegressor (300 trees)
- **건강 상태 분류**: RandomForestClassifier (400 trees, balanced class weights)
