"""
Streamlit 웹 앱: Rock Pine 최적 생육 환경 추천 시스템

CSV 파일을 업로드하고, 학습에 필요한 컬럼을 선택하여 모델을 학습하고 결과를 시각화합니다.
"""

import io
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (classification_report, f1_score,
                             mean_absolute_error, mean_squared_error)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# 페이지 설정
st.set_page_config(
    page_title="Rock Pine 최적 생육 환경 추천",
    page_icon="🌲",
    layout="wide",
)

# 세션 상태 초기화
if "df" not in st.session_state:
    st.session_state.df = None
if "selected_features" not in st.session_state:
    st.session_state.selected_features = []
if "height_target" not in st.session_state:
    st.session_state.height_target = '성장높이' #None
if "health_target" not in st.session_state:
    st.session_state.health_target = '건강상태' #None
if "artifacts" not in st.session_state:
    st.session_state.artifacts = None
if "metrics" not in st.session_state:
    st.session_state.metrics = None
if "recommendations" not in st.session_state:
    st.session_state.recommendations = None


def detect_column_types(df: pd.DataFrame, columns: List[str]) -> Tuple[List[str], List[str]]:
    """컬럼을 categorical과 numeric으로 분류합니다."""
    categorical = []
    numeric = []
    
    for col in columns:
        if col not in df.columns:
            continue
        # 숫자형이지만 고유값이 적으면 categorical로 간주
        if df[col].dtype in ["object", "string", "category"]:
            categorical.append(col)
        #elif df[col].nunique() < 3 and df[col].dtype in ["int64", "int32"]:
        # col '모종 연식'일 경우  categorical로 간주    
        elif col == '모종연식':
            categorical.append(col)
        else:
            numeric.append(col)
    
    return categorical, numeric


def build_preprocessor_dynamic(
    categorical_features: List[str],
    numeric_features: List[str],
) -> ColumnTransformer:
    """동적으로 전처리기를 생성합니다."""
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    transformers = []
    if categorical_features:
        transformers.append(("cat", categorical_pipeline, categorical_features))
    if numeric_features:
        transformers.append(("num", numeric_pipeline, numeric_features))

    preprocessor = ColumnTransformer(transformers=transformers)
    return preprocessor


def train_models_dynamic(
    df: pd.DataFrame,
    feature_columns: List[str],
    height_target: str,
    health_target: str,
    healthy_label: str = "0",
) -> Tuple[dict, dict]:
    """동적 컬럼 선택으로 모델을 학습합니다."""
    features = df[feature_columns].copy()
    height_targets = df[height_target].astype(float)
    health_targets = df[health_target].astype(str)

    # 데이터 타입 분류
    categorical_features, numeric_features = detect_column_types(features, feature_columns)

    # 전처리기 생성
    preprocessor = build_preprocessor_dynamic(categorical_features, numeric_features)

    # 데이터 분할
    (
        x_train,
        x_valid,
        height_train,
        height_valid,
        health_train,
        health_valid,
    ) = train_test_split(
        features,
        height_targets,
        health_targets,
        test_size=0.2,
        random_state=42,
        stratify=health_targets,
    )

    # 높이 예측 모델
    height_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    random_state=42,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    n_jobs=-1,
                ),
            ),
        ],
    )

    # 건강 상태 분류 모델
    health_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=400,
                    random_state=42,
                    class_weight="balanced",
                    min_samples_leaf=2,
                    max_features="sqrt",
                    n_jobs=-1,
                ),
            ),
        ],
    )

    # 모델 학습
    with st.spinner("모델 학습 중..."):
        height_pipeline.fit(x_train, height_train)
        health_pipeline.fit(x_train, health_train)

    # 검증 평가
    height_pred = height_pipeline.predict(x_valid)
    height_mae = mean_absolute_error(height_valid, height_pred)
    height_rmse = mean_squared_error(height_valid, height_pred) #, squared=False)

    health_pred = health_pipeline.predict(x_valid)
    health_f1 = f1_score(
        health_valid == healthy_label,
        health_pred == healthy_label,
        zero_division=0,
    )

    metrics = {
        "height_mae": float(height_mae),
        "height_rmse": float(height_rmse),
        "health_f1": float(health_f1),
        "health_classification_report": classification_report(
            health_valid,
            health_pred,
            digits=3,
            zero_division=0,
        ),
    }

    # 전체 데이터로 재학습
    height_pipeline.fit(features, height_targets)
    health_pipeline.fit(features, health_targets)

    artifacts = {
        "height_pipeline": height_pipeline,
        "health_pipeline": health_pipeline,
        "feature_columns": feature_columns,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
    }

    return artifacts, metrics


def recommend_environments_dynamic(
    artifacts: dict,
    df: pd.DataFrame,
    height_target: str,
    health_target: str,
    healthy_label: str = "0",
    top_k: int = 5,
) -> pd.DataFrame:
    """최적 환경을 추천합니다."""
    feature_columns = artifacts["feature_columns"]
    feature_space = df[feature_columns].drop_duplicates().reset_index(drop=True)

    if feature_space.empty:
        return pd.DataFrame()

    expected_height = artifacts["height_pipeline"].predict(feature_space)
    health_proba_all = artifacts["health_pipeline"].predict_proba(feature_space)
    
    try:
        healthy_idx = list(artifacts["health_pipeline"].named_steps["model"].classes_).index(healthy_label)
        healthy_probability = health_proba_all[:, healthy_idx]
    except ValueError:
        # healthy_label이 없으면 첫 번째 클래스 사용
        healthy_probability = health_proba_all[:, 0]

    recommendations = feature_space.copy()
    recommendations["예상 높이"] = expected_height
    recommendations["건강 확률"] = healthy_probability
    recommendations["score"] = (
        recommendations["예상 높이"] * recommendations["건강 확률"]
    )

    recommendations = recommendations.sort_values(
        by=["건강 확률", "예상 높이", "score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return recommendations.head(top_k)


# 메인 UI
st.title("🌲 Rock Pine 최적 생육 환경 추천 시스템")

# 사이드바: 파일 업로드 및 설정
with st.sidebar:
    st.header("📁 데이터 업로드")
    uploaded_file = st.file_uploader(
        "CSV 파일을 선택하세요",
        type=["csv"],
        help="학습에 사용할 데이터셋을 업로드하세요.",
    )

    if uploaded_file is not None:
        try:
            # 여러 인코딩 시도 (한국어 파일 지원)
            encodings = ['utf-8', 'cp949', 'euc-kr', 'latin-1']
            df = None
            last_error = None
            
            for encoding in encodings:
                try:
                    uploaded_file.seek(0)  # 파일 포인터를 처음으로 리셋
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                    break
                except (UnicodeDecodeError, UnicodeError) as e:
                    last_error = e
                    continue
            
            if df is None:
                raise last_error if last_error else Exception("파일 인코딩을 확인할 수 없습니다.")
            
            st.session_state.df = df
            st.success(f"✅ 데이터 로드 완료: {len(df)} 행, {len(df.columns)} 컬럼")
        except Exception as e:
            st.error(f"❌ 파일 읽기 오류: {str(e)}")
            st.session_state.df = None

    st.divider()

    if st.session_state.df is not None:
        st.header("⚙️ 모델 설정")
        top_k = st.slider("추천 개수", min_value=1, max_value=20, value=5)
        healthy_label = st.text_input("건강 상태 레이블", value="정상", help="정상 상태로 간주할 레이블 값")

# 메인 영역
if st.session_state.df is None:
    st.info("👈 사이드바에서 CSV 파일을 업로드하세요.")
    st.markdown("""
    ### 사용 방법
    1. 사이드바에서 CSV 파일을 업로드합니다
    2. 입력 변수(Features)와 타겟 변수(Targets)를 선택합니다
    3. 모델 학습 버튼을 클릭합니다
    4. 결과를 확인합니다
    """)
else:
    df = st.session_state.df

    # 탭 구성
    tab1, tab2, tab3, tab4 = st.tabs(["📊 데이터 미리보기", "🎯 컬럼 선택", "🤖 모델 학습", "📈 결과 확인"])

    with tab1:
        st.subheader("데이터 미리보기")
        st.dataframe(df.head(20), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("총 행 수", len(df))
        with col2:
            st.metric("총 컬럼 수", len(df.columns))
        
        st.subheader("컬럼 정보")
        st.dataframe(
            pd.DataFrame({
                "컬럼명": df.columns,
                "데이터 타입": df.dtypes.astype(str),
                "결측치 수": df.isnull().sum(),
                "고유값 수": [df[col].nunique() for col in df.columns],
            }),
            use_container_width=True,
        )

    with tab2:
        st.subheader("학습 컬럼 선택")
        
        all_columns = list(df.columns)
        
        # 모든 컬럼을 Categorical과 Numeric으로 분류
        categorical_cols, numeric_cols = detect_column_types(df, all_columns)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 입력 변수 (Features)")
            
            # 이전에 선택된 컬럼들
            prev_selected = set(st.session_state.selected_features if st.session_state.selected_features else [])
            
            selected_features = []
            
            # Categorical 컬럼 섹션
            if categorical_cols:
                st.markdown("#### 📋 Categorical 컬럼")
                for col in categorical_cols:
                    checked = st.checkbox(
                        f"`{col}`",
                        value=col in prev_selected,
                        key=f"cat_{col}",
                        help=f"타입: {df[col].dtype}, 고유값: {df[col].nunique()}개",
                    )
                    if checked:
                        selected_features.append(col)
            
            # Numeric 컬럼 섹션
            if numeric_cols:
                st.markdown("#### 🔢 Numeric 컬럼")
                for col in numeric_cols:
                    checked = st.checkbox(
                        f"`{col}`",
                        value=col in prev_selected,
                        key=f"num_{col}",
                        help=f"타입: {df[col].dtype}, 범위: {df[col].min():.2f} ~ {df[col].max():.2f}",
                    )
                    if checked:
                        selected_features.append(col)
            
            st.session_state.selected_features = selected_features
            
            # 선택 요약
            if selected_features:
                selected_cat, selected_num = detect_column_types(df, selected_features)
                st.info(f"✅ 선택됨: 총 {len(selected_features)}개 (Categorical: {len(selected_cat)}, Numeric: {len(selected_num)})")
            else:
                st.warning("⚠️ 최소 1개 이상의 입력 변수를 선택하세요.")

        with col2:
            st.markdown("### 타겟 변수 (Targets)")
            height_target = st.selectbox(
                "높이 타겟 변수 (Height)",
                options=all_columns,
                index=all_columns.index(st.session_state.height_target) if st.session_state.height_target in all_columns else 0,
                help="예측할 높이 변수를 선택하세요",
            )
            st.session_state.height_target = height_target

            health_target = st.selectbox(
                "건강 상태 타겟 변수 (Health Status)",
                options=all_columns,
                index=all_columns.index(st.session_state.health_target) if st.session_state.health_target in all_columns else 0,
                help="예측할 건강 상태 변수를 선택하세요",
            )
            st.session_state.health_target = health_target

        # 검증
        if selected_features and height_target and health_target:
            if height_target in selected_features:
                st.warning("⚠️ 높이 타겟 변수가 입력 변수에 포함되어 있습니다.")
            if health_target in selected_features:
                st.warning("⚠️ 건강 상태 타겟 변수가 입력 변수에 포함되어 있습니다.")

    with tab3:
        st.subheader("모델 학습 및 평가")
        
        if not st.session_state.selected_features:
            st.warning("⚠️ 먼저 '컬럼 선택' 탭에서 입력 변수를 선택하세요.")
        elif not st.session_state.height_target or not st.session_state.health_target:
            st.warning("⚠️ 타겟 변수를 선택하세요.")
        else:
            if st.button("🚀 모델 학습 시작", type="primary", use_container_width=True):
                try:
                    # 데이터 검증
                    required_cols = (
                        st.session_state.selected_features
                        + [st.session_state.height_target, st.session_state.health_target]
                    )
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    
                    if missing_cols:
                        st.error(f"❌ 다음 컬럼이 데이터에 없습니다: {missing_cols}")
                    else:
                        # 모델 학습
                        artifacts, metrics = train_models_dynamic(
                            df,
                            st.session_state.selected_features,
                            st.session_state.height_target,
                            st.session_state.health_target,
                            healthy_label,
                        )
                        
                        st.session_state.artifacts = artifacts
                        st.session_state.metrics = metrics
                        
                        # 추천 생성
                        recommendations = recommend_environments_dynamic(
                            artifacts,
                            df,
                            st.session_state.height_target,
                            st.session_state.health_target,
                            healthy_label,
                            top_k,
                        )
                        st.session_state.recommendations = recommendations
                        
                        st.success("✅ 모델 학습 완료!")
                        st.balloons()
                        
                except Exception as e:
                    st.error(f"❌ 오류 발생: {str(e)}")
                    st.exception(e)

            # 학습된 모델이 있으면 메트릭 표시
            if st.session_state.metrics:
                st.markdown("### 📊 모델 평가 지표")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "높이 MAE",
                        f"{st.session_state.metrics['height_mae']:.3f}",
                    )
                with col2:
                    st.metric(
                        "높이 RMSE",
                        f"{st.session_state.metrics['height_rmse']:.3f}",
                    )
                with col3:
                    st.metric(
                        "건강 상태 F1",
                        f"{st.session_state.metrics['health_f1']:.3f}",
                    )
                
                st.markdown("#### 분류 리포트")
                st.text(st.session_state.metrics["health_classification_report"])

    with tab4:
        st.subheader("최적 환경 추천 결과")
        
        if st.session_state.recommendations is None or st.session_state.recommendations.empty:
            st.info("👈 '모델 학습' 탭에서 모델을 학습하세요.")
        else:
            recommendations = st.session_state.recommendations
            
            st.markdown(f"### 상위 {len(recommendations)}개 추천 환경")
            st.dataframe(recommendations, use_container_width=True)
            
            # 시각화
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 높이 vs 건강 확률")
                st.scatter_chart(
                    recommendations,
                    x="예상 높이",
                    y="건강 확률",
                    size="score",
                    color="score",
                )
            
            with col2:
                st.markdown("#### 예상 높이 비교")
                st.bar_chart(
                    recommendations.set_index(
                        recommendations.index.map(lambda x: f"환경 {x+1}")
                    )["예상 높이"]
                )
            
            # 상세 정보
            st.markdown("#### 상세 환경 변수")
            for idx, row in recommendations.iterrows():
                with st.expander(f"환경 {idx+1} (점수: {row['score']:.2f})"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("예상 높이 (cm)", f"{row['예상 높이']:.2f}")
                    with col2:
                        st.metric("건강 확률", f"{row['건강 확률']:.3f}")
                    with col3:
                        st.metric("종합 점수", f"{row['score']:.2f}")
                    
                    st.json(row[st.session_state.selected_features].to_dict())
            
            # 다운로드 버튼
            csv = recommendations.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="📥 추천 결과 CSV 다운로드",
                data=csv,
                file_name="rock_pine_recommendations.csv",
                mime="text/csv",
            )

