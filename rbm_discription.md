# 영화 추천 시스템 (RBM 기반)

## 개요
이 코드드는 [Salakhutdinov et al. (2007)](https://grouplens.org/datasets/movielens/) 논문을 기반으로 Restricted Boltzmann Machine(RBM)을 활용한 협업 필터링 기반 영화 추천 시스템을 구현합니다. 본 시스템은 Netflix 데이터셋과 같은 대규모 사용자-아이템 행렬을 효과적으로 학습하여 개인화된 영화 추천을 제공합니다.

## 주요 기능
- **데이터 전처리**
  - MovieLens 데이터셋에서 영화 평점 및 메타데이터 로드
  - 데이터 전처리 및 훈련/테스트 데이터 분할
- **추천 모델 구현**
  - RBM (Restricted Boltzmann Machine)
  - Conditional RBM
  - Conditional Factored RBM
- **학습 및 최적화**
  - Contrastive Divergence (CD) 알고리즘을 활용한 학습
  - GPU 최적화 및 미니배치 학습 적용
- **추천 품질 평가**
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
  - HR (Hit Rate)
  - cHR (Cumulative Hit Rate)
  - AHAR (Average Hit Rank)
  - Coverage (추천 범위)
  - Diversity (추천 다양성)
  - Novelty (추천 참신성)

## 설치
### 필수 라이브러리
- Python 3.x
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Surprise (scikit-surprise)

### 설치 방법
1. 저장소를 클론합니다:
   ```sh
   git clone https://github.com/your-repo/movie-recommendation-system.git
   cd movie-recommendation-system
   ```
2. 필요한 패키지를 설치합니다:
   ```sh
   pip install -r requirements.txt
   ```
   또는 수동 설치:
   ```sh
   pip install torch numpy pandas matplotlib scikit-surprise
   ```
3. MovieLens 데이터셋 (`ratings.csv`, `movies.csv`)을 다운로드하여 적절한 디렉토리에 배치합니다.

## 사용법
### 1. 데이터 로드 및 전처리
```python
ml = MovieLens("/path/to/ratings.csv", "/path/to/movies.csv")
surprise_dataset = ml.loadMovieLensLatestSmall()
data_list = surprise_dataset_to_list(surprise_dataset)
train_data, test_data = train_test_split(data_list, test_ratio=0.2, seed=42)
```

### 2. 추천 모델 학습
#### a. RBM 모델 학습
```python
rbm_model = RBM_PT(num_users, num_items, K=5, F=50, learning_rate=0.01, epochs=5, batch_size=128)
rbm_model.train(train_data)
```
#### b. Conditional RBM 모델 학습
```python
crbm_model = ConditionalRBM_PT(num_users, num_items, K=5, F=50, lr=0.01, epochs=5, batch_size=128)
crbm_model.train(train_data)
```
#### c. Conditional Factored RBM 모델 학습
```python
cfrbm_model = ConditionalFactoredRBM_PT(num_users, num_items, K=5, F=50, C=10, lr=0.01, epochs=5, batch_size=128)
cfrbm_model.train(train_data)
```

### 3. 추천 품질 평가
```python
rbm_res = evaluate_model(rbm_model, train_data, test_data, "RBM", top_k_candidates=2000, N=10)
crbm_res = evaluate_model(crbm_model, train_data, test_data, "ConditionalRBM", top_k_candidates=2000, N=10)
cfrbm_res = evaluate_model(cfrbm_model, train_data, test_data, "CondFactoredRBM", top_k_candidates=2000, N=10)
```

### 4. 특정 사용자 추천 영화 출력
```python
sample_user = 1
if sample_user in user_topN:
    recommended_ids = user_topN[sample_user]
    recommended_titles = [ml_obj.movieID_to_name.get(mid, "Unknown") for mid in recommended_ids]
    print(f"[User {sample_user}] 추천 영화 리스트:", recommended_titles)
```

## 결과
각 모델의 평가 결과 예시:
```
=== [RBM] 평가 결과 ===
RMSE = 0.9421
MAE  = 0.7321
HR   = 0.6234
cHR  = 0.5123
AHAR = 5.3245
Coverage  = 0.8723
Diversity = 0.6589
Novelty   = 8.1123
```

## 향후 개선 사항
- RBM의 확장 모델(Deep Belief Networks, Autoencoders) 적용
- Hybrid 추천 시스템 도입 (RBM + Collaborative Filtering)
- 최적의 하이퍼파라미터 튜닝

## 참고 문헌
- Salakhutdinov, R., Mnih, A., & Hinton, G. (2007). "Restricted Boltzmann Machines for Collaborative Filtering." Proceedings of the 24th International Conference on Machine Learning.

## 라이선스
이 프로젝트는 MIT 라이선스를 따릅니다.

## 감사의 말
- [GroupLens Research](https://grouplens.org/)에서 제공한 MovieLens 데이터셋을 사용했습니다.
- 협업 필터링 모델 구현을 위해 Surprise 및 PyTorch를 활용했습니다.


