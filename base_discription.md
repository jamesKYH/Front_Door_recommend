# 영화 고전 추천 시스템 

## 개요
이 코드드는 [MovieLens 데이터셋](https://grouplens.org/datasets/movielens/)을 기반으로 협업 필터링 기법을 사용하여 영화 추천 시스템을 구현합니다. 사용자 기반 KNN, 아이템 기반 KNN, SVD 등의 추천 알고리즘을 평가하며, 다양한 평가 지표를 활용하여 성능을 측정합니다.

## 주요 기능
- **데이터 전처리:** MovieLens 평점 및 영화 메타데이터 로드 및 분석
- **추천 모델:** 다양한 협업 필터링 기법 구현
  - 사용자 기반 KNN
  - 아이템 기반 KNN
  - SVD (특이값 분해)
- **평가 지표:**
  - RMSE (평균 제곱근 오차)
  - MAE (평균 절대 오차)
  - 적중률 (HR)
  - 누적 적중률 (cHR)
  - 평균 적중 순위 (AHAR)
  - 커버리지
  - 다양성
  - 참신성
- **영화 추천:**
  - 사용자 맞춤형 Top-N 추천 리스트 생성

## 설치
### 필수 라이브러리
다음의 라이브러리가 필요합니다:
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
2. 필요한 Python 패키지를 설치합니다:
   ```sh
   pip install -r requirements.txt
   ```
   또는 수동으로 설치:
   ```sh
   pip install torch numpy pandas matplotlib scikit-surprise
   ```
3. MovieLens 데이터셋 (`ratings.csv`, `movies.csv`)을 다운로드하여 적절한 디렉토리에 배치합니다.

## 사용법
### 1. 데이터 로드 및 준비
`MovieLens` 클래스를 사용하여 데이터를 로드하고 전처리합니다.
```python
ml = MovieLens("/path/to/ratings.csv", "/path/to/movies.csv")
surprise_dataset = ml.loadMovieLensLatestSmall()
data_list = surprise_dataset_to_list(surprise_dataset)
train_data, test_data = train_test_split(data_list, test_ratio=0.2, seed=42)
```

### 2. 추천 모델 학습 및 평가
#### a. 사용자 기반 KNN 모델
```python
userKNN_model = ClassicRecommender(method='userKNN')
userKNN_model.train(train_data)
preds, trues = [], []
for (u, i, r) in test_data:
    preds.append(userKNN_model.predict(u, i))
    trues.append(r)
```
#### b. 아이템 기반 KNN 모델
```python
itemKNN_model = ClassicRecommender(method='itemKNN')
itemKNN_model.train(train_data)
```
#### c. SVD 모델
```python
svd_model = ClassicRecommender(method='svd')
svd_model.train(train_data)
```

### 3. Top-N 추천 생성
```python
pruned_items = prune_items_by_popularity(all_items, popularity_ranks, top_k=2000)
user_topN = get_topN_for_all_users(svd_model, all_users, pruned_items, user_items_dict, N=10)
```

### 4. 추천 평가
```python
hr_val, chr_val, ahar_val = evaluate_topN_metrics(user_topN, test_data, rating_threshold=4.0, N=10)
cov_val = coverage_with_topN(user_topN, all_items)
div_val = diversity_with_topN(user_topN, movie_genres)
nov_val = novelty_with_topN(user_topN, popularity_ranks)
```

### 5. 특정 사용자 추천 영화 출력
```python
sample_user = 1
if sample_user in user_topN:
    recommended_ids = user_topN[sample_user]
    recommended_titles = [ml_obj.movieID_to_name.get(mid, "Unknown") for mid in recommended_ids]
    print(f"[User {sample_user}] Top-10 추천 영화:", recommended_titles)
```

## 결과
추천 모델의 평가 결과 예시:
```
=== [UserKNN] 평가 결과 ===
RMSE = 0.9432
MAE  = 0.7345
HR   = 0.6213
cHR  = 0.5123
AHAR = 5.2134
Coverage  = 0.8754
Diversity = 0.6543
Novelty   = 8.2345
```

## 향후 개선 사항
- 딥러닝 기반 추천 모델 도입
- 하이브리드 추천 시스템 탐색
- 최적의 하이퍼파라미터 튜닝

## 라이선스
이 프로젝트는 MIT 라이선스를 따릅니다.

## 감사의 말
- [GroupLens Research](https://grouplens.org/)에서 제공한 MovieLens 데이터셋을 사용했습니다.
- 협업 필터링 모델 구현을 위해 Surprise 라이브러리를 활용했습니다.

