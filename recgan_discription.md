# RecGAN 기반 영화 추천 시스템

## 개요
이 코드드는 RecGAN(Recurrent Generative Adversarial Network)을 기반으로 영화 추천 시스템을 구현합니다. RecGAN은 순환 신경망(RNN)과 생성적 적대 신경망(GAN)을 결합하여 사용자 선호도의 시간적 변화를 모델링하며, 맞춤형 추천 품질을 향상시킵니다. 본 구현은 논문 *RecGAN: Recurrent Generative Adversarial Networks for Recommendation Systems*를 기반으로 개발되었습니다.

## RecGAN 모델 개요
RecGAN은 RNN(GRU) 기반 생성자(Generator)와 판별자(Discriminator)를 활용하여 사용자-아이템 시퀀스를 학습하고 추천 품질을 개선합니다.

- **Generator**: 사용자의 과거 시퀀스를 바탕으로 향후 선호도를 예측
- **Discriminator**: 생성된 추천이 실제 데이터와 유사한지 판별
- **커스텀 GRU**

## 주요 기능
- **데이터 전처리**
  - MovieLens 데이터셋에서 영화 평점 및 메타데이터 로드
  - 사용자별 시퀀스 데이터 생성
- **RecGAN 모델 구현**
  - 생성자(GRU 기반) 및 판별자(GRU 기반) 구조 설계
  - Leaky ReLU 기반 GRU 업데이트 게이트 도입
  - Collaborative Filtering 메커니즘 반영
- **학습 및 최적화**
  - Adversarial Learning (미니맥스 게임)
  - Backpropagation Through Time (BPTT) 적용
  - GPU 최적화 및 미니배치 학습 지원
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
ml = MovieLens("/path/to/ratings.csv")
df = ml.load_ratings_df()
train_df, test_df = split_train_test(df)
user_seq_train = build_user_sequences(train_df)
user_seq_test  = build_user_sequences(test_df)
```

### 2. RecGAN 학습
#### a. RecGAN1 (기본 GRU) 학습
```python
gen1 = RRecGAN_Generator(num_users, num_items, embed_dim=128, hidden_size=128, gru_type='basic')
dis1 = RRecGAN_Discriminator(num_users, num_items, embed_dim=128, hidden_size=128, gru_type='basic')
gen1, dis1 = train_rrecgan(gen1, dis1, train_loader, num_epochs=10, g_lr=5e-4, d_lr=5e-4, device=device)
```
#### b. RecGAN2 (ReLU 수정 GRU) 학습
```python
gen2 = RRecGAN_Generator(num_users, num_items, embed_dim=128, hidden_size=128, gru_type='modified')
dis2 = RRecGAN_Discriminator(num_users, num_items, embed_dim=128, hidden_size=128, gru_type='modified')
gen2, dis2 = train_rrecgan(gen2, dis2, train_loader, num_epochs=10, g_lr=5e-4, d_lr=5e-4, device=device)
```

### 3. 추천 품질 평가
```python
metrics1 = evaluate_model_rbm_style_for_recgan(gen1, train_data, test_data, all_users, all_items, user_items_dict, popularity_ranks, movie_genres)
metrics2 = evaluate_model_rbm_style_for_recgan(gen2, train_data, test_data, all_users, all_items, user_items_dict, popularity_ranks, movie_genres)
```

### 4. 특정 사용자 추천 영화 출력
```python
sample_user = 10
print_recommendations_for_user(gen2, sample_user, all_items, user_items_dict, popularity_ranks, ml, N=10)
```

## 결과
각 모델의 평가 결과 예시:
```
=== [RecGAN1] 평가 결과 ===
RMSE = 0.9213
MAE  = 0.7123
HR   = 0.6523
cHR  = 0.5231
AHAR = 4.8754
Coverage  = 0.8945
Diversity = 0.6712
Novelty   = 7.9812
```

## 향후 개선 사항
- RecGAN의 확장 모델(Transformer 기반 추천) 적용
- Hybrid 추천 시스템 도입 (RecGAN + Collaborative Filtering)
- 최적의 하이퍼파라미터 튜닝

## 참고 문헌
- Bharadhwaj et al. *RecGAN: Recurrent Generative Adversarial Networks for Recommendation Systems*, RecSys'18.

## 라이선스
이 프로젝트는 MIT 라이선스를 따릅니다.

## 감사의 말
- [GroupLens Research](https://grouplens.org/)에서 제공한 MovieLens 데이터셋을 사용했습니다.
- 협업 필터링 모델 구현을 위해 Surprise 및 PyTorch를 활용했습니다.


