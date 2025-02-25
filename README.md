# RBM, Conditional RBM, Conditional Factored RBM Implementation

본 레포지토리는 **Salakhutdinov, Mnih, Hinton (2007)**의 논문  
["Restricted Boltzmann Machines for Collaborative Filtering"](http://www.cs.toronto.edu/~rsalakhu/papers/rbmcf.pdf)에서  
제시된 아이디어를 바탕으로, **PyTorch**를 활용하여 **(1) RBM**, **(2) Conditional RBM**, **(3) Conditional Factored RBM**을 구현하고,  
MovieLens 데이터셋(ml-latest-small)으로 학습 및 평가하는 과정을 담고 있습니다.

---

## 1. Overview

1. **협업 필터링(Collaborative Filtering)**  
   - 사용자 × 아이템 평점을 예측하거나, Top-N 추천을 생성하기 위해 자주 사용되는 접근입니다.

2. **논문 요약**  
   - 기존 SVD 같은 선형 방식과 달리, RBM은 **비선형** 잠재 표현을 학습하여 더욱 강력한 모델을 형성합니다.  
   - **Conditional RBM**은 "사용자가 어떤 영화를 보았는지(rated/unrated)" 정보를 추가 반영합니다.  
   - **Factored RBM**은 파라미터 \(\mathbf{W}\)를 두 행렬 \(\mathbf{A}\), \(\mathbf{B}\)로 분해해 대규모 데이터에서의 파라미터 효율성을 높입니다.

3. **코드 요약**  
   - **PyTorch**에서 **mini-batch + GPU**를 사용하여 RBM 학습(Contrastive Divergence)을 구현했습니다.  
   - Top-N 추천 품질 지표(HR, cHR, AHAR, Coverage, Diversity, Novelty)와 예측 정확도(RMSE, MAE)를 함께 측정합니다.  
   - 평가 시간 단축을 위해, **아이템 후보군**(인기도 상위 N개)만 대상으로 Top-N을 생성하고, 한 번 구한 결과를 모든 지표가 공유하도록 최적화하였습니다.

---

## 2. File Description

- **Cell 1**: 임포트 & 환경 설정 (GPU, 멀티스레딩)  
- **Cell 2**:  
  - MovieLens 데이터 로더 및 Surprise → List 변환  
  - Train/Test 분할  
- **Cell 3**:  
  - 평가 지표(RMSE, MAE),  
  - 아이템 후보군(prune_items_by_popularity),  
  - 사용자별 Top-N 한 번만 계산하고 지표를 재사용하는 로직  
- **Cell 4**: **`RBM_PT`** 클래스  
  - (num_items, K) Visible 소프트맥스, (F) Hidden, 파라미터(W, bv, bh), Contrastive Divergence  
- **Cell 5**: **`ConditionalRBM_PT`** 클래스  
  - RBM에 rated/unrated 벡터(r) → Hidden에 추가 입력  
  - 파라미터(D) 추가  
- **Cell 6**: **`ConditionalFactoredRBM_PT`** 클래스  
  - `W = A×B`(Factored), plus rated/unrated  
  - 파라미터(A, B, D)  
- **Cell 7**: 전체 모델 학습, 평가, 시각화

---

## 3. Requirements

- Python 3.7+  
- PyTorch  
- scikit-surprise  
- NumPy, Matplotlib 등

---

## 4. How to Run

1. **MovieLens 데이터 다운로드**  
   - 예: `ml-latest-small`(https://grouplens.org/datasets/movielens/)  
   - `ratings.csv`, `movies.csv` 파일을 준비
2. **Cell** 순서대로 실행  
3. 각 모델(RBM, cRBM, cFRBM)이 5 Epoch씩 학습 완료 후,  
   Top-N 추천을 포함한 평가 결과를 출력합니다.

---

## 5. Key Implementation Details

1. **Mini-batch & GPU**  
   - 사용자별로 partial RBM을 구성하는 대신,  
   - 여러 사용자(`batch_size`)를 묶어, (B, num_items, K) 크기의 텐서를 한 번에 처리합니다.  
   - 파라미터 업데이트 시 PyTorch의 텐서 연산과 `einsum`으로 벡터화 연산을 수행합니다.
2. **Conditional RBM**  
   - rated/unrated 벡터 **r**를 (B, num_items) 형태로 추가 입력  
   - **D** 파라미터가 r[i] = 1일 때 Hidden에 추가적인 bias로 작용  
3. **Factored RBM**  
   - `W_{i,k,f} = sum_c( A_{i,k,c} * B_{c,f} )`  
   - 파라미터 수 감소, 대규모 데이터에 대한 확장성 증대  
4. **Evaluation**  
   - RMSE, MAE: 평점 예측 정확도  
   - HR, cHR, AHAR, Coverage, Diversity, Novelty: Top-N 추천 품질  
5. **Speed Optimization**  
   - 후보군(pruned) 아이템만 Top-N 계산 → 많은 아이템에 대한 `predict` 비용 절감  
   - 한 번 구한 `user_topN`을 재활용해 여러 지표를 계산 → 중복 루프 제거

---

## 6. Execution Log & Interpretation

### **학습(Reconstruction Error)**
- **(1) RBM**
[Epoch 1/5] Recon Error: 110058174.000 [Epoch 2/5] Recon Error: 110058165.000 [Epoch 3/5] Recon Error: 110057907.000 [Epoch 4/5] Recon Error: 110057613.000 [Epoch 5/5] Recon Error: 110057442.000
- Reconstruction Error가 약간씩 감소(1.10058e8 → 1.10057e8).  
- 절대값이 크지만, 데이터 크기·원핫 구조를 고려하면 Epoch별 감소 추세가 관건입니다.

- **(2) Conditional RBM**
[Epoch 1/5] Recon Error: 110058399.000 [Epoch 2/5] Recon Error: 110057839.000 [Epoch 3/5] Recon Error: 110057442.000 [Epoch 4/5] Recon Error: 110056310.000 [Epoch 5/5] Recon Error: 110055628.000
- 마찬가지로 1.10058e8 부근에서 시작해 점차 감소(마지막 1.100556e8).

- **(3) Conditional Factored RBM**
[Epoch 1/5] Recon Error: 110059666.000 [Epoch 2/5] Recon Error: 110060406.000 [Epoch 3/5] Recon Error: 110058679.000 [Epoch 4/5] Recon Error: 110057809.000 [Epoch 5/5] Recon Error: 110056169.000
- Factored 구조도 유사 범위의 Recon Error  
- Epoch 2에서는 소폭 증가(1.10060406e8) 후 감소. 일부 초기화·학습률 설정에 따라 오르내릴 수 있음.

**⇒** 전반적으로, **미니배치 CD**로 인한 잡음(진동)이 있긴 하지만, 전체적으로 조금씩 감소 추세를 보이며 학습이 진행됩니다.

### **평가 결과(RMSE, MAE, HR, cHR, AHAR, Coverage, Diversity, Novelty)**

1. **RBM**
RMSE = 1.1836 MAE = 0.9673 HR = 0.0675 cHR = 1.0817 AHAR = 10.5694 Coverage = 0.0049 Diversity = 0.8270 Novelty = 20.6653
- RMSE, MAE: 1.18, 0.97 정도  
- HR 6.75%, Coverage 0.49%, Diversity 0.827, Novelty 20.66

2. **Conditional RBM**
RMSE = 1.1843 MAE = 0.9678 HR = 0.0345 cHR = 0.5532 AHAR = 10.7809 Coverage = 0.0024 Diversity = 0.8010 Novelty = 514.7209
- 예측 정확도는 RBM과 유사 수준  
- HR은 3.45%로 다소 낮지만, Novelty가 매우 높게(514.72) 나옴.  
  - 이는 후보군에서 인기 없는 아이템들이 많이 추천되어 '참신성'(Novelty)은 높지만, 히트율은 떨어지는 경향을 의미.

3. **Conditional Factored RBM**
RMSE = 1.1823 MAE = 0.9698 HR = 0.0422 cHR = 0.6764 AHAR = 10.7076 Coverage = 0.0036 Diversity = 0.8318 Novelty = 102.6848
- RMSE 1.1823, MAE 0.97 전후로 RBM 대비 큰 차이는 없음.  
- HR 4.22%, Coverage 0.36%, Diversity 0.83.  
- Novelty 102.68로 Conditional RBM보다는 낮으나 RBM보다 훨씬 높은 편.  
  - "Conditional"로 인해 unrated 정보가 들어가면서, 좀 더 '비인기 아이템'도 추천하는 경향이 있는 것으로 추정.

**⇒** 전반적으로,  
- RMSE, MAE 차이는 세 모델 간 크지 않으나,  
- **Conditional** 구조가 Novelty 측면에서 매우 높은 점수를 보이는 등,  
- 추천 특성이 달라지고 있음을 확인할 수 있습니다.

---

## 7. License

- 이 코드는 자유롭게 수정·재배포가 가능합니다. (MIT or Apache 2.0 등 필요시 선택)
- MovieLens 데이터는 [GroupLens Research](https://grouplens.org/datasets/movielens/) 라이선스 약관에 따릅니다.

---

## Reference

- Salakhutdinov, R., Mnih, A., & Hinton, G. (2007). [**Restricted Boltzmann Machines for Collaborative Filtering**](http://www.cs.toronto.edu/~rsalakhu/papers/rbmcf.pdf). *Proceedings of the 24th International Conference on Machine Learning (ICML)*.
- [PyTorch 공식 문서](https://pytorch.org/)