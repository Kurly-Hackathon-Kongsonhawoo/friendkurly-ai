## **USER 기반 쇼핑 아이템 추천 AI 알고리즘**
- 개발 기간: 2022.08.16 ~ 2022.09.15
- 배경: 마켓컬리 공모전 과제3 아이디어로 제안한 것을 스터디로 전환하여 실용적인 추천 알고리즘 구현을 해보고자 함
- 목표: "Budget 맞춤형 추천 상품 제공 서비스"에 맞는 추천 알고리즘
 - 방식: 고객이 Budget을 입력하면 그 예산에 맞는 상품을 우선순위로 4개 option을 보여줌
 - 노출: 상품명, 상품 가격 (계산 방식에 따라 1개의 option에 속한 상품의 수는 1개 이상이 됨)

- 활용 데이터셋: 
 1) 기본: https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis
 2) 서브: `orders.csv` 의 user_id를 중복값 제거 후 속성 (성별, 나이, 멤버십 등급, 전월 구매금액, 관심사 1,2)를 추가 -- (최종 모델에는 미반영)

- 참고 e-commerce recommendation system: https://github.com/Kurly-Hackathon-idea/Ecommerce-Recommendation-System.git
- 이번에 구현한 알고리즘 적용 시스템: https://github.com/Kurly-Hackathon-idea/friendkurly-backend.git


### 1. 진행 순서: 
- Raw data import (from instacart dataset in Kaggle)
- Data merge (유저+주문데이터) & EDA, 이상치 재확인
- Data Sampling (1만명): 데이터가 너무 커서 1만명 샘플링하여 모델 학습
- **Deep learning modeling_1**: 
 - Data pre-processing : 유저기반 feature engineering
 - Youtube 후보 생성 모델 학습을 위해 index encoding하여 1개 user-id별 데이터 병합 (코드에서 `user_data`)
 - `user_data` => Train/Test data split (데이터 수 자유롭게 조정 가능)
 - predict_label 부여: 상품코드로 유저별 어떤 아이템을 미래에 살지 랜덤으로 부여
 - 모델 파라미터 정의
 - 커스텀 레이어 클래스 정의: 임베딩 레이어 마스킹 및 L2 정규화에 활용
 - `User`, `Product_hist:구매이력`, `Order_dow:주문요일`, `Order_hour_of_day:주문시간`, `Days_since_prior_order: 주문 후 지난 시간` Input feature로 설정
 - Dense layer 3개, L2 정규화 레이어로 구성
 - Input feature Embedding 후 `tf.keras.layers.Concatenate`로 병합
 - 모델 학습
 - Test data로 모델 예측 결과 확인 (시퀀스 형태로 맞추기 위해 `tf.keras.preprocessing.sequence.pad_sequences`로 형 변환 필요)
 - 2차원 array 형태로 나옴
- 모델을 저장하고,
- 후보가 생성된 모델을 바탕으로 상품별 재주문여부(`reordered`)를 0,1로 like, dislike로 라벨링하여 상품 데이터를 변환
- 이것을 user_id별로 정리하기 (`new_data`)
- Deep learning modeling_2: 순위 자동생성 모델
 - new_data의 input_feature를 `Min-Max Scaler`를 사용하여 평활화
 - 다중 레이어이므로 복잡한 데이터 간 관계, 특성을 반영하기 위해 활성화함수 'ReLU'사용
 - 예시로 학습 후 유저 5명을 위해 상품을 20개씩 추천하는 결과 확인



### 2. 모델 성능 결과
 - **Deep learning modeling_1**: `Accuracy 0.5, Loss 0.79`
 - **Deep learning modeling_2**: `Accuracy 0.7, Loss 0.55`
 - 파라미터 정의에서 embedding_layer dims, layer수 동일, Learning_rate 상이
 - 1을 바탕으로 2를 모델링하여 순위까지 잘 예측하는 것을 확인



### 3. 모델 개발 리뷰
 - 💡 개발로 얻은 것

  1) User기반 딥러닝 모델을 벤치마킹하여 학습 및 예측까지 잘 진행 된 점
  2) 샘플 수와 실제 학습 데이터의 수를 조정하여 예측 성능 개선 가능
  3) 순위까지 고려하지 않는 상황이라면 **후보 생성**만하고 상품데이터와 연결해 랜덤 추천이 가능하며,
  4) 순위까지 알아서 최적화하고 싶은 상황이라면 **순위 모델**로 결과를 나타내기
  5) 캐글 데이터 활용 및 추가적으로 user data의 demo, purchse data feature를 더해 데이터셋을 구성해본 것 (feature engineering)


 - 🔖 아쉬운 점

  1) User의 선호도, 관심, 연령이 추가된 feature로 구성하여 딥러닝 해보니 acc가 매우 낮고, loss값은 매우 높게 나옴
   - 유저 기반으로 데이터셋을 재구성하는 과정에서 같은 값이 반복적으로 들어가다보니 상품과 연관성이 낮아서 학습 성능 결과가 그렇게 나오는 것으로 추정
   - 이를 제대로 반영하기 위해 딥러닝 모델 전 유저 간 유사도를 계산하고 그 유사도를 적용할 수 있는 방법을 찾아봐야 할 듯
   - User dataset만 구성하여 학습하면 개인화 추천에 가깝게 적용될 것이라고 본게 pain point 였다.

  2) Predict label 값 random 구성하여 학습이 맞는 건가? 
   - 유저가 장바구니에 담아 주문한 상품 중에서 label를 부여하는 것으로도 구성해서 학습을 시켰었다.
   - 마찬가지로 modeling의 학습 성능이 0.5이하로 나와서 생각해보니 오히려 한정적인 라벨범위로만 학습되기에 과적합되어 발생한 문제였다.
   - 다시 전체 product_id 범위로 진행하여 전체 과정을 진행했지만 다른 딥러닝 프로젝트에는 어떻게 진행하는 지 더 면밀히 파악해 봐야겠다.

  3) 신규 유저 데이터를 반영하여 결과값을 예측 가능한가?
   - 신규 유저의 demo 속성 (연령, 성별, 선호카테고리 등)을 미리 알 수 있다면 유저 유사도를 라벨링하여 매칭하는 방법으로 개발 가능할 것으로 예상
   - 이번에 개발한 모델로는 기존 유저에 포커스가 되어있다.

