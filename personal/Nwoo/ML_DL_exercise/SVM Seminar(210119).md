## SVM Seminar(210119)

### 전체적인 SVM 사용방법

* ##### 데이터 전처리 후 train data와 test data를 나눈다.

* ##### Pipeline를 통해 Scaler와 SVM을 연결시켜 PipeLine 객체를 만든다.

* ##### SVM에서  최적의 Hyper Parameter를 설정하기 위해 GridSearch를 실행한다.

### SVM에 맞지 않는 Data

* 범주형 데이터(카테고리형 데이터)는 SVM 보다는 의사결정 트리 알고리즘이 더 어울림

### Kernel 사용목적

* 저차원의 비선형 관계의 데이터들의 분류가 어려울 때 Kernel함수를 통해 고차원으로 이동하여 선형적으로 분류하여 저차원으로 mapping 시킨다.

### Kernel 종류

* #### Linear

* #### RBF(Ridial Basis Function)

* #### Poly(다항 커널)

* #### Sigmoid

가장 보편적으로 쓰이는 Kernel은 가우시안 커널(RBF)이고 각각의 Kernel특성이 다르지만 역할은 비슷하므로 

Kernel 각각 실행시켜 가장 정확도가 높은 Kernel을 쓰자

### 불편사항

* **Feature들이 많은 다차원의 데이터들의 관계가 선형적으로 분류가 되는 지 되지 않는 지 알기 어렵다.**

  => kernel svm 적용 조건을 알기 힘들었다.

* **범주형 데이터들의 경우 데이터들이 이산적이기 때문에 다른 연속적인 Feature간 관계를 파악하는 것이 제한적이다. 따라 SVM 적용시 분류 효율이 떨어진다.**

* **kernel SVM 적용시 어떤 kernel을 사용해야 할지 결정하기 어렵다.**

  **주로 rbf를 사용하지만 각각의 kernel들이 어떤 특징을 가지는 지 등을 판단하기 어렵다.**

* **C나 gamma 같은 hyper parameter들을 gridsearch로 tuning할 수 있지만 처음 사용하는 입장에서 볼 때 **

  **직관적을 받아드리기 힘들다.**











