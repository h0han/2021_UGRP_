# PyTorch로 선형회귀 구현하기

## 커스텀 데이터셋을 클래스로 구현하기

* 굳이 데이터셋을 클래스로 구현하지 않아도 됨
* 클래스로 구현하면 직관적으로 데이터 셋을 이해할 수 있다.
* 기본적인 custom dataset 구조

```python
class CustomDataset(torch.utils.data.Dataset): 
    def init(self): 
    # 데이터셋의 전처리를 해주는 부분
    
    def len(self): 
    # 데이터셋의 길이. 즉, 총 샘플의 수를 적어주는 부분
    
    def getitem(self, idx): 
    # 데이터셋에서 특정 1개의 샘플을 가져오는 함수
```

* 이후 커스텀데이터셋을 불러오기 위해 `torch.utils.data`에 있는 `DataLoader`를 불러와 `batch_size`, `shuffle`등의 파라미터 값을 할당하여 `DataLoader`객체를 생성한다.
* dataloader의 핵심적인 역할
  * 데이터를 배치로 나눠준다. (`batch_size` 파라미터 활용)
  * 데이터를 셔플링해준다. (`shuffle`파라미터를 True로 할당)

>* 참고사항: [데이터셋 구성과 데이터로더 활용하는 과정](https://medium.com/@smha_61749/pytorch-%EB%82%98%EB%A7%8C%EC%9D%98-%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%A1%9C-dataloader-%EC%9E%91%EC%84%B1%ED%95%98%EA%B8%B0-1-c1d785a9b871)



## Model을 클래스로 구현하기

* 마찬가지로 class로 구현하면 모델 전반적인 기능이나 구조를 직관적으로 이해할 수 있다.
* 클래스 구현예제(LinearRegression)

```python
class LinearRegressionModel(nn.Module): # torch.nn.Module을 상속받는 파이썬 클래스
    def __init__(self): #
        super().__init__()
        self.linear = nn.Linear(1, 1) # 단순선형회귀이므로 input_dim=1, output_dim=1.

    def forward(self, x):
        return self.linear(x)
```



## PyTorch를 활용한 전반적인 모델 구성과 모델 학습과정

​	1. **데이터를 다양한 시각화도구및 pandas나 numpy등의 library들을 활용하여 데이터를 전처리한다.**

​		이때 데이터셋을 class로 구현하기 위해서는 이러한 전처리 과정들을 **함수화** 하여 class 내부의 method 함수		화 시킨다.

 2. **커스텀 데이터 셋**을 (class로 (optinal) ) 구성한다.

 3. **커스텀 딥러닝 혹은 머신러닝 모델**을 (class로 (optional) ) 구성한다.

 4. optimizer, cost(citer), epoch등 Hyper Parameter들을 설정한다.

 5. epoch수만큼 for문을 실행시키게 하여 optimizer를 통해 최적해를 찾는다.

    ```python
    optimizer = torch.optim."optimizer이름"(모델의 parameters,lr = learning_rate)
    optimizer.zero_grad() # optimizer는 parameter의 gradient를 누적하여 계산하므로 0으로 초기화 							시켜주는 과정이 필요하다.
    loss.backward()		# loss의 gradient를 계산하는 과정
    optimizer.step()	# gradient를 바탕으로 optimizer를 통해 최적화 시켜나간다.
    ```







































