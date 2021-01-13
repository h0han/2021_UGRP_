# Object Detection

영상 상에서 특정 개체를 인식하고 나머지 다른 개체는 속성을 지닌 다른 개체로 인식

[1.퓨샷 러닝(few-shot learning) 연구 동향을 소개합니다.](https://www.kakaobrain.com/blog/106)

[Learning to Compare: Relation Network for Few-Shot Learning](https://arxiv.org/abs/1711.06025)

> 한 태스크에서 {고양이, 자동차, 사과}처럼 서로 완전히 다른 성격의 범주를 분류하는 문제라면 물체의 모양(shape) 정보만으로도 쿼리 데이터의 범주를 쉽게 예측할 수 있을 것입니다. 하지만 특징 추출기가 같은 범주의 데이터를 더 가깝게, 다른 범주의 데이터를 더 멀게 할 정도로 충분히 복잡하지 않다면 어떨까요? 그렇다면 {러시안블루, 페르시안, 먼치킨}처럼 고양이의 종류를 구분하는 태스크를 풀기 어려울 것입니다.
>
> 이 논문에서는 특징 추출기에 CNN을 적용했을 뿐만 아니라 거리 계산 함수에도 다층 퍼셉트론[[5\]](https://www.kakaobrain.com/blog/106#ref_list_5)을 적용시켰습니다. 다층 퍼셉트론은 같은 범주 또는 다른 범주의 서포트 데이터와 쿼리 데이터를 분류하는 법을 배웁니다.

[One Shot Learning](https://www.edwith.org/deeplearningai4/lecture/34911/)



classification + localization (물체 분류 + bounding box를 통해 위치 정보 탐색)

- 1-stage Detector : 두 문제를 동시에 행하는 방법
  - 비교적 빠르지만 낮은 정확도
  - YOLO 계열, SSD 계열
- 2-stage Detector : 두 문제를 순차적으로 행하는 방법
  - 비교적 느리지만 높은 정확도
  - R-CNN 계열



1. detection (bounding box별 인식)

   영상 내 object 중 의도한 특정 object cluster만 선별적으로 detect --> 목적에 따라 기법은 다를 수 있으나 높은 정확도를 추구 : R-CNN

2. few shot + reidentification

   선별된 object cluster 중 입력된 특정 object만 인식되도록 구성

