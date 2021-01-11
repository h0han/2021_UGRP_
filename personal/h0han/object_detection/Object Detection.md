# Object Detection

classification + localization (물체 분류 + bounding box를 통해 위치 정보 탐색)

- 1-stage Detector : 두 문제를 동시에 행하는 방법
  - 비교적 빠르지만 낮은 정확도
  - YOLO 계열, SSD 계열
- 2-stage Detector : 두 문제를 순차적으로 행하는 방법
  - 비교적 느리지만 높은 정확도
  - R-CNN 계열