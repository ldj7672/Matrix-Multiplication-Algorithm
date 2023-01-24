# Matrix-Multiplication-Algorithm
Analysis of some matrix multiplication algorithm (C++)

1. Baseline 
2. Indexing 변경 (IJK -> KIJ)
3. SIMD (Single Instruction Multiple Data) 적용
4. SIMD + Indexing 변경
5. OpenCV library
6. Eigen library


## # 실험 결과

모든 실험은 (1000x1000 size matrix) * (1000x1000 size matrix) 의 행렬곱 연산의 평균 소요 시간(ms)입니다.
(CUDA 방법의 경우 메모리에 올리는 시간까지 포함)

![image](https://user-images.githubusercontent.com/96943196/214231173-3f3bff2a-5582-4607-bdee-35926bb983ca.png)
