# 비지도학습 개론(Introduction to Unsupervised Learning)

> 분류 : Machine Learning - 비지도학습 - 차원축소, 클러스터링

- 지도학슫
    - 회귀분석(Regression) : 스칼라 값을 예측
    - 분류(Classification) : 범주를 예측

- 지도학습 vs 비지도학습
    - 지도학습 : 얻고자 하는 답으로 구성된 데이터
    - 비지도학습 : 답이 정해져 있지 않은 데이터에서 숨겨진 구조를 파악

- 비지도학습
    - input 데이터에 레이블링이 되어있지않다.
    - 차원축소(Dimension Reduction)
    - 클러스터링(Clustering)

==========

# Hard clustering vs Soft clustering

## 숨겨진 구조(hidden structure)?
- 데이터 포인트들은 비슷한 것들끼리 뭉쳐있다 = Hard clustering
    - 이 데이터는 100% 고양이이다. 0% 강아지이다. 선을 딱 긋고 싶을 때 Hard clustering 사용
    - 대표적인 알고리즘의 종류
        - Hierarchical Clustering
        - K-Means
        - BDSCAN
        - OPTICS
- 한 개의 데이터 포인트는 숨겨진 클러스터들의 결합이다 = Soft clustering
    - 책을 예시로 들면 '사이언스 픽션' 장르의 경우 60% 과학과 35% 판타지와 5% 역사가 합쳐진 장르이다.
    - 대표적인 알고리즘의 종류
        - Topic Models
        - FCM
        - Gaussian Mixture Modles(EM, GMM)
        - Soft K-Means
- 보통 자연의 문제는 Soft Clustering에 가깝다. 다만 SC를 적용하지 못하면 Hard Clustering을 쓰곤 한다.

============

# K-means clustering

## Finding the K : 군집의 개수 k
1. 눈으로 확인
2. 모델이 데이터를 얼마나 잘 설명하는가   
    - Elbow method
    - Silhouette method
    - Gap statistic method
3. 고려할 것들   
    - 데이터의 특성
        - 어떻게 만들어진 데이터인가?
        - 데이터 포인트 외 다른 feature
    - 분석 결과로 얻고자 하는 것
        - 고양이 vs 개 분류(K=2)
        - 사람들의 행동 분석('풀고자하는 문제에 대해 분석 결과 k가 n이다'와 같은 사전지식)
        - 가격 대비 효율성 분석

===========

# K를 정하기 어려울 때 도움이 되는 것이 '차원축소' 알고리즘이다.

## 와인 분석하기

당신은 엘리스 와이너리의 데이터 사이언티스트다.   
지금까지 엘리스 와이너리에서는 수백 가지의 와인을 생산했고,   
생산팀은 그 **와인들에 대한 13가지 특성**을 측정해서 정리했다.   

```
1) Alcohol
2) Malic acid
3) Ash
4) Alcalinity of ash
5) Magnesium
6) Total phenols
7) Flavanoids
8) Nonflavanoid phenols
9) Proanthocyanins
10) Color intensity
11) Hue
12) OD280/OD315 of diluted wines
13) Proline
```

- 문제 : 지금까지 생산한 와인들을 종류별로 모아서 라인업으로 만들고 싶다. 178개의 와인들을 몇 가지로 분류할 수 있겠는가?
    - 13개의 Feature를 13차원으로 나타내서 파악하긴 힘들다.
    - 13차원을 2~3차원으로 환원해서 줄이면 파악하기 용이해진다.
- 목표 : “이렇게 세 개의 브랜드를 런칭하면 되겠군!”

=============

# 주성분 분석, PCA 알고리즘(Principal Component Analysis) : 왜 사용하는가?

1. 고차원의 데이터를 저차원으로 줄이기 위해 (예: 시각화)
```
1,14.23,1.71,2.43,15.6,127,2.8,3.06,.28,2.29,5.64,1.04,3.92,1065
1,13.2,1.78,2.14,11.2,100,2.65,2.76,.26,1.28,4.38,1.05,3.4,1050
1,13.16,2.36,2.67,18.6,101,2.8,3.24,.3,2.81,5.68,1.03,3.17,1185
1,14.37,1.95,2.5,16.8,113,3.85,3.49,.24,2.18,7.8,.86,3.45,1480
1,13.24,2.59,2.87,21,118,2.8,2.69,.39,1.82,4.32,1.04,2.93,735
1,14.2,1.76,2.45,15.2,112,3.27,3.39,.34,1.97,6.75,1.05,2.85,1450
1,14.39,1.87,2.45,14.6,96,2.5,2.52,.3,1.98,5.25,1.02,3.58,1290
1,14.06,2.15,2.61,17.6,121,2.6,2.51,.31,1.25,5.05,1.06,3.58,1295
1,14.83,1.64,2.17,14,97,2.8,2.98,.29,1.98,5.2,1.08,2.85,1045
1,13.86,1.35,2.27,16,98,2.98,3.15,.22,1.85,7.22,1.01,3.55,1045
1,14.1,2.16,2.3,18,105,2.95,3.32,.22,2.38,5.75,1.25,3.17,1510
1,14.12,1.48,2.32,16.8,95,2.2,2.43,.26,1.57,5,1.17,2.82,1280
1,13.75,1.73,2.41,16,89,2.6,2.76,.29,1.81,5.6,1.15,2.9,1320
```
- 이러한 데이터를 눈으로 보기위해 2차원으로 축소한다. (세 개의 군집으로 표현)
- 와인예시의 경우 13차원을 2차원으로 축소하기 때문에 필연적으로 데이터의 손실이 발생한다. 데이터의 손실을 최대한 방지하는게 PCA의 목표이다.

2. **데이터 정제**
- 데이터를 관찰할 땐 필연적으로 오류가 발생한다.(노이즈)
- 평면에서 위 아래로 데이터가 퍼져 3차원이 되었다고 했을 때, 이를 2차원 평면으로 줄임으로써 차원을 줄이고 데이터를 정제할 수 있다.
- K-means는 기계적으로 데이터간의 거리를 측정해 군집화할 뿐이기 때문에 차원이 크게 중요하지는 않다. 데이터의 정제가 더 중요하다. 
- 차원이 많으면 자유도가 높고, 자유도가 높으면 표현할 수 있는 정보가 너무 많기 때문에 군집화하기 힘들다. 이럴 때 PCA를 이용하는 데이터 사이언스의 인사이트가 필요하다.

3. 수학적으로 PCA를 해석하기는 매우 어렵다. 일단 실습에 집중하자
- sklearn.decomposition.PCA
- PCA를 사용하기 위하여 scikit-learn에 구현된 함수를 호출합니다. PCA() 함수를 이용해 쉽게 PCA를 적용할 수 있습니다.
- 다음은 13차원의 데이터 X를 2차원으로 줄이는 코드입니다. 이때 X의 shape은 (n, 13) 형식이 되어야 합니다.   
```
pca = sklearn.decomposition.PCA(n_components=2)
pca.fit(X)
pca_array = pca.transform(X)
```

## 실습 : PCA

```
# 와인 데이터를 입력받아 PCA를 통해 2차원으로 줄인 뒤, 이것을 matplotlib을 이용해 출력하는 연습해보겠습니다.
# sklearn.decomposition.PCA
import sklearn.decomposition
import matplotlib.pyplot as plt
import numpy as np
import elice_utils
import csv

def main():
    X, attributes = input_data()
    pca_array = normalize(X)
    pca, pca_array = run_PCA(X, 2)
    visualize_2d_wine(pca_array)

def input_data():
    X = []
    
    attributes = []
    with open("data/attributes.txt") as fp :
        attributes = fp.readlines()
    attributes = [x.strip() for x in attributes]
    
    csvreader = csv.reader(open("data/wine.csv"))
    for line in csvreader :
        float_numbers = [float(x) for x in line]
        X.append(float_numbers)

    return np.array(X), attributes

def run_PCA(X, num_components): # n_components:주성분개수를 몇 차원으로 줄일건지
    pca = sklearn.decomposition.PCA(n_components = num_components)
    pca.fit(X)
    pca_array = pca.transform(X)
    
    return pca, pca_array
    
def normalize(X):
    '''
    각각의 feature에 대해,
    178개의 데이터에 나타나는 해당하는 feature의 값이 최소 0, 최대 1이 되도록
    선형적으로 데이터를 이동시킵니다.
    '''
    for i in range(X.shape[1]) : # 13개의 차원에 대해 각각의 feature 하나씩 처리
        # i번째 column에 대한 모든 row
        X[:,i] = X[:,i] - np.min(X[:,i])
        X[:,i] = X[:,i] / np.max(X[:,i])
    return X

def visualize_2d_wine(X):
    '''X를 시각화하는 코드를 구현합니다.'''
    plt.scatter(X[:,0], X[:,1])
    
    plt.savefig("image.png")
    elice_utils.send_image("image.png")

if __name__ == '__main__':
    main()
```

========

# K-means 클러스터링

> 클러스터링 : 주어진 데이터를 비슷한 그룹 (클러스터) 으로 묶는 알고리즘

- K-means는 K를 사용자가 입력해야한다.
- K-means는 서로 뭉친 데이터를 어떻게 군집화할지 정해준다.

엘리스 와이너리는 세 개의 와인 브랜드를 런칭하기로 했다.(PCA를 통해 13->2차원으로 줄였을 때 3개(K)로 군집화하기로 결정)      
당신은 어떤 와인들을 각각의 브랜드에 할당할지 판단해야 한다.   

## K-means : 반복을 이용한 클러스터링 알고리즘

1. **중심** Centroid : 각 클러스터의 "중심"을 의미
    - 해당 군집의 각 데이터 포인터들의 x좌표의 평균값, y좌표의 평균값을 구해 중심값을 구한다.

2. **중심과의 거리** distance : 중심과 데이터 포인트와의 거리
    - 일반적으로 Norm으로 결정

- Step 0

K-means는 **중심(Centroid)**의 위치에 의해 클러스터링을 진행   
알고리즘을 시작할 때, 초기 중심값은 데이터 중 **임의로 선정**   

- Step 1

    - 중심값이 정해지면, 각각의 데이터 포인트에 대해 다음을 계산 :   
    "내게서 가장 가까운 중심점은 어디인가?"   
    모든 데이터 포인트에 대해서 임의로 선정한 중심값 A, B에 대하여 Norm으로 거리를 구해 가까운 중심점을 찾아낸다.

    - 주황색 점에서 가장 가까운 중심점은 중심 B   
    이 데이터 포인트를 클러스터 B에 할당   
    
    - 나머지 모든 데이터 포인트에 대해 동일한 작업을 통해 각각 클러스터를 할당한다.

- Step 2

정해진 클러스터에서 중심점을 다시 계산   
중심점은 해당 클러스터 내 데이터 포인터 위치의 무게중심값(또는 평균)    

- 다시 Step 1

다시 각각의 데이터 포인트에 대해 다음을 계산 :    
"내게서 가장 가까운 중심점은 어디인가?"

모든 데이터 포인트에 대해   
거리 계산 후 가장 가까운 클러스터 할당   

- 다시 Step 2

다시, 중심점 업데이트   

- Step 1

모든 데이터 포인트에 대해 가장 가까운 중심의 클러스터로 할당

- Step 1

**어떠한 데이터 포인트의 할당도 변하지 않으면, 알고리즘 종료**

## k-means 구현

2차원으로 줄여 시각화한 와인 데이터를 이용해 어떤 와인이 비슷한지 알아내고, 비슷한 와인을 묶는 알고리즘을 작성해보겠습니다.

클러스터링, 또는 클러스터 분석은 주어진 개체에서 비슷한 개체를 선별하고 묶는(grouping) 작업을 의미합니다. 또한, 대푯값을 추출하여 각 클러스터를 대표하는 개체를 찾아낼 수 있습니다.

예로, 여러 종의 생물이 나열되어 있을 때, 각 종에 대한 세부 정보를 이용해 동물 클러스터와 식물 클러스터를 자동으로 추출해 낼 수 있습니다.

K-Means 클러스터링은 주어진 데이터를 K개의 클러스터로 묶는 알고리즘입니다. 알고리즘은 어떠한 선행 학습 없이 자동으로 클러스터를 분류하고 개체들의 묶음을 추출해 냅니다.

K의 개수를 조정하면 클러스터의 일반도를 조정할 수 있습니다. 생물 데이터에서, K=2 인 경우에는 동물과 식물 클러스터가 나올 가능성이 높습니다.

K가 매우 크다면, 동물/식물 내의 세부 분류, 강/목/속/종 등의 분류가 나올 수 있습니다. K-means는 완전한 자율 알고리즘이기 때문에 K를 제외한 다른 입력값이 없으며, random 값을 사용하므로 여러 번을 반복 실행하여 평균 성능을 측정하는 것이 일반적입니다.

1. 주성분 분석 결과를 확인합니다. PCA 차원 축소 실습에서 구현한 코드를 사용합니다.
2. K-Means 알고리즘을 구현합니다. K-means 알고리즘은 다음 값을 입력받습니다.
- num_clusters - 클러스터의 개수.
- initial_centroid_indices - 초기 중심점. initial_centroid_indices가 [0, 1, 2]일때 initial_centroids 는 첫 번째, 두 번째, 그리고 세 번째 데이터 포인트입니다.
3. K-Means를 실행한 후 각 데이터 포인트에 대한 클러스터 결과(label)를 리턴합니다. label은 0부터 시작합니다.

```
import sklearn.decomposition
import sklearn.cluster
import matplotlib.pyplot as plt
import numpy as np
import elice_utils

def main():
    X, attributes = input_data()
    X = normalize(X)
    pca, pca_array = run_PCA(X, 2) # 13차원 -> 2차원
    labels = kmeans(pca_array, 3, [0, 1, 2]) # (178,2) array를받아 3개의 클러스터를 찾고 중심점의 시작점이 [0, 1, 2]번째 좌표이다.
    # labels = kmeans(pca_array, 5, [0, 1, 2, 3, 4])
    visualize_2d_wine(pca_array, labels) # 색깔을 입힌 2d scatter 그래프 그리기

def input_data():
    X = []
    attributes = []
    
    with open('data/wine.csv') as fp:
        for line in fp:
            X.append([float(x) for x in line.strip().split(',')])
    
    with open('data/attributes.txt') as fp:
        attributes = [x.strip() for x in fp.readlines()]

    return np.array(X), attributes

def run_PCA(X, num_components):
    pca = sklearn.decomposition.PCA(n_components=num_components)
    pca.fit(X)
    pca_array = pca.transform(X)

    return pca, pca_array

def kmeans(X, num_clusters, initial_centroid_indices):
    '''
    X : (178,2)
    num_clusters : 클러스터의 개수(k)
    initial_centroid_indices : 시작할 중싱점들의 인덱스
    '''
    import time
    N = len(X)
    centroids = X[initial_centroid_indices] # 중심점들의 집합은 X[0], X[1], X[2]로 이루어진 array
    labels = np.zeros(N) # 각각의 데이터포인터들이 클러스터 0,1,2이다 정의. 초기값으로 모든 데이터 포인트들의 클러스터가 0으로 초기화.
    
    while True:
        '''
        Step 1. 각 데이터 포인트 i 에 대해 가장 가까운
        중심점을 찾고, 그 중심점에 해당하는 클러스터를 할당하여
        labels[i]에 넣습니다.
        가까운 중심점을 찾을 때는, 유클리드 거리를 사용합니다.
        미리 정의된 distance 함수를 사용합니다.
        '''
        is_changed = False
        for i in range(N) : # 178개 반복
            distances = []
            for k in range(num_clusters) : # 0,1,2번째 클러스터 확인
                # X[i]와 centriods[k]의 거리를 구하고 최소값을 lables[i]에 넣기
                k_dist = distance(X[i], centroids[k])
                distances.append(k_dist) # 각 데이터포인트들에서 각 중심점까지의 거리가 리스트에 들어간다.
            if labels[i] != np.argmin(distances) :
                is_changed = True
            # 각 데이터 포인트에서 가장 가까운 중심점을 찾아내기
            labels[i] = np.argmin(distances) # 클러스터가 할당된다.

        '''
        Step 2. 할당된 클러스터를 기반으로 새로운 중심점을 계산합니다.
        중심점은 클러스터 내 데이터 포인트들의 위치의 *산술 평균*
        으로 합니다.
        '''
        for k in range(num_clusters) :
            x = X[labels == k][:,0]
            y = X[labels == k][:,1]
            
            x = np.mean(x)
            y = np.mean(y)
            centroids[k] = [x,y] # 새로운 중심값 설정
        
        '''
        Step 3. 만약 클러스터의 할당이 바뀌지 않았다면 알고리즘을 끝냅니다.
        아니라면 다시 반복합니다.
        '''
        if not is_changed :
            break
            
    return labels

def distance(x1, x2): # Norm 함수
    return np.sqrt(np.sum((x1 - x2) ** 2)) # 두 거리의 차이를 제곱한 합에 루트를 씌운다.
    
def normalize(X):
    for dim in range(len(X[0])):
        X[:, dim] -= np.min(X[:, dim])
        X[:, dim] /= np.max(X[:, dim])
    return X

'''
이전에 더해, 각각의 데이터 포인트에 색을 입히는 과정도 진행합니다.
'''
def visualize_2d_wine(X, labels):
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:,0], X[:,1], c=labels) # labels의 값을 참고해서 color를 자동으로 결정
    plt.savefig("image.svg", format="svg")
    elice_utils.send_image("image.svg")

if __name__ == '__main__':
    main()
```
