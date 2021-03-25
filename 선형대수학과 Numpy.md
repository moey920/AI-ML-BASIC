# 선형대수학

벡터 공간, 벡터, 선형 변환, 행렬, 연립 선형 방정식 등을 연구하는 대수학의 한 분야이다. 현대 선형대수학은 그중에서도 벡터 공간이 주 연구 대상이다. 추상대수학, 함수해석학에 널리 쓰이고 있다.

대수학(代數學, 영어: Algebra)은 일련의 공리들을 만족하는 수학적 구조들의 일반적인 성질을 연구하는 수학의 한 분야이다. 

## Scalar와 Vector

**Scalar** : 길이, 넓이, 질량, 온도처럼 **크기만 존재**하는 양, 숫자라고 생각하면 된다.   
**Vector** : 속도, 위치 이동, 힘, **크기와 방향이 모두 존재**하는 양

## Vector

- Vector Arithmetic(벡터 산술)
    - 산술은 2개 이상의 수를 결합하는 모든 법칙을 다룬다

```
x-> = (x1, x2) # 2차원 벡터   
y-> = (y1, y2)    
x-> + y-> = (x1+y1, x2+y2) # 2차원 벡터의 합     
kx-> = (kx1, kx2) # 벡터의 스칼라 곱   
```

- N-Dim Vectors

```
x-> = (x1, x2, x3) # 3차원 벡터   
y-> = (y1, y2, y3)    
x-> + y-> = (x1+y1, x2+y2, x3+y3) # 3차원 벡터의 합     
kx-> = (kx1, kx2, kx3) # 벡터의 스칼라 곱   
```

## 벡터 공간/내적

### Norm : 원점 O에서 점 (x1, x2, ..., xn)까지 이르는 거리 

> n차원 벡터 x-> = (x1, x2, ..., xn)에 대해   
Norm = ||x->|| = root(x1^2 + x2^2 + ... + xn^2)

- 피타고라스의 정리가 두 점까지의 거리를 두 점의 제곱의 합에 루트를 씌우듯이 n차원까지의 거리도 모두 동일한 식을 가진다.

### 내적 : Euclidean inner product, Dot product!

- 두 벡터가 **같은 차원일 때** 내적을 구할 수 있다.

```
x-> = (x1, x2, x3) # 3차원 벡터   
y-> = (y1, y2, y3)    
x->·y-> = (x1y1 + x2y2 + x3y3) # 3차원 벡터의 내적, 같은 차원끼리 곱해서 더한다, n차원까지도 똑같다.         
```

## 행렬(Matrix)

- 실수를 다음과 같이 직사각형 모양으로 배열한 것을 행렬이라고 한다.
- 열 (column)
- 행 (row)

### Matrix Arithmetic(행렬 산술)

- **같은 차원을 가진 행렬끼리만** 더하거나 뺄 수 있다. 

- 행렬끼리 곱할 때는 차원을 주의해야 한다.
    - A는 (3, 2), B는 (2, 3) 의 차원을 가진다.
    - A[1]과 B[0]의 차원이 같을 때 행렬 곱이 가능하며, 결과는 (A[0],B[1]) 차원이 된다. # (3,3)

- **전치행렬**은 원행렬의 **행과 열을 뒤바꾼 행렬**이다.
    - (1,3) -> (3,1)
    - A, AT(T를 지수로 붙여서 표현)

----------------------------------

# Numpy : Python에서 사용되는 과학 컴퓨팅용 라이브러리

- Scientific Computing : 과학적인 결과(데이터 분석, 통계, 변수간 관계 분석 등)을 내기 위한 라이브러리, Scipy, Scikit-learn 등도 유명하다.

- Python 언어에서 기본으로 지원하지 않는 행렬과 같은 데이터 구조 지원 및 수학/과학 계산 함수 포함

## 행렬이 왜 필요한가?

- 머신러닝에서 대부분의 데이터는 행렬로 표현됨

### 행렬 만들기

- np.array() 이용

```
import numpy as np
A = np.array([[1, 2],
            [3, 4]])
print(A)
[[1 2]
 [3 4]]
```

#### Numpy Array

- 이렇게 만들어진 행렬은 곱셈, 덧셈, 뺄셈이 가능
```
print(A * 3)
print(A + A)
print(A - A)
[[ 3 6]
 [ 9 12]]
[[2 4]
 [6 8]]
[[0 0]
 [0 0]]
```

#### Numpy Array - 산술 연산

- 행렬 내 원소에 대한 산술연산도 가능 : “element-wise operation”

```
print(A ** 2)
print(3 ** A)
print(A * A)
[[ 1 4]
 [ 9 16]]
[[ 3 9]
 [27 81]]
[[ 1 4]
 [ 9 16]]
```

#### 행렬 곱셈

- np.dot(x, y)는 x * y와 다릅니다. 어떻게? 내적과 단순 곱은 다르다.

```
x = np.array([[1, 2], [3, 4]])
y = np.array([[3, 4], [3, 2]])

print(np.dot(x, y))
print(x * y)
[[ 9 8]
 [21 20]]
[[3 8]
 [9 8]]
```

#### 비교 연산

- 비교연산을 통해 **array 내의 값**을 빠르게 비교 가능

```
a = np.array([1, 2, 3, 4])
b = np.array([4, 2, 2, 4])

print(a == b)
print(a > b)
[False, True, False, True]
[False, False, True, False]
```

#### 논리 연산

- logical_and, logical_or 함수를 이용하면 array 내의 element-wise 연산 수행 가능

```
a = np.array([1, 1, 0, 0], dtype=bool)
b = np.array([1, 0, 1, 0], dtype=bool)

np.logical_or(a, b)
np.logical_and(a, b)
[True, True, True, False]
[True, False, False, False]
```


#### Reductions : array를 하나의 스칼라 값으로 만들어주는 연산

- argmin/max: 최소/최대값의 **인덱스를 반환**

```
a = np.array([1, 2, 3, 4, 5])
np.sum(a) # 15
a.sum() # 15
a.min() # 1
a.max() # 5
a.argmin() # 0
a.argmax() # 4
```

#### Logical Reductions : array가 bool형일때 하나로 축약하는 연산

```
a = np.array([True, True, True])
b = np.array([True, True, False])
np.all(a) # 모든게 True인가 , True
np.all(b) # False
np.any(a) # 하나라도 True인가, True
np.any(b) # True
```

#### Statistical Reductions : 통계적 리덕션

`np.mean(x): 평균값`   
`np.median(x): 중간값`   
`np.std(x): 표준편차`   

```
x = np.array([1, 2, 3, 1])
print(np.mean(x))
print(np.median(x))
print(np.std(x))
1.75
1.5
0.82915619758884995
```

# Numpy 실습

## Numpy 행렬 만들기

- 함수 안에 배열 3×4 의 크기를 가진 행렬 A를 선언하세요.

```
import numpy as np

def main():
    print(matrix_tutorial())

def matrix_tutorial():
    # Create the matrix A here...
    A = np.array([[1,4,5,8],
                [2,1,7,3],
                [5,4,5,9]])
    return A

if __name__ == "__main__":
    main()
```

## Numpy 산술 연산자

```
import numpy as np

def main():
    print(matrix_tutorial())

def matrix_tutorial():
    A = np.array([[1,4,5,8], [2,1,7,3], [5,4,5,9]])

    # A 원소의 합이 1이 되도록 표준화(Normalization)를 적용하고 결괏값을 A에 다시 저장하세요. 예를 들어, [42, 58][42,58] 에 표준화를 적용하면 [0.42, 0.58][0.42,0.58] 이 됩니다.
    A = A / np.sum(A)
    
    # matrix_tutorial() 함수가 A의 분산(Variance)값을 리턴하도록 코드를 작성하세요. 어느정도 퍼져있는지 수치화한다.
    return np.var(A)

if __name__ == "__main__":
    main()
```

## Numpy 논리 연산자

```
import numpy as np

def main():
    A = get_matrix()
    print(matrix_tutorial(A))

def get_matrix():
    mat = []
    
    # 행렬 입력받기
    [n, m] = [int(x) for x in input().strip().split(" ")] 
    for i in range(n):
        row = [int(x) for x in input().strip().split(" ")]
        mat.append(row)
    return np.array(mat)

def matrix_tutorial(A):
    
    # A의 전치행렬(transpose) B를 생성하세요.
    # np.transpose() or A.T
    B = A.T
    print(B)
    
    # B의 역행렬을 구하여 C에 저장하세요. 역행렬을 구하는 것이 불가능하면 문자열 "not invertible"을 리턴합니다.
    # inv()는 행렬의 역행렬(inverse)를 구할 때 사용됩니다. NumPy의 선형대수학 관련 세부 패키지 linalg를 사용하기 때문에, 조금 더 긴 명령어를 사용합니다.
    # np.linalg.inv(), 그러나 역행렬을 구할 수 있는 행렬과 구할 수 없는 행렬리 있으므로 조심해야한다.
    try : 
        C = np.linalg.inv(B)
    except :
        return "not invertible"
    
    # matrix_tutorial() 함수의 리턴값으로 0보다 큰 C의 원소를 모두 세어 개수를 리턴하세요. sum()의 인자로 해당 조건 C > 0을 입력하면 0보다 큰 원소를 쉽게 셀 수 있습니다.
    return np.sum(C > 0)

if __name__ == "__main__":
    main()
```

## 벡터 연산으로 그림 그리기

이번 미션에서는 Numpy를 이용한 벡터 연산으로 원과 다이아몬드 그림을 그리고, 이것들을 조합해 smile 그림을 그려보겠습니다.

- 캔버스

그림을 그리기 위해서는 그림을 그릴 공간이 필요합니다. 이 프로젝트에서는 이 공간을 xrange, yrange 라는 변수로 지정하겠습니다.

만약,
```
xrange = [1, 3]
yrange = [2, 4]
```
라면, 그림을 그릴 캔버스는 (1, 2)(1,2), (3, 4)(3,4) 로 지정된 공간을 사용하게 됩니다.

- 그림 그리기

그림을 그리기 위해서, 다음 방식을 사용하겠습니다. 어떤 함수 f와 매우 작은 숫자 threshold에 대해,

- 캔버스 내에 점 P = (x, y)을 임의로 생성한다.
- f(P) < threshold 라면 점을 찍는다. 만약 그렇지 않다면, 점을 찍지 않는다.
- 이것을 100,000 회 반복한다.
    - 왜 f(P) == 0 일 때 점을 찍지 않고, 아주 작은 값 threshold 보다 작을 때 점을 찍는지, 한번 생각해 보세요!

- 원 그리기

(0, 0)(0,0) 이 중심이고, 반지름 1인 원의 방정식은 다음과 같습니다.

`x^2 + y^2 = 1`

원의 그림을 그리는 방식을 생각하면, 정확히 원 위에 있는 점들에 대해서 `circle(P) 는 0`을 가져야 합니다. 그러므로, circle(P) 는 다음과 같이 정의할 수 있습니다.

```
x = P[0]
y = P[1]
return sqrt(x ** 2 + y ** 2) - 1
```

위 코드의 리턴값은 수업시간에 배운 것과 같이 다음과 같습니다.

`return sqrt(np.sum(P * P)) - 1`

Norm의 개념을 사용한다면 다음과도 같습니다.

`return np.linalg.norm(P) - 1`

- 다이아몬드 그리기

(0, 0)(0,0) 이 중심이고, 원점에서 각 꼭지점까지 이르는 거리가 1인 다이아몬드의 방정식은 다음과 같습니다.

`|x| + |y| = 1`

원 그리기와 같은 방식으로 생각해본다면, right_eye(P)는 다음과 같이 정의할 수 있습니다.

`return np.abs(P[0]) + np.abs(P[1]) - 1`

```
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import elice_utils
import numpy as np
elice = elice_utils.EliceUtils()

def circle(P):
    return np.linalg.norm(P) - 1 # 밑의 코드와 동일하게 동작합니다.
    # return np.sqrt(np.sum(P * P)) - 1
    
def diamond(P):
    return np.abs(P[0]) + np.abs(P[1]) - 1
    
def smile(P):
    def left_eye(P):
        eye_pos = P - np.array([-0.5, 0.5])
        return np.sqrt(np.sum(eye_pos * eye_pos)) - 0.1
    
    def right_eye(P):
        eye_pos = P - np.array([0.5, 0.5])
        return np.sqrt(np.sum(eye_pos * eye_pos)) - 0.1
    
    def mouth(P):
        if P[1] < 0:
            return np.sqrt(np.sum(P * P)) - 0.7
        else:
            return 1
    
    return circle(P) * left_eye(P) * right_eye(P) * mouth(P)

def checker(P, shape, tolerance):
    return abs(shape(P)) < tolerance

def sample(num_points, xrange, yrange, shape, tolerance):
    accepted_points = []
    rejected_points = []
    
    for i in range(num_points):
        x = np.random.random() * (xrange[1] - xrange[0]) + xrange[0]
        y = np.random.random() * (yrange[1] - yrange[0]) + yrange[0]
        P = np.array([x, y])
        
        if (checker(P, shape, tolerance)):
            accepted_points.append(P)
        else:
            rejected_points.append(P)
    
    return np.array(accepted_points), np.array(rejected_points)

xrange = [-1.5, 1.5] # X축 범위입니다.
yrange = [-1.5, 1.5] # Y축 범위입니다.
accepted_points, rejected_points = sample(
    100000, #  점의 개수를 줄이거나 늘려서 실행해 보세요. 너무 많이 늘리면 시간이 오래 걸리는 것에 주의합니다.
    xrange, 
    yrange, 
    smile, # smile을 circle 이나 diamond 로 바꿔서 실행해 보세요.
    0.005) # Threshold를 0.01이나 0.0001 같은 다른 값으로 변경해 보세요.

plt.figure(figsize=(xrange[1] - xrange[0], yrange[1] - yrange[0]), 
           dpi=150) # 그림이 제대로 로드되지 않는다면 DPI를 줄여보세요.
           
plt.scatter(rejected_points[:, 0], rejected_points[:, 1], c='lightgray', s=0.1)
plt.scatter(accepted_points[:, 0], accepted_points[:, 1], c='black', s=1)

plt.savefig("graph.png")
elice.send_image("graph.png")
```