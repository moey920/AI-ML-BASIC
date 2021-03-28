# 인기 있는 TED 강연 분석하기

TED 강연의 관련 통계 자료를 분석하고, 인기 있는 강연은 어떤 요소들이 있는지 찾아내 보겠습니다.

- 데이터 형식   
강의 데이터가 담긴 ted.csv 파일은 다음과 같은 내용으로 구성되어 있습니다. 1열부터 순서대로입니다.

- 댓글 개수
- 강연에 대한 부가 설명
- 강연 길이 (초 단위)
- 행사명 (예: TED2009)
- 녹화 일자
- 번역된 언어 수
- 연사 이름
- 연사 이름과 강연 제목
- 연사 수
- 강연 공개 일자
- 강연 평가 (JSON 형식, CSV 파일을 직접 참고하세요)
- 연관된 강연
- 연사 직업
- 태그 (관련 키워드)
- 강연 제목
- 강연 동영상 URL 주소
- 조회수

## 지시사항

get_popular_speaking 함수에 대한 설명입니다.

- 매개변수 n : 양의 정수 n이 주어집니다.
- 반환값 : ted.csv 데이터에 주어진 강연 중, 조회수가 n회 이상의 강연들만 저장된 리스트를 조회수에 대한 내림차순으로 반환합니다.
- 반환값 형식 : 반환되는 리스트의 각 요소는 두 개의 자료 s와 n을 가진 튜플 (s, n)이어야 합니다. s는 문자열 자료형이며 강연의 제목을, n은 정수 자료형이며 강연의 조회수를 나타냅니다.

예를 들어 n이 40000000이라면, 아래처럼 조회수가 40000000 이상인 강연들만 리스트에 담겨 내림차순으로 반환되어야 합니다.

`[('Do schools kill creativity?', 47227110), ('Your body language may shape who you are', 43155405)]`

### 내가 푼 풀이

```
import csv
import json
import pandas as pd

def get_popular_speaking(n) :
    talks = []
    ted = pd.read_csv('ted.csv', names=['댓글개수', '강연에 대한 부가 설명', '강연길이', '행사명', '녹화 일자', '번역된 언어 수', '연사 이름', '연사 이름과 강연 제목', '연사 수', '강연 공개 일자', '강연 평가', '연관된 강연', '연사 직업', '태그', '강연 제목', '강연 동영상 URL 주소', '조회수'], header=None)

    hot_ted = ted[ted['조회수']>=n]

    df = pd.concat([hot_ted['강연 제목'], hot_ted['조회수'].astype('int64')], axis=1)
    
    sorted_df = df.sort_values('조회수', ascending=False).reset_index()

    for i in range(len(sorted_df)) :
        s, n = sorted_df.loc[i][1], sorted_df.loc[i][2]
        talks.append((s,n))

    return talks

def main():
    n = int(input())
    print(get_popular_speaking(n))


if __name__ == "__main__":
    main()
```

===========

# 나이브 베이즈 분류 (Naive Bayes Classifier)

나이브 베이즈 분류는 데이터의 확률적 속성을 가지고 클래스를 판단하는, 꽤 높은 성능을 가지는 머신러닝 알고리즘입니다.

이를 이해하기 위해서 Bayes’ Theorem에 친숙해 질 필요가 있습니다.

> Bayes’ theorem : P(Y|X) = P(X|Y) * P(Y)/P(X)

X는 관찰 값, Y는 결과 값을 표현합니다. data set 내에서 X와 Y의 빈도수를 활용하여 연산에 필요한 각각의 확률 값을 계산 할 수 있습니다.

엘리스의 이메일을 한 번 들여다 보면서 Bayes’ theorem을 이해해 볼까요? 앨리스의 이메일은 다음과 같은 텍스트 목록을 가지고 있습니다.   
```
타입	    텍스트
Spam	    “(광고) XXX 지금 확인 해보세요.” 첨부파일 : exe
Ham	        “[긴급]앨리스님, 확인 부탁드립니다.” 첨부파일 : exe
Ham	        “Git 오프라인 수업을 3일 간 합니다”
Spam	    “제목없음” 첨부파일 : exe
Spam	    “놓칠 수 없는 기회, 확인 해보세요.”
```

스팸 메일과 정상 메일에서 공통적으로 나타나는 키워드인 “확인“이 등장했을 때, 이 메일이 스팸 메일 인지, 정상 메일이 되는 지에 대해 판별해 보도록 하겠습니다.   
```
>>> P( "스팸 메일" | "확인" ) = ?
>>> P( "정상 메일" | "확인" ) = ?
```

## 지시사항

앨리스의 메일함에는 총 20개의 메일이 들어있습니다. 그중 스팸 메일은 8개, 정상 메일은 12개로 분류되어 있습니다. “확인” 키워드를 가지는 메일이 7개, “확인”을 제외한 메일이 13개라고 할 때, 다음과 같은 분포를 가집니다.

```
스팸 메일	정상    메일	개수
“확인”	    5	    2	    7
나머지	    3	    10	    13
개수	    8	    12	    20
```

### 내가 푼 풀이

```
import numpy as np


# 리스트 안에 값들을 정규화 합니다.
def normalization(x):
    return [element / sum(x) for element in x]


# 1. P(“스팸 메일”) 의 확률을 구하세요.
p_spam = 8/20
print(p_spam)

# 2. P(“확인” | “스팸 메일”) 의 확률을 구하세요.
p_confirm_spam = 5/8

# 3. P(“정상 메일”) 의 확률을 구하세요.
p_ham = 12/20

# 4. P(“확인” | "정상 메일" ) 의 확률을 구하세요.
p_confirm_ham = 2/12

# 5. P( "스팸 메일" | "확인" ) 의 확률을 구하세요.
p_spam_confirm = 5/7

# 6. P( "정상 메일" | "확인" ) 의 확률을 구하세요.
p_ham_confirm = 2/7

print("P(spam|confirm) = ",p_spam_confirm, "\nP(ham|confirm) = ",p_ham_confirm, "\n")

# 두 값을 비교하여 확인 키워드가 스팸에 가까운지 정상 메일에 가까운지 확인합니다.
value = [p_spam_confirm, p_ham_confirm]
result = normalization(value)

print("P(spam|confirm) normalization = ",result[0], "\nP(ham|confirm) normalization = ",result[1], "\n")

if p_spam_confirm > p_ham_confirm:
    print( round(result[0] * 100, 2), "% 의 확률로 스팸 메일에 가깝습니다.")
else :
    print( round(result[1] * 100, 2), "% 의 확률로 일반 메일에 가깝습니다.")
```

===============

# 선형 회귀 직접 구현하기

선형 회귀는 종속 변수 y와 한 개 이상의 독립 변수 X와의 선형 상관 관계를 모델링하는 회귀분석 기법을 말합니다.

이번 시간에는 y와 x가 주어졌을 때, ‘y = ax+b’ 라는 형태의 직선을 회귀식으로 하는 단순한 선형 회귀(Linear Regression)를 직접 구현해보도록 합시다.

- 선형 회귀의 절차

1. x라는 값이 입력되면 ‘ax+b’라는 계산식을 통해 값을 산출하는 예측 함수를 정의합니다.

2. 실제 값 y와 예측 함수를 통해 계산한 예측값 간의 차이를 계산합니다.

3. a와 b를 업데이트하는 규칙을 정의하고 이를 바탕으로 a와 b의 값을 조정합니다.

4. 위의 과정을 특정 반복 횟수만큼 반복합니다.

5. 반복적으로 수정된 a와 b를 바탕으로 ‘y=ax+b’라는 회귀식을 정의합니다.

## 지시사항

1. Numpy 배열 a, b, x 를 받아서 ‘ax+b’를 계산하는 prediction 함수를 정의합니다.(Numpy 활용)
2. 실제값(y)과 예측값의 차이를 계산하여 error를 정의합니다.
3. 본문 내에서 정의한 함수를 이용하여 a와 b값의 변화값을 저장합니다.

### 내가 푼 풀이

```
import numpy as np
import elice_utils
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use("Agg")
eu = elice_utils.EliceUtils()
learning_rate = 1e-4
iteration = 10000

x = np.array([[8.70153760], [3.90825773], [1.89362433], [3.28730045], [7.39333004], [2.98984649], [2.25757240], [9.84450732], [9.94589513], [5.48321616]])
y = np.array([[5.64413093], [3.75876583], [3.87233310], [4.40990425], [6.43845020], [4.02827829], [2.26105955], [7.15768995], [6.29097441], [5.19692852]])

##입력값(x)과 변수(a,b)를 바탕으로 예측값을 출력하는 함수를 만들어 봅니다.
def prediction(a,b,x):
    #1.Numpy 배열 a,b,x를 받아서 'x*(transposed)a + b'를 계산하는 식을 만듭니다.
    equation = x * a.T + b
    
    return equation

##변수(a,b)의 값을 어느정도 업데이트할 지를 정해주는 함수를 만들어 봅니다.
def update_ab(a,b,x,error,lr):
    ## a를 업데이트하는 규칙을 만듭니다..
    delta_a = -(lr*(2/len(error))*(np.dot(x.T, error)))
    ## b를 업데이트하는 규칙을 만듭니다.
    delta_b = -(lr*(2/len(error))*np.sum(error))
    
    return delta_a, delta_b

# 반복횟수만큼 오차(error)를 계산하고 a,b의 값을 변경하는 함수를 만들어 봅니다.
def gradient_descent(x, y, iters):
    ## 초기값 a= 0, b=0
    a = np.zeros((1,1))
    b = np.zeros((1,1))    
    
    for i in range(iters):
        #2.실제 값 y와 prediction 함수를 통해 예측한 예측값의 차이를 error로 정의합니다.
        error = y - prediction(a,b,x)
        #3.위에서 정의한 함수를 이용하여 a와 b 값의 변화값을 저장합니다.
        a_delta, b_delta = update_ab(a,b,x,error,lr=learning_rate)
        ##a와 b의 값을 변화시킵니다.
        a -= a_delta
        b -= b_delta
    
    return a, b

##그래프를 시각화하는 함수입니다.
def plotting_graph(x,y,a,b):
    y_pred=a[0,0]*x+b
    plt.scatter(x, y)
    plt.plot(x, y_pred)
    plt.savefig("test.png")
    eu.send_image("test.png")

##실제 진행 절차를 확인할 수 있는 main함수 입니다.
def main():
    a, b = gradient_descent(x, y, iters=iteration)
    print("a:",a, "b:",b)
    plotting_graph(x,y,a,b)
    return a, b

main()
```

=============

# 넷플릭스 시청 데이터 분석하기

netflix.json 파일에는 유저별 시청한 영화 정보가 담겨 있습니다. 데이터의 key는 영화 코드이고 value는 해당 영화를 감상한 유저 코드 리스트가 주어집니다.

movies.py에 titles 딕셔너리는 key가 영화 코드를 정수 자료형으로, value는 해당 영화의 제목을 문자열 자료형으로 담고 있습니다.

`titles = {1: 'Dinosaur Planet', ... }`

## 지시사항

get_top_movies 함수에 대한 설명입니다.

- 매개변수 n : 양의 정수 n이 주어집니다.
- 반환값 : netflix.json 데이터에 주어진 영화 중, 시청한 유저의 수가 n회 이상의 영화들만 저장된 리스트를 유저의 수에 대한 내림차순으로 반환합니다.
- 반환값 형식 : 반환되는 리스트의 각 요소는 두 개의 자료 s와 n을 가진 튜플 (s, n)이어야 합니다. s는 문자열 자료형이며 영화 제목을, n은 정수 자료형이며 영화를 시청한 유저 수를 나타냅니다.

예를 들어 n이 23500이라면, 아래처럼 시청 유저 수가 23500 이상인 영화들만 리스트에 담겨 내림차순으로 반환되어야 합니다.

`[('Pirates of the Caribbean: The Curse of the Black Pearl', 24786), ('Forrest Gump', 24557), ('The Sixth Sense', 24284), ('The Matrix', 23956), ("Ocean's Eleven", 23891), ('Independence Day', 23879), ('Spider-Man', 23649)]`

### 내가 푼 풀이

```
import json
from movies import titles

def get_top_movies(n) : # 매개변수 n : 양의 정수 n이 주어집니다.
    # netflix.json 데이터를 이용하여 주어진 문제를 해결하세요.
    '''
    반환값 : netflix.json 데이터에 주어진 영화 중, 시청한 유저의 수가 n회 이상의 영화들만 저장된 리스트를 유저의 수에 대한 내림차순으로 반환합니다.
    '''
    movies = []
    with open('./netflix.json') as file:
        json_string = file.read() 
        dic = json.loads(json_string) # json파일을 dictionary로 변환해서 리턴한다.
        for i in dic :
            if len(dic[i]) >= n :
                # print(titles[int(i)])
                s = titles[int(i)]
                p = len(dic[i])
                movies.append((s,p))
                
    movies.sort(key = lambda x:x[1], reverse=True) # 튜플의 두 번째 원소를 기준으로 내림차순으로 정렬하기.
    
    return movies


def main():
    n = int(input())
    print(get_top_movies(n))

if __name__ == "__main__":
    main()
```