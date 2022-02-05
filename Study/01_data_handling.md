## 01. 데이터 처리

Python에서 사용하는 데이터형에 대해 먼저 알아보자.  
기계학습에서는 대부분, Data Frame을 활용하기때문에 `Pandas`를 주로 사용하게 되므로, 그렇게 큰 비중이 있는 내용은 아니지만,알아두는 것이 Python 공부에도 더욱 도움이 된다.

Python에서 활용하는 collection literal은 총 4가지이다.  
  
|종류|형식|데이터수정,삭제|index 이용|
|---|---|---|---|
|리스트(list)|`num = [1, 2, 3]`| Mutable | O |
|튜플(tuple)|`num = (1, 2, 3)`| Immutable | O |
|집합(set)|`num = {1, 2, 3}`| Mutable | X |
|딕셔너리(dictionary)|`num = {'key1' : 1, 'key2' : 2}`| Mutable | X |  
  
실제로 data를 분석할 때에는 `tuple`을 가장 많이 활용하니, tuple의 특성을 잘 기억해두자.  



### Numpy

Python의 list를 그대로 활용하여서 내부의 값을 계산하기에는 어려움이 많다.  
이를 행렬과 같이 계산할 수 있도록 도와주는 package가 numpy이다.  

해당 Part에서 Numpy의 모든 것을 다루는 것이 아니라 배열의 기본 기능에 대해서만 이야기해보자.


```python
num1 = [1, 2, 3, 4]
num2 = [5, 6, 7, 8]
num = [num1, num2]
# num은 2차원 list가 됨
num
```




    [[1, 2, 3, 4], [5, 6, 7, 8]]




```python
# 계산 불가
# 하나의 리스트로 합쳐짐
num1 + num2
```




    [1, 2, 3, 4, 5, 6, 7, 8]



`numpy`를 사용하기 위해, 먼저 import문을 작성하자.


```python
import numpy as np
```

`numpy`의 배열(==행열)을 활용해 연산하는 방법은 다음과 같다.


```python
num = np.array(num)
num1 = np.array(num1)
num2 = np.array(num2)
```


```python
num1.sum()
```




    10




```python
num1.std()
```




    1.118033988749895



#### 배열 추출

배열의 인덱싱은 **인덱스연산자**인 `[,]`를 활용하여 가능하다.
slicing또한 `[:행, :열]`을 활용하여 편하게 추출할 수 있다.


```python
# 배열 인덱싱
num[1, 3]

```




    8




```python
# 배열 슬라이싱
num3 = num[:2, :2]
num3
```




    array([[1, 2],
           [5, 6]])



#### 배열 메소드

`numpy`의 기본적인 메소드로는  
`sum()` : 열 또는 행 기준 합  
`add()` : 행렬 합  
`dot()` : 행렬 곱  
이 존재하나, 자주 사용하지는 않으니 참고만 하자.



```python
# 열 기준으로 합계
num.sum(axis=0)
```




    array([ 6,  8, 10, 12])




```python
# 행 기준으로 합계
num.sum(axis=1)
```




    array([10, 26])




```python
# 덧셈
np.add(num1, num2)
```




    array([ 6,  8, 10, 12])




```python
# 행렬 곱
np.dot(num1, num2)
```




    70



### Pandas

기계학습에서 가장 중요한 Package가 바로 **Pandas**이다.  
`pandas`의 기본적인 사용 방법을 알아보자.  
  
`numpy`와 마찬가지로 먼저 import문을 작성해야 한다.  


```python
import pandas as pd
```

자신이 분석하기 원하는 `.csv` 파일의 경로를 `read_csv()`함수의 매개변수로 전달하여 파일을 열 수 있다.  
해당 메소드로 파일을 열면, 자동으로 data frame으로 인식된다.  


```python
data = pd.read_csv('../Data/train.csv')

# 상위 5개 data만 출력
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>employee_id</th>
      <th>department</th>
      <th>region</th>
      <th>education</th>
      <th>gender</th>
      <th>recruitment_channel</th>
      <th>no_of_trainings</th>
      <th>age</th>
      <th>previous_year_rating</th>
      <th>length_of_service</th>
      <th>awards_won?</th>
      <th>avg_training_score</th>
      <th>is_promoted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>65438</td>
      <td>Sales &amp; Marketing</td>
      <td>region_7</td>
      <td>Master's &amp; above</td>
      <td>f</td>
      <td>sourcing</td>
      <td>1</td>
      <td>35</td>
      <td>5.0</td>
      <td>8</td>
      <td>0</td>
      <td>49</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>65141</td>
      <td>Operations</td>
      <td>region_22</td>
      <td>Bachelor's</td>
      <td>m</td>
      <td>other</td>
      <td>1</td>
      <td>30</td>
      <td>5.0</td>
      <td>4</td>
      <td>0</td>
      <td>60</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7513</td>
      <td>Sales &amp; Marketing</td>
      <td>region_19</td>
      <td>Bachelor's</td>
      <td>m</td>
      <td>sourcing</td>
      <td>1</td>
      <td>34</td>
      <td>3.0</td>
      <td>7</td>
      <td>0</td>
      <td>50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2542</td>
      <td>Sales &amp; Marketing</td>
      <td>region_23</td>
      <td>Bachelor's</td>
      <td>m</td>
      <td>other</td>
      <td>2</td>
      <td>39</td>
      <td>1.0</td>
      <td>10</td>
      <td>0</td>
      <td>50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>48945</td>
      <td>Technology</td>
      <td>region_26</td>
      <td>Bachelor's</td>
      <td>m</td>
      <td>other</td>
      <td>1</td>
      <td>45</td>
      <td>3.0</td>
      <td>2</td>
      <td>0</td>
      <td>73</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



그런데, 계속해서 데이터를 만지다보면 `list`나 `numpy` 형태로 계속해서 변환되는 현상이 발생할 수도 있다.  
그럴 때는 다시 pandas의 data frame 형태로 변환하여 분석을 이어가면 된다.  

*python으로만 데이터 분석을 해 온 사람은 아마 헷갈릴 일이 없겠지만, R이 더 익숙한 사람들은 이 개념이 어렵게 느껴질 수도 있다.*


```python
data = pd.DataFrame(data)
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>employee_id</th>
      <th>department</th>
      <th>region</th>
      <th>education</th>
      <th>gender</th>
      <th>recruitment_channel</th>
      <th>no_of_trainings</th>
      <th>age</th>
      <th>previous_year_rating</th>
      <th>length_of_service</th>
      <th>awards_won?</th>
      <th>avg_training_score</th>
      <th>is_promoted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>65438</td>
      <td>Sales &amp; Marketing</td>
      <td>region_7</td>
      <td>Master's &amp; above</td>
      <td>f</td>
      <td>sourcing</td>
      <td>1</td>
      <td>35</td>
      <td>5.0</td>
      <td>8</td>
      <td>0</td>
      <td>49</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>65141</td>
      <td>Operations</td>
      <td>region_22</td>
      <td>Bachelor's</td>
      <td>m</td>
      <td>other</td>
      <td>1</td>
      <td>30</td>
      <td>5.0</td>
      <td>4</td>
      <td>0</td>
      <td>60</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7513</td>
      <td>Sales &amp; Marketing</td>
      <td>region_19</td>
      <td>Bachelor's</td>
      <td>m</td>
      <td>sourcing</td>
      <td>1</td>
      <td>34</td>
      <td>3.0</td>
      <td>7</td>
      <td>0</td>
      <td>50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2542</td>
      <td>Sales &amp; Marketing</td>
      <td>region_23</td>
      <td>Bachelor's</td>
      <td>m</td>
      <td>other</td>
      <td>2</td>
      <td>39</td>
      <td>1.0</td>
      <td>10</td>
      <td>0</td>
      <td>50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>48945</td>
      <td>Technology</td>
      <td>region_26</td>
      <td>Bachelor's</td>
      <td>m</td>
      <td>other</td>
      <td>1</td>
      <td>45</td>
      <td>3.0</td>
      <td>2</td>
      <td>0</td>
      <td>73</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



#### Data 분석시 유용한 명령어

`pandas`의 data frame이 가지고 있는 강력한 명령어들이 많이 있다.   
실제 data 분석 시에도 자주 활용되므로 익혀놓는 것이 좋다.  


```python
# 배열의 행, 열의 갯수 확인
data.shape
```




    (54808, 13)




```python
# 행 번호 확인
data.index
```




    RangeIndex(start=0, stop=54808, step=1)




```python
# column(열) 확인
# dtype은 data type을 의미함
data.columns
```




    Index(['employee_id', 'department', 'region', 'education', 'gender',
           'recruitment_channel', 'no_of_trainings', 'age', 'previous_year_rating',
           'length_of_service', 'awards_won?', 'avg_training_score',
           'is_promoted'],
          dtype='object')




```python
# 값 확인
data.values
```




    array([[65438, 'Sales & Marketing', 'region_7', ..., 0, 49, 0],
           [65141, 'Operations', 'region_22', ..., 0, 60, 0],
           [7513, 'Sales & Marketing', 'region_19', ..., 0, 50, 0],
           ...,
           [13918, 'Analytics', 'region_1', ..., 0, 79, 0],
           [13614, 'Sales & Marketing', 'region_9', ..., 0, 45, 0],
           [51526, 'HR', 'region_22', ..., 0, 49, 0]], dtype=object)



#### DataFrame 인덱싱 추출

`pandas`의 인덱싱 추출 및 슬라이싱 방법은 일반적인 python의 방법과는 차이가 있으므로 유의하자.  
**index는 언제나 0번째부터 시작한다**는 것을 잊지 말자.


```python
# 행 추출 - 행번호 활용
data.iloc[5]
```




    employee_id                  58896
    department               Analytics
    region                    region_2
    education               Bachelor's
    gender                           m
    recruitment_channel       sourcing
    no_of_trainings                  2
    age                             31
    previous_year_rating           3.0
    length_of_service                7
    awards_won?                      0
    avg_training_score              85
    is_promoted                      0
    Name: 5, dtype: object



행을 추출할 때 유의할 점은 행번호와 index는 다를 수 있다는 것이다.  
현재는 행 번호와 index가 모두 동일하게 나타나지만, 행 수정 및 index setting에 따라 값이 얼마든지 달라질 수 있다.  
  
실제 data를 만지다보면 필요하지 않은 data들도 많이 있고, 정렬도 다시 해줘야 하는 경우도 많다.
때문에 행번호와 index가 헷갈리면 분석에 있어서 어려움을 겪을 수 있으니 조심하자.


```python
# 행 추출 - index 활용
data.loc[5]
```




    employee_id                  58896
    department               Analytics
    region                    region_2
    education               Bachelor's
    gender                           m
    recruitment_channel       sourcing
    no_of_trainings                  2
    age                             31
    previous_year_rating           3.0
    length_of_service                7
    awards_won?                      0
    avg_training_score              85
    is_promoted                      0
    Name: 5, dtype: object




```python
# 열 추출 - index 활용
data.iloc[:, 1]
```




    0        Sales & Marketing
    1               Operations
    2        Sales & Marketing
    3        Sales & Marketing
    4               Technology
                   ...        
    54803           Technology
    54804           Operations
    54805            Analytics
    54806    Sales & Marketing
    54807                   HR
    Name: department, Length: 54808, dtype: object




```python
# 열 추출 - column 명 활용
data['department']
```




    0        Sales & Marketing
    1               Operations
    2        Sales & Marketing
    3        Sales & Marketing
    4               Technology
                   ...        
    54803           Technology
    54804           Operations
    54805            Analytics
    54806    Sales & Marketing
    54807                   HR
    Name: department, Length: 54808, dtype: object




```python
# 열(column) 삭제
test = data.drop('employee_id', axis=1)
test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>department</th>
      <th>region</th>
      <th>education</th>
      <th>gender</th>
      <th>recruitment_channel</th>
      <th>no_of_trainings</th>
      <th>age</th>
      <th>previous_year_rating</th>
      <th>length_of_service</th>
      <th>awards_won?</th>
      <th>avg_training_score</th>
      <th>is_promoted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Sales &amp; Marketing</td>
      <td>region_7</td>
      <td>Master's &amp; above</td>
      <td>f</td>
      <td>sourcing</td>
      <td>1</td>
      <td>35</td>
      <td>5.0</td>
      <td>8</td>
      <td>0</td>
      <td>49</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Operations</td>
      <td>region_22</td>
      <td>Bachelor's</td>
      <td>m</td>
      <td>other</td>
      <td>1</td>
      <td>30</td>
      <td>5.0</td>
      <td>4</td>
      <td>0</td>
      <td>60</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sales &amp; Marketing</td>
      <td>region_19</td>
      <td>Bachelor's</td>
      <td>m</td>
      <td>sourcing</td>
      <td>1</td>
      <td>34</td>
      <td>3.0</td>
      <td>7</td>
      <td>0</td>
      <td>50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sales &amp; Marketing</td>
      <td>region_23</td>
      <td>Bachelor's</td>
      <td>m</td>
      <td>other</td>
      <td>2</td>
      <td>39</td>
      <td>1.0</td>
      <td>10</td>
      <td>0</td>
      <td>50</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Technology</td>
      <td>region_26</td>
      <td>Bachelor's</td>
      <td>m</td>
      <td>other</td>
      <td>1</td>
      <td>45</td>
      <td>3.0</td>
      <td>2</td>
      <td>0</td>
      <td>73</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



아주 기초적인 data 처리 방법들에 대해 알아봤다.  
data의 수가 많아질 수록, 기본적인 개념이 헷갈리면 난항을 많이 겪으니 처음부터 기초를 확실하게 잡는 것이 좋다.  
다음 Part에서는 머신러닝의 기본 골격을 잡아보자.  
