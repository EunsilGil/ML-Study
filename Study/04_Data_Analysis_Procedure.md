## 04. 데이터 분석 절차

part 01 ~ 03 에서 알아본 내용을 전체적으로 정리해보았다.

### 1) Package 설정


```python
# 기본
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # Graph

# 데이터 가져오기
import pandas as pd

# 데이터 전처리
from sklearn.preprocessing import StandardScaler    # 연속 변수 표준화
from sklearn.preprocessing import LabelEncoder      # 범주형 변수 수치화

# 훈련/검증용 데이터 분리
from sklearn.model_selection import train_test_split    # 훈련과 테스트를 위한 데이터 분리

# # 분류 모델
# from sklearn.tree import DecisionTreeClassifier     # 의사결정나무
# from sklearn.naive_bayes import GaussianNB      # 나이브 베이즈 분류
# from sklearn.ensemble import RandomForestClassifier     # 랜덤 포레스트
# from sklearn.ensemble import BaggingClassifier  # 앙상블
# from sklearn.linear_model import LogisticRegression     # 로지스틱 회귀분석
# from sklearn.svm import SVC      # SVM(서포트벡터머신)
# from sklearn.neural_network import MLPClassifier    # 다층 인공신경망

# 모델 검정
from sklearn.metrics import confusion_matrix, classification_report # 정오분류표
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer  # 정확도, 민감도 등
from sklearn.metrics import roc_curve   # ROC 곡선

# 최적화
from sklearn.model_selection import cross_validate  # 교차 타당도
from sklearn.pipeline import make_pipeline  # 파이프라인 구축
from sklearn.model_selection import learning_curve, validation_curve # 학습곡선, 검증곡선
from sklearn.model_selection import GridSearchCV    # 하이퍼파라미터 튜닝
```

### 2) 데이터 가져오기

#### 2-1) 데이터 프레임으로 저장
* csv 데이터를 dataframe으로 가져오기


```python
train = pd.read_csv('../Data/train.csv')
test = pd.read_csv('../Data/test.csv')

# train data의 상위 5개 출력
train.head()
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



* 자료구조 살펴보기


```python
print(train.info())
print(test.info())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 54808 entries, 0 to 54807
    Data columns (total 13 columns):
     #   Column                Non-Null Count  Dtype  
    ---  ------                --------------  -----  
     0   employee_id           54808 non-null  int64  
     1   department            54808 non-null  object 
     2   region                54808 non-null  object 
     3   education             52399 non-null  object 
     4   gender                54808 non-null  object 
     5   recruitment_channel   54808 non-null  object 
     6   no_of_trainings       54808 non-null  int64  
     7   age                   54808 non-null  int64  
     8   previous_year_rating  50684 non-null  float64
     9   length_of_service     54808 non-null  int64  
     10  awards_won?           54808 non-null  int64  
     11  avg_training_score    54808 non-null  int64  
     12  is_promoted           54808 non-null  int64  
    dtypes: float64(1), int64(7), object(5)
    memory usage: 5.4+ MB
    None
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 23490 entries, 0 to 23489
    Data columns (total 12 columns):
     #   Column                Non-Null Count  Dtype  
    ---  ------                --------------  -----  
     0   employee_id           23490 non-null  int64  
     1   department            23490 non-null  object 
     2   region                23490 non-null  object 
     3   education             22456 non-null  object 
     4   gender                23490 non-null  object 
     5   recruitment_channel   23490 non-null  object 
     6   no_of_trainings       23490 non-null  int64  
     7   age                   23490 non-null  int64  
     8   previous_year_rating  21678 non-null  float64
     9   length_of_service     23490 non-null  int64  
     10  awards_won?           23490 non-null  int64  
     11  avg_training_score    23490 non-null  int64  
    dtypes: float64(1), int64(6), object(5)
    memory usage: 2.2+ MB
    None



```python
# 배열의 행, 열의 갯수 확인
print(train.shape)
print(test.shape)
```

    (54808, 13)
    (23490, 12)



```python
# column 값 확인
train.keys()
```




    Index(['employee_id', 'department', 'region', 'education', 'gender',
           'recruitment_channel', 'no_of_trainings', 'age', 'previous_year_rating',
           'length_of_service', 'awards_won?', 'avg_training_score',
           'is_promoted'],
          dtype='object')



#### 2-2) Data와 Target으로 분리
* 필요한 data만 추출
* data : X , target : y 로 분리


```python
X = train.drop(['is_promoted'], axis=1)
X.head()
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
    </tr>
  </tbody>
</table>
</div>




```python
y = train['is_promoted']
y.head()
```




    0    0
    1    0
    2    0
    3    0
    4    0
    Name: is_promoted, dtype: int64



### 3) 데이터 전처리

#### 3-1) data(X) 레이블 인코딩
* 문자형 자료를 숫자(범주형)로 인코딩  
* 숫자형 자료를 표준화  
* 의사결정나무, 랜덤 포레스트, 나이브 베이즈 분류에서는 원본 데이터를 그대로 유지하면 됨  

#### 3-2) Class(Target) 레이블 인코딩
내가 가지고 있는 data의 target은 int형 값이므로 새롭게 인코딩 하지는 않았지만, 만약 레이블링 해야한다면 다음 code를 활용할 수 있다.


```python
# class_le = LabelEncoder()
# y = class_le.fit_transform(y)
```

### 4) 훈련 / 검증용 데이터 분할


```python
X_train, X_test, y_train, y_test = \
    train_test_split(X, y,
                     test_size = 0.3,   # test set의 비율
                     random_state = 1,  # 무작위 시드 번호
                     stratify = y)      # 결과 레이블의 비율대로 분리

```

### 5) 모델 구축

모델은 각각 직접 사용할 때 다시 호출하겠다.  

### 6) 모델 검정

훈련용 데이터의 정확도와 검증용 데이터의 정확도를 비교하며 검증하는 것이 일반적이다.   
이외에 검증용 데이터로 예측하는 방법도 존재한다.   
`predict()` : class의 결과값으로 표시  
`predict_proba()` : 확률 값으로 표시  


**아직 model을 선정하지도, 돌려보지도 않았기 때문에 아래의 code는 오류가 난다.**


```python
y_pred = tree.predict(X_test)

```


```python
y_pred = tree.predict_proba(X_test)
```

#### 정오분류표
`confusion_matrix()` : table을 만듦
`y_test` : 실제 값 (행)
`y_pred` : 예측 값 (열)


```python
confmat = pd.DataFrame(confusion_matrix(y_test, y_pred),
                       index=['True[0]', 'True[1]'],
                       columns = ['Predict[0]', 'Predict[1]'])
```


```python
print('Classification Report')
print(classification_report(y_test, y_pred))
```

#### 정확도, 민감도 확인
* 정밀도와 재현율은 class가 2개인 경우에만 실행


```python
print(f'잘못 분류된 sample 갯수 : {(y_test != y_pred).sum()}')
print(f'정확도 : {accuracy_score(y_test, y_pred):.2f}%')
print(f'정밀도 : {precision_score(y_true=y_test, y_pred=y_pred):.3f}%')
print(f'재현율 : {recall_score(y_true=y_test, y_pred=y_pred):.3f}%')
print(f'F1 : {f1_score(y_true=y_test, y_pred=y_pred):.3f}%')
```

#### 결과값을 해석하는 방법

|정오행렬|분류행렬|
|---|---|
|정확도|오류율|
|TPR|FPR|
|민감도|특이도|
|재현율|정밀도|

실제로는 같은 식을 사용하기에 아래 3가지는 같은 의미이다.   
* TPR = 민감도(Sensitivity) = 재현율(Recall)   

정확도는 class 0과 1 모두를 정확하게 분류함   
오류율은 class 0과 1 모두를 정확하게 분류하지 못함  

TPR(True Positive Rate)는 실제 class 1 중에 잘 맞춘 것을 의미    
FPR(False Positive Rate)는 실제 calss 0 중에 못 맞춘 것을 의미    
* 때문에 FPR은 `1-FPR`로 많이 사용한다.   
*`1-FPR`은 특이도와 같다.*    

민감도는 실제 class 1 중에 잘 맞춘 것을 의미하므로 TPR과 같다.   
특이도는 실제 class 0 중에 잘 맞춘 것을 의미하므로 `1-FPR`이 된다.  

재현율(Recall)은 **실제** class 1 중에 잘 맞춘 것이므로 `민감도`, `TPR`과 다 동일하게 사용할 수 있다.   
정밀도(Precision)은 **예측** class 1 중에 잘 맞춘 것을 의미한다.   

이것을 모두 합쳐 사용하는 개념이 `F1`이다.    
실제로 잘 맞춘 것과 예측에서도 잘 맞춘 것을 한꺼번에 계산하는 것이다.    

$$F1 = 2 \times {{재현율\times정밀도} \over {재현율+정밀도}}$$    

때문에 F1값이 높은 model의 성능이 뛰어나다고 이야기 할 수 있다.  

#### ROC 곡선
* decision_function 사용이 가능한 모델일 경우 : `tree.decision_function(X_test)`
* decision_function 사용이 불가능한 모델일 경우 : `tree.predict_proba(X_test)[:,1]`


```python
# 둘 중 model에 맞는 구문을 활용
# fpr, tpr, thresholes = roc_curve(y_test, tree.decision_function(X_test))
fpr, tpr, thresholes = roc_curve(y_test, tree.predict_proba(X_test)[:. 1])
```


```python
# 값 확인
fpr, tpr, thresholes
```


```python
# Graph 생성
plt.plot(fpr, tpr, '--', label="Decision Tree")
plt.plot([0,1], [0,1], 'k--', label="random guess")
plt.plot([fpr], [tpr], 'r--', ms=10)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title("ROC Curve")
plt.show()
```
