#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/greatsong/2020-deep-learning-basic-class/blob/master/%E1%84%83%E1%85%B5%E1%86%B8%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC_%E1%84%87%E1%85%A7%E1%86%BC%E1%84%8B%E1%85%A1%E1%84%85%E1%85%B5%E1%84%87%E1%85%A1%E1%86%AB_%E1%84%8B%E1%85%A7%E1%84%89%E1%85%A5%E1%86%BA_%E1%84%87%E1%85%A5%E1%86%AB%E1%84%8D%E1%85%A2_%E1%84%89%E1%85%B5%E1%84%80%E1%85%A1%E1%86%AB_%E1%84%89%E1%85%A9%E1%86%AB%E1%84%80%E1%85%B3%E1%86%AF%E1%84%8A%E1%85%B5_%E1%84%8B%E1%85%A8%E1%84%8E%E1%85%B3%E1%86%A8_%E1%84%86%E1%85%AE%E1%86%AB%E1%84%8C%E1%85%A6.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


# tensorflow 라이브러리 설치
get_ipython().system('pip install tensorflow')


# In[ ]:


# 딥러닝 관련 라이브러리 불러오기
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


# MNIST 데이터셋 불러오기
mnist = keras.datasets.mnist


# In[ ]:


# MNIST 데이터셋 학습용(x,y), 테스트용(x,y)으로 나누기
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[ ]:


# 학습용 데이터 형태 살펴보기
x_train.shape


# In[ ]:


# 학습용 첫 번째 데이터 살펴보기1
x_train[0]


# In[ ]:


# 학습용 첫 번째 데이터 살펴보기2
y_train[0]


# In[ ]:


# 데이터 전처리(0 ~ 1 사이 숫자로)
x_train = x_train / 255
x_test = x_test / 255


# In[10]:


# 데이터 전처리 결과 확인
x_train[0]


# In[11]:


# 모델 만들기 : 입력층(784) - 은닉층1(256) - 은닉층2(128) - 은닉층3(64) - 출력층(10)
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape = (28, 28)),
    keras.layers.Dense(256, activation = 'relu'),
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dense(64, activation = 'relu'),
    keras.layers.Dense(10, activation = 'softmax') # 확률 출력해준다
])


# In[12]:


# 모델 컴파일 : 최적화 함수, 손실 함수 설정 + 평가 지표 설정 + 가중치 초기화)
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])


# In[13]:


# 모델 확인
model.summary()


# In[ ]:


# 모델 학습 : 전체 데이터는 5번 반복
model.fit(x_train, y_train, epochs = 5)


# In[ ]:


# 모델 평가
model.evaluate(x_test, y_test)


# In[ ]:


# 예측 - 0번째 숫자 이미지로 보기
plt.imshow(x_train[0], cmap='gray')
plt.show()


# In[ ]:


# 예측 - 0번째 숫자 예측하기1
print(model.predict(x_train[0].reshape(1, 28, 28)))


# In[ ]:


# 예측 - 0번째 숫자 예측하기1
print(np.argmax(model.predict(x_train[0].reshape(1, 28, 28))))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




