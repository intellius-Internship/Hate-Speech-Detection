# Hate-Speech-Detection
혐오/선정적 발화 분류

### **Evaluation**
Accuracy와 F1 Score로 성능 측정

| CATEGORY | LABEL | COUNT |
|--|--|--|
| 폭력 | 0 | 89868 |
| 일반 | 1 | 99920 |

<br>


## 📈 **메소드 별 성능 비교**

<br>

### **Performance of PLM** 

| Model | Accuracy | F1 Score |
|--|--|--|
| `monologg/koelectra-base-v3` | 95.39 | 95.37 | 
| `monologg/kobert` | 52.64 | 34.49 |
| `monologg/kobigbird-bert-base` | 95.46 | 95.44 |

<br>

### **Performance of ML Model**

| Model | Accuracy | F1 Score |
|--|--|--|
| XGBoost | 84.94 | 84.93 | 
| Decision Tree | 79.01 | 78.84 |
| SVC | 89.09 | 89.02 |
| KNeighbors | 53.63 | 36.92 |


<br>


## **분류 예시**

0 (폭력), 1 (일반)으로 분류

```bash
# ELECTRA 분류

2배줬다 수고해라 게이야	0
사람을 가지고 놀다가 제 자리에 갖다 놓는구나. 8년의 세월과 시련은 누가 보상하냐?	 0 
개돼지뇬    1
어이구 개좆병신아	0
```


```bash
# BigBird 분류

2배줬다 수고해라 게이야	0
사람을 가지고 놀다가 제 자리에 갖다 놓는구나. 8년의 세월과 시련은 누가 보상하냐?	 1 
개돼지뇬    0
어이구 개좆병신아	0
```

```bash
# SVC 분류

2배줬다 수고해라 게이야	0
사람을 가지고 놀다가 제 자리에 갖다 놓는구나. 8년의 세월과 시련은 누가 보상하냐?	 0 
개돼지뇬    1
어이구 개좆병신아	0
```