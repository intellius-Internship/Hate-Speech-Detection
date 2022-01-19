# Hate-Speech-Detection
혐오/선정적 발화 분류

### **Evaluation**
Accuracy와 F1 Score로 성능 측정

| CATEGORY | LABEL | COUNT |
|--|--|--|
| 혐오 | 0 | 89868 |
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

