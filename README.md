# 축구선수의 유망 여부 예측 AI 경진대회

```
Final Ranking : 13/89 (Top 14.6%)
```
</br>

## Introduction
</br>

__This repository is a place to share "축구선수의 유망 여부 예측 AI 경진대회" solution code.__
</br>

```
주최 및 주관 : 데이콘
```

<br>
## Repository Structure

```
│  README.md
│  
├─Data_Preprocessing
│      Data_Preprocessing.ipynb
│      
└─Model
    ├─Another_Build_Model
    │      xgb_k_fold_ensemble.py
    │      xgb_optuna.py
    │      
    └─Final_Submission_Model
            Gradient_Boosting_Model.py
```
<br>

## Development Environment
</br>

```
CPU : Intel i9-10900F
GPU : NVIDIA GeForce RTX 3080 Ti
RAM : 32GB
```
</br>

## Approach Method Summary
</br>

```
*  Many Independent Variables (Total 66) 
    -> Feature removal through Pearson Correlation Analysis between Independent Variables


* Data Imbalanced -> SMOTE


* Building various Models 
```

## What I learned from this Competition
</br>

```

Before the release of the Private score, the Public score is 3rd place
    
    -> But Private score is 13th place
    
    -> The smaller the number of test data, the greater the change in rank with 
        one or two correct answers.
    

Conclusion

    -> I plan to improve my ability to build robust models not only for train data
        but also for test data.
    
```