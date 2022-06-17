# KoBART-Question Generation (+Post-training)

## How to Train
- KoBART Question Generation fine-tuning
```bash
[use gpu]
python train.py 

```

## How to Inference
```bash
[use gpu]
python generate.py 

```

## Generation Sample
| ||Text|
|-------|-------|-------|
|1|Answer|1989년 2월 15일|
|1|Label|임종석이 여의도 농민 폭력 시위를 주도한 혐의로 지명수배 된 날은?|
|1|koBART|임종석이 지명수배된 날짜는?|

| ||Text|
|-------|-------|-------|
|2|Answer|임수경|
|2|Label|1989년 6월 30일 평양축전에 대표로 파견 된 인물은?|
|2|koBART|1989년 6월 30일 평양축전에 임종석이 대표로 파견된 인물은?|

| ||Text|
|-------|-------|-------|
|3|Answer|1989년|
|3|Label|임종석이 여의도 농민 폭력 시위를 주도한 혐의로 지명수배된 연도는?|
|3|koBART|임종석이 지명수배된 해는?|



## Model Performance
- Test Data 기준으로 BLEU score를 산출함
 
  
| |BLEU-1|BLEU-2|BLEU-3|BLEU-4|
|------|:-------:|:-------:|:-------:|:-------:|
|Score|43.59|32.54|24.84|19.20|

  
## Reference
- [KoBART](https://github.com/SKT-AI/KoBART)
- [KoBART-summarization](https://github.com/seujung/KoBART-summarization)