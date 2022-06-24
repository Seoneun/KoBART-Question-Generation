# KoBART-Question Generation (+Post-training)

## Post-training
 - BART의 Text infilling loss와 QG loss를 합쳐 Post-training을 진행
 - 추가로, KoBART의 pre-training에서는 사용하지 않은 Sentence permutation을 포함하여 (Sentence permutation + Text infilling) loss와 QG loss를 합쳐 Post-training을 진행 (+ 2022.06.24)
 - Post-training에서 QG는 AI-hub의 기계독해 데이터셋을 사용

## How to Train
- KoBART Question Generation fine-tuning
```bash
[use gpu]
python post_train.py 

```


## Generation Sample
- Text infilling + QG


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


- Text infilling + Sentence permutation + QG


| ||Text|
|-------|-------|-------|
|1|Answer|1989년 2월 15일|
|1|Label|임종석이 여의도 농민 폭력 시위를 주도한 혐의로 지명수배 된 날은?|
|1|koBART|임종석이 지명수배된 날짜는?|

| ||Text|
|-------|-------|-------|
|2|Answer|임수경|
|2|Label|1989년 6월 30일 평양축전에 대표로 파견 된 인물은?|
|2|koBART|1989년 당시 서울지방검찰청 공안부는 누구를 대표로 파견하여 국가보안법위반 혐의를 추가하였는가?|

| ||Text|
|-------|-------|-------|
|3|Answer|1989년|
|3|Label|임종석이 여의도 농민 폭력 시위를 주도한 혐의로 지명수배된 연도는?|
|3|koBART|임종석이 지명수배된 해는?|



## Model Performance
- Test Data 기준으로 BLEU score를 산출함
 
- Text infilling + QG

| |BLEU-1|BLEU-2|BLEU-3|BLEU-4|
|------|:-------:|:-------:|:-------:|:-------:|
|Score|43.59|32.54|24.84|19.20|

- Text infilling + Sentence permutation + QG


| |BLEU-1|BLEU-2|BLEU-3|BLEU-4|
|------|:-------:|:-------:|:-------:|:-------:|
|Score|43.84|32.73|25.08|19.34|


## Reference
- [KoBART](https://github.com/SKT-AI/KoBART)
- [KoBART-summarization](https://github.com/seujung/KoBART-summarization)
