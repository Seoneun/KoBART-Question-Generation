# KoBART-Question Generation

## Load KoBART
- huggingface.co에 있는 binary를 활용
  - https://huggingface.co/gogamza/kobart-base-v1

## Download binary
```python
import torch
from transformers import PreTrainedTokenizerFast
from transformers import BartForConditionalGeneration

tokenizer = PreTrainedTokenizerFast.from_pretrained('Sehong/kobart-QuestionGeneration')
model = BartForConditionalGeneration.from_pretrained('Sehong/kobart-QuestionGeneration')

text = """
1989년 2월 15일 여의도 농민 폭력 시위를 주도한 혐의(폭력행위등처벌에관한법률위반)으로 지명수배되었다. 1989년 3월 12일 서울지방검찰청 공안부는 임종석의 사전구속영장을 발부받았다. 같은 해 6월 30일 평양축전에 임수경을 대표로 파견하여 국가보안법위반 혐의가 추가되었다. 경찰은 12월 18일~20일 사이 서울 경희대학교에서 임종석이 성명 발표를 추진하고 있다는 첩보를 입수했고, 12월 18일 오전 7시 40분 경 가스총과 전자봉으로 무장한 특공조 및 대공과 직원 12명 등 22명의 사복 경찰을 승용차 8대에 나누어 경희대학교에 투입했다. 1989년 12월 18일 오전 8시 15분 경 서울청량리경찰서는 호위 학생 5명과 함께 경희대학교 학생회관 건물 계단을 내려오는 임종석을 발견, 검거해 구속을 집행했다. 임종석은 청량리경찰서에서 약 1시간 동안 조사를 받은 뒤 오전 9시 50분 경 서울 장안동의 서울지방경찰청 공안분실로 인계되었다. <unused0> 1989년 2월 15일
"""

raw_input_ids = tokenizer.encode(text)
input_ids = [tokenizer.bos_token_id] + raw_input_ids + [tokenizer.eos_token_id]

summary_ids = model.generate(torch.tensor([input_ids]),  num_beams=4,  max_length=512,  eos_token_id=1)
tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)

'임종석이 지명수배된 날짜는?'

```
## Requirements
```
torch==1.8.0
transformers==4.18.0
```

## Training Environment
 - Ubuntu
 - RTX 3090

## Data
- KorQuAD1.0 의 학습 데이터를 활용함
- 데이터 탐색에 용이하게 tsv 형태로 데이터를 변환함
- Data 구조
    - Train Data : 60,408
    - Test Data : 5,775
- default로 data/train.tsv, data/dev.tsv 형태로 저장함
  
| content  | question |
|-------|--------:|
| 본문 + <unused0> + 정답| 질문 |  

## How to Train
- KoBART summarization fine-tuning
```bash
pip install -r requirements.txt

[use gpu]
python train.py 

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
|2|koBART|1989년 6월 30일 평양축전에 누구를 대표로 파견하여 국가보안법위반 혐의가 추가되었는가?|

| ||Text|
|-------|-------|-------|
|3|Answer|1989년|
|3|Label|임종석이 여의도 농민 폭력 시위를 주도한 혐의로 지명수배된 연도는?|
|3|koBART|임종석이 서울지방검찰청 공안부에서 사전구속영장을 발부받은 해는?|



## Model Performance
- Test Data 기준으로 rouge score를 산출함
- Score 산출 방법은 Dacon 한국어 문서 생성요약 AI 경진대회 metric을 활용함
 

| | BLEU-1 |BLEU-2|BLEU-3|BLEU-4|
|-------|--------:|--------:|--------:|--------:|
| Precision| 0.515 | 0.351|0.415|0|

## Demo
  
https://huggingface.co/Sehong/kobart-QuestionGeneration
  
## Reference
- [KoBART](https://github.com/SKT-AI/KoBART)
- [KoBART-summarization](https://github.com/haven-jeon/KoBART-chatbot](https://github.com/seujung/KoBART-summarization)
