# Analysis of Zero Shot Crosslingual Learning between English and Korean for Named Entity Recognition

## Introduction

- This repository contains the dataset and implementation of the paper : Analysis of Zero Shot Crosslingual Learning between English and Korean for Named Entity Recognition (The paper is accepted by MRL 2021 and the URL will be updated soon)

## Dataset

- The dataset is based on the Korean-English AI Training Text Corpus (KEAT) provided by AI Open Innovation Hub, which is operated by the Korean National Information Society Agency (NIA).
  - URL for the original KEAT corpus : https://aihub.or.kr/aidata/87
- The original KEAT corpus contains 1.6M Korean-English parallel sentences from various sources such as news media, government website/journal, law & administration, conversation and etc.
- In our research, we preprocess and annotate 800K parallel sentences from the news portion to make our final dataset.



- The dataset is in a JSON format as follows :

  ~~~python
  [
  	{ # example of the 1st article
  	'catagory' : ['문화', '학술_문화재'],
  	'date' : '2019-05-31T00:00:00.000Z',
  	'source' : '국민일보',
  	'text' :[
  		{ # example of the 1st sentence in the 1st article
  		'aihub_id' : 1530019,
          'en_ko_ner_gold' : [[0, 0], [2, 1]],
          'en_ner_auto' : [[2, 3, 'DATE'], [4, 8, 'ORG'], [22, 27, 'DATE']],
          'en_ner_gold' : [[2, 3, 'DATE'], [5, 8, 'ORG'], [22, 27, 'CARDINAL']],
          'en_text' : 'Founded in 1990, the Special Needs Institute has developed a variety of curative programs to study brain development and educate children under the age of 5 with developmental disabilities and autism.',
          'en_tokens' : ['Founded', 'in', '1990', ',', 'the', 'Special', 'Needs', 'Institute', 'has', 'developed', 'a', 'variety', 'of', 'curative', 'programs', 'to', 'study', 'brain', 'development', 'and', 'educate', 'children', 'under', 'the', 'age', 'of', '5', 'with', 'developmental', 'disabilities', 'and', 'autism', '.'], 
          'ko_ner_auto' : [[0, 2, 'DAT'], [14, 17, 'NOH']],
          'ko_ner_gold' : [[0, 2, 'DATE'], [14, 17, 'CARDINAL']],
          'ko_text' : '1990년 설립된 특수요육원은 뇌 발달을 연구해 만 5세 이하 발달장애 및 자폐증 아이를 치료하면서 교육할 수 있는 다양한 요육프로그램을 개발했다.', 
          'ko_tokens' : ['1990', '년', '설립', '된', '특수', '요', '육', '원', '은', '뇌', '발달', '을', '연구', '해', '만', '5', '세', '이하', '발달', '장애', '및', '자폐증', '아이', '를', '치료', '하', '면서', '교육', '할', '수', '있', '는', '다양', '한', '요육', '프로그램', '을', '개발', '했', '다', '.']
          },
          ... # example of the nth sentence in the 1st article
      ]
  	},
  	... # example of the nth article
  ]
  ~~~

  

### Statistics

| Split | # of Parallel Sentences | # of Parallel Articles |
| ----- | ----------------------- | ---------------------- |
| TRAIN | 747,853                 | 379,773                |
| DEV   | 5,034                   | 700                    |
| TEST  | 4,810                   | 700                    |
| TOTAL | 757,697                 | 381,173                |



## Implementation

TBU

### Architecture

### Basic Setup

### Evaluation

### Training

## Citation

TBU