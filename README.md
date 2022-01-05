# Analysis of Zero-Shot Crosslingual Learning between English and Korean for Named Entity Recognition

## Introduction

- This repository contains the dataset and implementation of the paper : Analysis of Zero-Shot Crosslingual Learning between English and Korean for Named Entity Recognition (https://aclanthology.org/2021.mrl-1.19.pdf)

## Dataset

- The dataset is based on the Korean-English AI Training Text Corpus (KEAT) provided by AI Open Innovation Hub, which is operated by the Korean National Information Society Agency (NIA).
  - URL for the original KEAT corpus : https://aihub.or.kr/aidata/87
- The original KEAT corpus contains 1.6M Korean-English parallel sentences from various sources such as news media, government website/journal, law & administration, conversation and etc.
- In our research, we preprocess and annotate 800K parallel sentences from the news portion to make our final dataset.

### Statistics

| Split | # of Parallel Sentences | # of Parallel Articles |
| ----- | ----------------------- | ---------------------- |
| TRAIN | 747,853                 | 379,773                |
| DEV   | 5,034                   | 700                    |
| TEST  | 4,810                   | 700                    |
| TOTAL | 757,697                 | 381,173                |

### Format

#### Development and Test set

- Development and Test set contain gold labels (en_ner_gold, ko_ner_gold) and their alignment(en_ko_ner_gold).

- The dataset is in a JSON format as follows :

  ~~~python
  [
  	{ # example of the 1st article
          'catagory' : ['문화', '학술_문화재'],
          'date' : '2019-05-31T00:00:00.000Z',
          'source' : '국민일보',
          'text' :
          	[
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
                  { # example of the 2nd sentence in the 1st article
                      ...
                  },
                  ... 
          	]
  	},
      { # example of the 2nd article
          ...
      },
  	... 
  ]
  ~~~


#### Train set

- Train set does not contain gold labels and label alignment between entities, but has pseudo labels annotated with ELIT.

- The dataset is in a JSON format as follows :

  ```python
  [
  	{ # example of the 1st article
          'catagory' : ['문화', '학술_문화재'],
          'date' : '2019-05-31T00:00:00.000Z',
          'source' : '국민일보',
          'text' :
          	[
                  { # example of the 1st sentence in the 1st article
                  'aihub_id' : 1530019,
                  'en_ner_auto' : [[2, 3, 'DATE'], [4, 8, 'ORG'], [22, 27, 'DATE']],
                  'en_text' : 'Founded in 1990, the Special Needs Institute has developed a variety of curative programs to study brain development and educate children under the age of 5 with developmental disabilities and autism.',
                  'en_tokens' : ['Founded', 'in', '1990', ',', 'the', 'Special', 'Needs', 'Institute', 'has', 'developed', 'a', 'variety', 'of', 'curative', 'programs', 'to', 'study', 'brain', 'development', 'and', 'educate', 'children', 'under', 'the', 'age', 'of', '5', 'with', 'developmental', 'disabilities', 'and', 'autism', '.'], 
                  'ko_ner_auto' : [[0, 2, 'DAT'], [14, 17, 'NOH']],
                  'ko_text' : '1990년 설립된 특수요육원은 뇌 발달을 연구해 만 5세 이하 발달장애 및 자폐증 아이를 치료하면서 교육할 수 있는 다양한 요육프로그램을 개발했다.', 
                  'ko_tokens' : ['1990', '년', '설립', '된', '특수', '요', '육', '원', '은', '뇌', '발달', '을', '연구', '해', '만', '5', '세', '이하', '발달', '장애', '및', '자폐증', '아이', '를', '치료', '하', '면서', '교육', '할', '수', '있', '는', '다양', '한', '요육', '프로그램', '을', '개발', '했', '다', '.']
                  },
                  { # example of the 2nd sentence in the 1st article
                      ...
                  },
                  ... 
          	]
  	},
      { # example of the 2nd article
          ...
      },
  	... 
  ]
  ```

## Citation

```
@inproceedings{kim-etal-2021-analysis,
    title = "Analysis of Zero-Shot Crosslingual Learning between {E}nglish and {K}orean for Named Entity Recognition",
    author = "Kim, Jongin  and
      Choi, Nayoung  and
      Lim, Seunghyun  and
      Kim, Jungwhan  and
      Chung, Soojin  and
      Woo, Hyunsoo  and
      Song, Min  and
      Choi, Jinho D.",
    booktitle = "Proceedings of the 1st Workshop on Multilingual Representation Learning",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.mrl-1.19",
    doi = "10.18653/v1/2021.mrl-1.19",
    pages = "224--237",
    abstract = "This paper presents a English-Korean parallel dataset that collects 381K news articles where 1,400 of them, comprising 10K sentences, are manually labeled for crosslingual named entity recognition (NER). The annotation guidelines for the two languages are developed in parallel, that yield the inter-annotator agreement scores of 91 and 88{\%} for English and Korean respectively, indicating sublime quality annotation in our dataset. Three types of crosslingual learning approaches, direct model transfer, embedding projection, and annotation projection, are used to develop zero-shot Korean NER models. Our best model gives the F1-score of 51{\%} that is very encouraging, considering the extremely distinct natures of these two languages. This is pioneering work that explores zero-shot cross-lingual learning between English and Korean and provides rich parallel annotation for a core NLP task such as named entity recognition.",
}
```

