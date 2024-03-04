---
jupyter:
  jupytext:
    cell_metadata_filter: -all
    formats: md,ipynb
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

## Install required packages

```python
#!pip3 install -U spacy
#!python -m spacy download en_core_web_sm

# !pip3 install -U 'transformers[torch]'
# !pip install -U optimum
# !pip3 install -U dataset
# !pip3 install -U evaluate

# !pip3 install -U numpy
# !pip install -U scikit-learn
```

## Prepare training data for Spacy NER
The data is annotated using [Label Studio](https://labelstud.io/), includes the first 232 items in sample.txt  
Data is transformed to Spacy format, saved to "./ner/train.spacy"

```python
import json

with open('ner/ner_train.json', mode='r', encoding='utf-8') as fd:
    ner_train_data = json.loads(fd.read());

print('Number of samples: ', len(ner_train_data))
```

```python
print(ner_train_data[5])
```

```python
print(ner_train_data[16])
```

```python
import spacy
from spacy.tokens import DocBin

nlp = spacy.load("en_core_web_sm")

db = DocBin()
for item in ner_train_data:
    text = item['data']['text']
    annotations = item['annotations']

    if len(text) < 20:
        continue

    doc = nlp(text)
    ents = []

    # [E1010] Unable to set entity information for token 0 which is included in more than one span in entities, blocked, missing or outside.
    # Eg:
    #   text: Gold Price News and Forecast: XAU/USD is trapped in daily support and resistance. Posted by: EUR Editor  in EUR  1 min ago  XAU/USD struggles below $1,800 amid risk-off mood. Gold steps back from intraday high while flashing $1,772 as a quote amid Friday’s Asian session. In doing … Read Full Story at source (may require registration)     Latest posts by EUR Editor ( see all )
    #   ents: [Gold Price News, XAU/USD, daily, 1 min ago, XAU/USD, 1,800, 1,772, Friday, Asian, … Read Full Story, EUR Editor, Gold]
    #   span as text[0:16] and span as text[0:4] are overlaping each other -> token 0 which is included in more than one span

    annotated = annotations[0]

    for r in annotated['result']:
        value = r['value']
        span = doc.char_span(
            value['start'],
            value['end'],
            label=value['labels'][0]
            )
        if span is None:
            print('[result] ', r);
            print('[span] ', span);
            print('[text] ', item['data']['text']);
            raise Error('The entity is marked inproperly that makes span to None');
        else:
            ents.append(span)

    for ent in doc.ents:
        span = doc.char_span(ent.start_char, ent.end_char, label=ent.label_)
        is_span_overlaping = False
        for included_span in ents:
            if ((included_span.start <= span.start and span.start <= included_span.end)
                or (included_span.start <= span.end and span.end <= included_span.end)
                or (span.start <= included_span.start and included_span.start <= span.end)
                or (span.start <= included_span.end and included_span.end <= span.end)):
                    is_span_overlaping = True

        if is_span_overlaping:
            continue

        ents.append(span)

    if len(ents) == 0:
        continue
    doc.ents = ents
    db.add(doc)

db.to_disk("./ner/train.spacy")
```
## Prepare training data for Spacy TextCategorizer
The data is annotated using Label Studio, includes 45 out of the first 232 items in sample.txt  
Data is transformed to Spacy format, saved to "./sentiment_analysis_spacy/train.spacy"  

```python
import json

with open('sentiment_analysis_spacy/sentiment_train.json', mode='r', encoding='utf-8') as fd:
    sent_train_data = json.loads(fd.read());
```
```python
print(sent_train_data[5])
```

```python
from pathlib import Path
import spacy
from spacy.tokens import DocBin

def read_categories():
    return Path('sentiment_analysis_spacy/categories.txt').open().read().strip().split("\n")
categories = read_categories();
print('categories: ', categories)

nlp = spacy.blank("en")
db = DocBin()
for item in sent_train_data:
    text = item['data']['text']
    annotated = item['annotations'][0]

    if len(text) < 20:
        continue

    result = annotated['result']
    if len(result) == 0:
        continue;
    label = result[0]['value']['choices'][0]

    doc = nlp.make_doc(text)
    doc.cats = {category: 0 for category in categories}
    # True labels get value 1
    doc.cats[label] = 1

    db.add(doc)

print('Number of samples: ', len(db))

db.to_disk("./sentiment_analysis_spacy/train.spacy")
```

## Train Spacy NER model
The training set up is placed at directory './ner', two important files are config.cfg and train.spacy.  
Note: To avoid spending too much time in annotation, training data also is used as validation data. In actual practice validation data must differ from training data to evaluate model's generisation.  
Trained model is stored in "./ner/output" for later use.

```python
!python3 -m spacy train ner/config.cfg --output ./ner/output --paths.train ./ner/train.spacy --paths.dev ./ner/train.spacy
```

## Train Spacy TextCategorizer model
The training set up is placed at directory './sentiment_analysis_spacy', two important files are config.cfg and train.spacy.  
Note: To avoid spending too much time in annotation, training data also is used as validation data. In actual practice validation data must differ from training data to evaluate model's generisation.  
Trained model is stored in "./sentiment_analysis_spacy/output" for later use.

```python
!python3 -m spacy train sentiment_analysis_spacy/config.cfg --output ./sentiment_analysis_spacy/output --paths.train ./sentiment_analysis_spacy/train.spacy --paths.dev ./sentiment_analysis_spacy/train.spacy
```

## Evaluate Sentiment Analysis on gold and silver commodities related content
Two models are used to recognise sentiment, the results are then compared against each other.  
One model is Spacy model, another is [FinancialBERT](https://huggingface.co/ahmedrachid/FinancialBERT-Sentiment-Analysis) model from Huggingface  
Note: one disadvantage of FinancialBERT model is that it can not process text that has more than 512 tokens.

```python
# to exclude first 232 rows that have been used for training
padding = 231

with open('sample.txt', mode='r', encoding='utf-8') as fd:
    samples = fd.readlines()
samples = samples[padding:]
```

#### Load Spacy model

```python
import spacy
ner = spacy.load("ner/output/model-last")
```

#### Load Huggingface model

```python
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
from datasets import Dataset, DatasetDict

pretrainedFinancialBERT = BertForSequenceClassification.from_pretrained("ahmedrachid/FinancialBERT-Sentiment-Analysis",num_labels=3)
pretrainedFinancialBERT.to_bettertransformer()
tokenizer = BertTokenizer.from_pretrained("ahmedrachid/FinancialBERT-Sentiment-Analysis")

bert_sent = pipeline("sentiment-analysis", model=pretrainedFinancialBERT, tokenizer=tokenizer)

sentences = ["Operating profit rose to EUR 13.1 mn from EUR 8.7 mn in the corresponding period in 2007 representing 7.7 % of net sales.",  
             "Bids or offers include at least 1,000 shares and the value of the shares must correspond to at least EUR 4,000.", 
             "Raute reported a loss per share of EUR 0.86 for the first half of 2009 , against EPS of EUR 0.74 in the corresponding period of 2008.", 
             ]

results = bert_sent(sentences)
print(results)
```

#### Filter gold and silver commodities related content
Use Spacy model that has been trained before to perform NER

```python
gold_silver_com_sample = []
com_label = 'COM'
for sample in samples:
    doc = ner(sample)
    for ent in doc.ents:
        if ent.label_ == com_label and ent.text.lower() in ['gold', 'silver']:
            # print(sample);
            # print(ent.text, ent.start_char, ent.end_char, ent.label_)
            gold_silver_com_sample.append(sample);
```

#### Evaluate and collect data

```python
import spacy
sent = spacy.load("sentiment_analysis_spacy/output/model-last")

text_length = 1000
data = []
for sample in gold_silver_com_sample:

    doc = sent(sample)
    spacy_label = max(doc.cats.items(), key=lambda x: x[1])[0]
    spacy_label = spacy_label.lower()

    # model's max_seq_length is 512, FinancialBERT is not able to process too long documents
    if len(sample.split()) < 320:
        bert_predicted = bert_sent(sample)
        bert_label = bert_predicted[0]['label'].lower()
        unmatched = 'X' if spacy_label != bert_label else 'O'
    else:
        bert_label = 'n/a'
        unmatched = ''


    data.append([sample[:text_length], spacy_label, bert_label, unmatched])
```

#### Display data

```python
# !pip3 install -U tabulate
import tabulate

headers = ['Sample', 'Spacy', 'FinancialBERT', 'Comparison']
table = tabulate.tabulate(
    data,
    headers=headers,
    tablefmt='simple',
    colalign=('center', 'left','center', 'center', 'center'),
    maxcolwidths=[4, 60, 8, 8, 8], showindex="always"
    )
print(table)
```

## Fine tune FinancialBERT using the same training data that is used for training Spacy TextCategorizer
In model overview we see that FinancialBERT include bert layer, classifier layer. The fine tuning experiment is trying to train classifier layer only, bert layer is kept unchanged during training in order to preserve pretrained performance.

```python
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
from datasets import Dataset, DatasetDict

model = BertForSequenceClassification.from_pretrained("ahmedrachid/FinancialBERT-Sentiment-Analysis",num_labels=3)
tokenizer = BertTokenizer.from_pretrained("ahmedrachid/FinancialBERT-Sentiment-Analysis")
print('Model summarization: \n\n', model)

for param in model.bert.embeddings.parameters():
  param.requires_grad = False

for param in model.bert.encoder.parameters():
  param.requires_grad = False

for param in model.bert.pooler.parameters():
  param.requires_grad = False

```

#### Prepare training data

```python
from pathlib import Path

id2label = model.config.id2label
label2id = model.config.label2id

dataset = []
for item in sent_train_data:
    text = item['data']['text']
    annotated = item['annotations'][0]

    if len(text) < 20:
        continue

    result = annotated['result']
    if len(result) == 0:
        continue;
    label = result[0]['value']['choices'][0]
    label = label.lower()

    dataset.append({
        'text': text,
        'label': label2id[label]
        })

print('number of sample: ', len(dataset))
print(dataset[3])

```

```python
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

dataset = Dataset.from_list(dataset)
tokenized_datasets = dataset.map(tokenize_function, batched=True)
```
#### Training (fine tuning) 
Note: traing_dataset and eval_datasset are the same.

```python
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return metric.compute(predictions=predictions, references=labels)

batch_size = 10
training_args = TrainingArguments(
  output_dir="output_trainer",
  num_train_epochs=120,
  per_device_train_batch_size=batch_size,
  per_device_eval_batch_size=batch_size,
  evaluation_strategy="epoch",
  save_total_limit=2,
  gradient_checkpointing=True,  #  If True, use gradient checkpointing to save memory at the expense of slower backward pass
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,
    compute_metrics=compute_metrics,
)
```

```python
trainer.train()
```

```python
model.save_pretrained('models/FinancialBERT')
```

## Perform Spacy model, FinancialBERT model, fine tuned FinancialBERT model


#### Evaluate and colllect data

```python
import spacy
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
from datasets import Dataset, DatasetDict

# Load Spacy model
sent = spacy.load("sentiment_analysis_spacy/output/model-last")

# Load FinancialBERT
tokenizer = BertTokenizer.from_pretrained("ahmedrachid/FinancialBERT-Sentiment-Analysis")

# Load fine tuned model
pretrainedFinancialBERT = BertForSequenceClassification.from_pretrained("ahmedrachid/FinancialBERT-Sentiment-Analysis",num_labels=3)
pretrainedFinancialBERT.to_bettertransformer()
bert_sent1 = pipeline("sentiment-analysis", model=pretrainedFinancialBERT, tokenizer=tokenizer)

tunedFinancialBERT = BertForSequenceClassification.from_pretrained('models/FinancialBERT')
tunedFinancialBERT.to_bettertransformer()
bert_sent2 = pipeline("sentiment-analysis", model=tunedFinancialBERT, tokenizer=tokenizer)

text_length = 1000
data2 = []
for sample in gold_silver_com_sample:

    doc = sent(sample)
    spacy_label = max(doc.cats.items(), key=lambda x: x[1])[0]
    spacy_label = spacy_label.lower()

    # model's max_seq_length is 512, FinancialBERT is not able to process too long documents
    if len(sample.split()) < 320:
        bert_predicted1 = bert_sent1(sample)
        bert_label1 = bert_predicted1[0]['label'].lower()
        unmatched1 = 'X' if spacy_label != bert_label1 else 'O'

        bert_predicted2 = bert_sent2(sample)
        bert_label2 = bert_predicted2[0]['label'].lower()
        unmatched2 = 'X' if spacy_label != bert_label2 else 'O'
    else:
        bert_label = 'n/a'
        unmatched = ''


    data2.append([sample[:text_length], spacy_label, bert_label1, unmatched1, bert_label2, unmatched2])
```

#### Display data
Manually checking some samples that fine tuned model predicts differently from origin one, found that funed one is slightly better.

```python
# !pip3 install -U tabulate
import tabulate

headers = ['Sample', 'Spacy', 'FinancialBERT', 'vs Spacy', 'tuned FinancialBERT', 'vs Spacy']
table = tabulate.tabulate(
    data2,
    headers=headers,
    tablefmt='simple',
    colalign=('center', 'left','center', 'center', 'center', 'center', 'center'),
    maxcolwidths=[4, 50, 8, 8, 8, 8, 8], showindex="always"
    )
print(table)
```
