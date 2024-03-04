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

```python
#!pip3 install -U spacy
#!python -m spacy download en_core_web_md
```

```python
import json

with open('ner/ner_train.json', mode='r', encoding='utf-8') as fd:
    ner_train_data = json.loads(fd.read());

print(ner_train_data[0])
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
```python
text = "Gold Rate Today, 30 April 2021: Gold, Silver fall, know – what are the 10 grams gold rate today https://t.co/kSVOTVuXw5"

```

```python
doc = nlp(text)
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
```

```python
text = "Tbvh, Adekunle Gold really did magic on this song he issued out titled #IIWII .. Man always stepping up his game.. https://t.co/LEDEq0AdF6"
doc = nlp(text)
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
```

```python
text = "Did and Dig!. Gold is one of the most desired and useful metals in the world. Not only could it be beautifully shaped and sculpted, the precious yellow metal conducts electricity and does not tarnish. This article could possibly give you more knowledge about the uses of the precious metal gold. Check it here!   These qualities could make it the metal of choice for the industrial, medical and technology businesses. Arguably no other metal has been said to have a record throughout history, with almost every established culture using gold to symbolise power, beauty, purity and accomplishment.    Today, gold still seems to occupy an important place important place in our culture and society people could posibly use it to make our one of the most prized objects: wedding rings, Olympic medals, money, jewellery, Oscars, Grammys, crucifixes, art and many more. This is a sponsored post. Check disclaimer on profile and landing page"
doc = nlp(text)
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
```

```python
text = "Gold Forecast: Hovering Between Moving Averages. Pay close attention to the rate of change in the yield of bonds, because if it is a slow and gradual rise, then I believe that gold should do fairly well. Gold markets initially tried to rally during the course of the trading session on Thursday but failed to continue going higher at the 200 day EMA as it continues to be quite resistive. At this point, the market then felt to reach down below the 50 day EMA, which of course was a very negative sign. However, by the end of the session we turned around to break above the 50 day EMA and now it looks like the market is essentially doing the same thing it has been doing for several days. These are two major averages that a lot of people pay attention to, so it should not be overly surprising that we are stuck in this general vicinity. Furthermore, you have to pay close attention to the US dollar, because the gold market is highly influenced by it. Now that that we have broken down below the 50 day EMA, it looks like the market is finding value hunters underneath just above the $1750 level. I think what we are looking at here is a market that is trying to figure out where we are getting ready to go longer term. When you look at the charts you can see that we have formed a double bottom just below the $1700 level, so I think if we break down below the $1750 level, then we will go testing that area. However, the $1750 level above had been resisted previously, broken out above, and now has been retested. At this point, we have to question whether or not this attempt to form a basing pattern has started to stick? A break above the $1800 level opens up the possibility of a much bigger move, and therefore I think that is the key for buying this market. Until then, I would be hesitant to put a lot of money into the gold market, but it certainly looks as if we are trying to turn things around. It could be very noisy over the next couple of weeks, but I do have my eye on this market as a confirmation of the return could very well send this market looking towards the highs again. Pay close attention to the rate of change in the yield of bonds, because if it is a slow and gradual rise, then I believe that gold should do fairly well"
doc = nlp(text)
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
```
```python
import json

with open('sentiment_analysis_spacy/sentiment_train.json', mode='r', encoding='utf-8') as fd:
    sent_train_data = json.loads(fd.read());

print(sent_train_data[5])
```
```python
from pathlib import Path
import spacy
from spacy.tokens import DocBin

def read_categories():
    return Path('sentiment_analysis_spacy/categories.txt').open().read().strip().split("\n")

categories = read_categories();
print(categories)
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

    print(doc)
    db.add(doc)

db.to_disk("./sentiment_analysis_spacy/train.spacy")
```

```python
# number of row has been used for training
padding = 231
with open('sample.txt', mode='r', encoding='utf-8') as fd:
    samples = fd.readlines()
samples = samples[padding:]
```

```python
import spacy
ner = spacy.load("ner/output/model-last")
# for doc in nlp.pipe(texts, disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"]):
#   print([(ent.text, ent.label_) for ent in doc.ents])
```

```python
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
from datasets import Dataset, DatasetDict

model = BertForSequenceClassification.from_pretrained("ahmedrachid/FinancialBERT-Sentiment-Analysis",num_labels=3)
tokenizer = BertTokenizer.from_pretrained("ahmedrachid/FinancialBERT-Sentiment-Analysis")

bert_sent = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

sentences = ["Operating profit rose to EUR 13.1 mn from EUR 8.7 mn in the corresponding period in 2007 representing 7.7 % of net sales.",  
             "Bids or offers include at least 1,000 shares and the value of the shares must correspond to at least EUR 4,000.", 
             "Raute reported a loss per share of EUR 0.86 for the first half of 2009 , against EPS of EUR 0.74 in the corresponding period of 2008.", 
             ]

#results = bert_sent(sentences)
#print(results)
```

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

```python
import spacy
sent = spacy.load("sentiment_analysis_spacy/output/model-last")

text_length = 1000
headers = ['Sample', 'label by Spacy', 'label by FinancialBERT', 'Comparison']
data = []
for sample in gold_silver_com_sample:

    doc = sent(sample)
    spacy_label = max(doc.cats.items(), key=lambda x: x[1])[0]
    spacy_label = spacy_label.lower()

    # model's max_seq_length is 512, FinancialBERT is not able to process too long documents
    if len(sample.split()) < 320:
        bert_predicted = bert_sent(sample)
        bert_label = bert_predicted[0]['label'].lower()
        unmatched = 'X' if spacy_label != bert_label else ''
    else:
        bert_label = 'n/a'
        unmatched = ''


    data.append([sample[:text_length], spacy_label, bert_label, unmatched])
```

```python
# !pip3 install -U tabulate
import tabulate
table = tabulate.tabulate(
    data,
    headers=headers,
    tablefmt='simple',
    colalign=('center', 'left','center', 'center'),
    maxcolwidths=[4, 65, 8, 8], showindex="always"
    )
print(table)
```

```python
# !pip3 install -U 'transformers[torch]'
# !pip3 install -U dataset
# !pip3 install -U numpy
# !pip3 install -U evaluate
# !pip install -U scikit-learn
```

```python
model
```

```python
for param in model.bert.embeddings.parameters():
  param.requires_grad = False
```

```python
for param in model.bert.encoder.parameters():
  param.requires_grad = False
```

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

print(len(dataset))
print(dataset[3])

```

```python
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

dataset = Dataset.from_list(dataset)
tokenized_datasets = dataset.map(tokenize_function, batched=True)
```
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
  num_train_epochs=20,
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

```
