# DiS-ReX: A Multilingual Dataset for Distantly Supervised Relation Extraction

We release DiS-ReX, a multilingual dataset for distantly supervised relation extraction. The dataset has over 1.5 million instances, spanning 4 languages (English, Spanish, German and French). Our dataset has 36 positive relation types + 1 no relation (NA) class. We release our dataset and make it openly available on this [link](https://zenodo.org/record/4704084#.YH5-MugzZPZ)

# Format
The dataset folder has 5 text files
```
english.txt
german.txt
french.txt
spanish.txt
rel2id.txt
```
For files named `<language>.txt`, each line is a unique instance represented as a Python dictionary. An example is shown below:
```
{"token": ["at", "the", "58th", "annual", "grammy", "awards", "in", "february", "the", "eagles", "joined", "by", "leadon", "touring", "guitarist", "steuart", "smith", "and", "co-writer", "jackson", "browne", "performed", "\"take", "it", "easy\"", "in", "honor", "of", "frey"], "h": {"name": "steuart smith", "id": "Q3498822", "pos": [15, 17]}, "t": {"name": "eagles", "id": "Q2092297", "pos": [9, 10]}, "relation": "http://dbpedia.org/ontology/associatedBand"}
```

Here the keys and values have the following meaning:

1. token: A list representing the context sentence. Every element in the list represents a word.
2. h: A dictionary for head entity. has the following keys:
   -  name: name of the head 
   -  entityid: wikidata id for the entity
   -  pos: a tuple of the form [start index, end index] according to head entity's positition in the token list
3. t: A dictionary for tail entity. has the following keys:
   -  name: name of the tail
   -  entityid: wikidata id for the entity
   -  pos: a tuple of the form [start index, end index] according to tail entity's positition in the token list
4. relation: relation for the tuple (head entity, tail entity)


The dataset format is same as presented in [OpenNRE](https://github.com/thunlp/OpenNRE). For a bag with more than one possible relations, the instances are repeated with a different value for the relation key. An example is shown below:

```
{"token": ["huxley", "who", "had", "twice", "visited", "the", "soviet", "union", "was", "originally", "not", "anti-communist", "but", "the", "ruthless", "adoption", "of", "lysenkoism", "by", "joseph", "stalin", "ended", "his", "tolerant", "attitude"], "h": {"name": "joseph stalin", "id": "Q855", "pos": [19, 21]}, "t": {"name": "the soviet union", "id": "Q15180", "pos": [5, 8]}, "relation": "http://dbpedia.org/ontology/country"}
{"token": ["huxley", "who", "had", "twice", "visited", "the", "soviet", "union", "was", "originally", "not", "anti-communist", "but", "the", "ruthless", "adoption", "of", "lysenkoism", "by", "joseph", "stalin", "ended", "his", "tolerant", "attitude"], "h": {"name": "joseph stalin", "id": "Q855", "pos": [19, 21]}, "t": {"name": "the soviet union", "id": "Q15180", "pos": [5, 8]}, "relation": "http://dbpedia.org/ontology/deathPlace"}
```

The file named `rel2id.txt` contains relation types and the corresponding indices we use during training our model.

# DS-RE Models

We have also uploaded code to reproduce results in the paper "DiS-ReX: A Multilingual Dataset for Distantly Supervised Relation Extraction". It also has the implementation of the 3 baseline models used to obtain results on the dataset.

First, 2 datasets - "RELX-Distant" and "DiSReX" need to be downloade from the following link :-

```
https://drive.google.com/file/d/1yVZIJKeRyuLIfDCwxJb8zxHISw4pcvu8/view?usp=sharing
```

Unzip the file and copy the folders "disrex_dataset" and "relx_distant" into this folder. After copying files in this folder woul be as follows :

```

disrex_dataset/
mBERT_Att/
mnre/
PCNN_Att/
relx_distant/
```

In order to run any of the mBERT+ATT or MNRE+ATT on disrex dataset, go to the respective directory and run the following command :-

```
python main.py


```
In order to run mBERT+Att on relx_distant, go to the respective directory and run the following command :-

```
python main.py --train_file ../relx_distant/relx_train.txt --val_file ../relx_distant/relx_val.txt --test_file ../relx_distant/relx_test.txt --rel2id_file ../relx_distant/relx_rel2id.txt
```

To run PCNN , you first need to download the multilingual glove embedings from this link :--

```

https://drive.google.com/file/d/16KbJCTvTIC6hXEs527uOzJoPBE4cazcI/view?usp=sharing

```

Unzip and copy the folder into the PCNN_Att directory. Then run one of the following commands based on language:

```
     English : python main_multi.py --train_file ../disrex_dataset/english/train.txt --val_file ~/scratch/disrex_dataset/unseen_new/english/val.txt --test_file ~/scratch/disrex_dataset/unseen_new/english/test.txt --bag_size 2 --rel2id_file ~/scratch/disrex_dataset/rel2id.txt --metric auc --max_epoch 60 --ckpt disrex_pcnn_shakuntala_only_english_unseen --embedding_file multilingual_glove/multilingual_embeddings.en --only_test --out_file predictions/pred_out_english.tsv
     Spanish : python main_multi.py --train_file ../disrex_dataset/spanish/train.txt --val_file ../disrex_dataset/spanish/val.txt --test_file ../disrex_dataset/spanish/test.txt --bag_size 2 --rel2id_file ../disrex_dataset/rel2id.txt --metric auc --max_epoch 2 --ckpt pcnn_spanish_baseline --embedding_file multilingual_glove/multilingual_embeddings.es
     French : python main_multi.py --train_file ../disrex_dataset/french/train.txt --val_file ../disrex_dataset/french/val.txt --test_file ../disrex_dataset/french/test.txt --bag_size 2 --rel2id_file ../disrex_dataset/rel2id.txt --metric auc --max_epoch 2 --ckpt pcnn_french_baseline --embedding_file multilingual_glove/multilingual_embeddings.fr
     German : python main_multi.py --train_file ../disrex_dataset/german/train.txt --val_file ../disrex_dataset/german/val.txt --test_file ../disrex_dataset/german/test.txt --bag_size 2 --rel2id_file ../disrex_dataset/rel2id.txt --metric auc --max_epoch 2 --ckpt pcnn_german_baseline --embedding_file multilingual_glove/multilingual_embeddings.de

```
Alternatively, you can run the bash script pcnn_langs_unseen.sh which runs the above 4 commands in sequence.

# Creating the dataset

To build the dataset you first need to download the raw wikipedia texts from wikimedia.dumps.org using command :--
 ```
 wget --no-check-certificate https://dumps/wikimedia.org/eswiki/latest/eswiki-latest-pages-articles.xml.bz2
 ```
Then we need to download the knowledge base and process it into entities file , relations file and kg file. The KG file contains triples of the form (ent1 , rel , ent2) seperated by tabs on the same line. 

To download kg , use command :--
 ```
 wget --no-check-certificate http://downloads.dbpedia.org/3.9/es/mappingbased_properties_es_nt.bz2
  ```

This needs to be decompressed using command :--
 ```
 bzip2 -d mappingbased_properties_es_nt.bz2
 ```
Then we need to process it using the script processKG.py

 ```
 python processKG.py mappingbased_properties_es_nt <ent_file_path> <rel_file_path> <kg_file_path>
  ```

Now we have the KG in proper processed format

Next, we need to process the raw wikipedia texts and convert into sentences .

Run the following 2 commands :--
 ```
 python get_wiki_articles.py eswiki-latest-pages-articles.xml.bz2 spanish_articles.txt
 python getSentences.py spanish_articles spanish_sentences_wiki.txt
 ```
Now we have a file "spanish_sentences_wiki.txt" which has one sentence on each line.

Now running build_dataset.py with appropriate command line arguments, we can get the labelled dataset.Example of a command is :

 ```
 python build_dataset.py --entities_file arabic/ent.txt --relations_file arabic/rel.txt --kg_file arabic/kg.tsv --text_file arabic_sentences.txt  --num_sentences 10000
  ```

# Cite
The dataset is a part of the pre-print [DiS-ReX: A Multilingual Dataset for Distantly Supervised Relation Extraction](https://arxiv.org/abs/2104.08655). We also release our baseline results using mBERT+Bag Attention and present it in our paper. If you use or extend our work, please cite the following paper:
```
@misc{bhartiya2021disrex,
      title={DiS-ReX: A Multilingual Dataset for Distantly Supervised Relation Extraction}, 
      author={Abhyuday Bhartiya and Kartikeya Badola and Mausam},
      year={2021},
      eprint={2104.08655},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
