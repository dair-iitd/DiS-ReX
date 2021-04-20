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
