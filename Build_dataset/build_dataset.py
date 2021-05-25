import spacy
import pickle
import re
import argparse
import nltk
from nltk.corpus import stopwords
import nltk
import ssl
import json
import os
from camel_tools.ner import NERecognizer

from camel_tools.utils.dediac import dediac_ar
import threading
from camel_tools.tokenizers.word import simple_word_tokenize
import time

sources_tagged_arabic = []
def getEntities(sources,lang = "english"):
    
    if(lang == "english"):
        nlp = spacy.load('en_core_web_md') ## Change this according to the language
    elif(lang == "french"):
        nlp = spacy.load("fr_core_news_md")
    elif(lang == "dutch"):
        nlp = spacy.load("de_core_new_md")
    elif(lang == "spanish"):
        nlp = spacy.load("es_core_news_md")
    elif(lang == "greek" ):
        nlp = spacy.load("el_core_news_md")
    elif(lang == "chinese"):
        nlp = spacy.load("zh_core_news_md")
    elif(lang == "italian"):
        nlp = spacy.load("it_core_news_md")
    elif(lang == "greek"):
        nlp = spacy.load("el_core_news_md")
    sources_tagged = []
    count = 0
    print("Starting with Named Entity Tagging...")
    for sentence in sources:
        doc = nlp(sentence)
        temp_dict = {}
        temp_dict["text"] = sentence

        temp_dict["entities"] = []
        for ent in doc.ents:
            temp_dict["entities"].append((ent.text, ent.start_char, ent.end_char))

        sources_tagged.append(temp_dict)
        count += 1
        if (count % 500 == 0):
            print("Done with {}".format(count))

    return sources_tagged

def getEntitiesArabic(index,sources):
    ner = NERecognizer.pretrained()
    sources_tagged = []
    count = 0
    print("Starting with Named Entity Tagging...")
    for sentence in sources:
        tokens = simple_word_tokenize(sentence)
        if(len(tokens) > 150):
            continue
        try:
            labels = ner.predict_sentence(tokens)
        except:
            continue
        entities = []
        temp = ""
        foundEnt = False
        labels.reverse()
        tokens.reverse()        

        for i in range(len(labels)):
            if(foundEnt):
                if(labels[i] == 'I-LOC' or labels[i] == 'B-LOC'):
                    temp.append(tokens[i])
                else:
                    temp = temp[::-1]
                    ent = dediac_ar(" ".join(temp))
                    entities.append(ent)
                    #print(ent)
                    
                    temp = ""
                    foundEnt = False
            else:
                if(labels[i] == 'I-LOC' or labels[i] == 'B-LOC'):
                    temp = [tokens[i]]
                    foundEnt = True
         
        temp_dict = {}
        temp_dict["text"] = sentence

        temp_dict["entities"] = []
        for ent in entities:
            temp_dict["entities"].append((ent, -1, -1))

        sources_tagged.append(temp_dict)
        count += 1
        if (count % 500 == 0):
            print("Done with {}".format(count))

    chunkMappingList[index] = sources_tagged 
    return 

def getKGEntities(filename = "fr.tsv"):
    file_entites = open(filename)  ## Change based on language

    text = file_entites.read()
    dbpedia_entities = text.split("\n")

    kg_entities = []

    for ent in dbpedia_entities:
        ent2 = ent.split("/")[-1]
        ent2 = re.sub("[_-]", " ", ent2)
        ent2 = re.sub(r"\(.+\)", "", ent2).strip()
        kg_entities.append(ent2)

    return kg_entities

def PreprocessKGEntities(kg_entities):

    match_dict = {}
    count_error = 0
    for i, ent2 in enumerate(kg_entities):
        try:
            ent = ent2.lower()
            head2 = ent.split()
            head = [x for x in head2 if x not in stop_set]
            #print(head)
            ent = " ".join(head)
            key = head[0]
            if key in match_dict.keys():
                match_dict[key].append((ent, i))
            else:
                match_dict[key] = [(ent, i)]
        except:
            count_error += 1
            continue

    return match_dict

def matchString(r, s):
    temp1 = r.split()
    tokens2 = s.split()

    tokens1 = []
    for token in temp1:
        if (token in stop_set):
            continue
        else:
            tokens1.append(token)
    
    r = " ".join(tokens1)
    return (r == s)

    if (len(tokens1) < 2):
        return (r == s)

    if (len(tokens1) > len(tokens2)):
        return False

    for i in range(len(tokens1)):
        try:
            if (tokens1[i] != tokens2[i]):
                return False
        except:
            print(tokens1, tokens2)

    return True

## Aligns entities present in the knowledge graph with entities detected in sentences by NER tagger
def AlignEntities(sources_tagged,match_dict):
    count = 0
    count2 = 0
    for i in range(len(sources_tagged)):
        if (i % 5000 == 0):
            print("Done with ", i)
        inst = sources_tagged[i]
        try:
            sources_tagged[i]["kg_entities"] = []
        except:
            print(sources_tagged[i])
        for j, ent in enumerate(inst["entities"]):
            ent_text = ent[0].lower()
            head2 = ent_text.split()
            head = [x for x in head2 if x not in stop_set]
            # print(head)
            ent_text = " ".join(head)
            flag = True
            head = ent_text.split()
            if (len(head) > 0):
                if head[0] in match_dict.keys():
                    for kg_ent in match_dict[head[0]]:
                        if (matchString(kg_ent[0], ent_text)):
                            sources_tagged[i]["kg_entities"].append(kg_ent)
                            flag = False
                            break
            if flag:
                sources_tagged[i]["kg_entities"].append(("NONE", -1))
                count2 += 1
            else:
                count += 1

    print("Found matches for {} entities".format(count))
    print("No matches for {} entities".format(count2))

    return sources_tagged

#Parses file containing triples and returns dictionary in the form of adjacency list.
#Each edge is denoted by (entity,type of relation) tuple
def getKG(filename):
    graph_file = open(filename, "r")

    kg = {}

    s = graph_file.readline()

    while s:
        try:
            tokens = s.split("\t")
            head = int(tokens[0])
            relation = tokens[1]
            tail = int(tokens[2])
            if head in kg.keys():
                kg[head].append((tail, relation))
            else:
                kg[head] = [(tail, relation)]
        except:
            print("ERROR")
        s = graph_file.readline()
    return kg


def getEntityPos(text_tokens, head_tokens):
    pos1 = -1
    pos2 = -1
    for i, t1 in enumerate(text_tokens):
        try:
            if (head_tokens[0] == t1):
                flag = True
                for j, t2 in enumerate(head_tokens):
                    if (t2 != text_tokens[i + j]):
                       flag = False
                       break
                if (flag):
                    pos1 = i
                    break
        except:
           print("ERROR2 !!")
           print(text_tokens)
           print(head_tokens)
           return [-1,1] 
    # if(pos1 < 0):
    #     #print("ERROR for head string : ", text)
    pos2 = pos1 + len(head_tokens)

    return [pos1, pos2]

def alignRelations(sources_tagged,kg,relations_file):
    count = 0
    relation_file = open(relations_file)

    rel_dict = {}

    rels = relation_file.read().split("\n")
    for i, rel in enumerate(rels):
        rel_dict[i] = rel
    final_texts = []
    final_texts_na = []
    count2 = 0
    for i in range(len(sources_tagged)):
        # print("TEST")
        if (i % 5000 == 0):
            print("Done with ", i)
        inst = sources_tagged[i]
        if(len(inst["text"].split()) > 150):
            continue
        sources_tagged[i]["relation"] = []
        for j, ent in enumerate(inst["kg_entities"]):
            if (ent[1] < 0):
                continue
            else:
                for k, ent2 in enumerate(inst["kg_entities"]):
                    if (j == k):
                        continue
                    if (ent2[1] < 0):
                        continue
                    else:
                        if (ent[1] in kg.keys()):
                            isrel = False
                            for edge in kg[ent[1]]:
                                if (edge[0] == (ent2[1])):
                                    dict_temp = {}
                                    dict_temp["token"] = inst["text"].lower()
                                    dict_temp["token"] = re.sub(r"[(]", r" -lrb- ", dict_temp["token"])
                                    dict_temp["token"] = re.sub(r"[)]",r" -rrb- ",dict_temp["token"])
                                    dict_temp["token"] = dict_temp["token"].split()
                                    head = inst["entities"][j][0]
                                    dict_temp["h"] = {}
                                    dict_temp["h"]["name"] = head.lower()
                                    dict_temp["h"]["id"] = ent[1]
                                    dict_temp["h"]["pos"] = getEntityPos(dict_temp["token"],
                                                                         dict_temp["h"]["name"].split())
                                    # print(dict_temp["h"]["pos"])
                                    head = inst["entities"][k][0]
                                    dict_temp["t"] = {}
                                    dict_temp["t"]["name"] = head.lower()
                                    dict_temp["t"]["id"] = ent2[1]
                                    dict_temp["t"]["pos"] = getEntityPos(dict_temp["token"],
                                                                         dict_temp["t"]["name"].split())

                                    # print(dict_temp["h"]["name"],dict_temp["t"]["name"])
                                    dict_temp["relation"] = rel_dict[int(edge[1])]

                                    count += 1
                                    final_texts.append(dict_temp)
                                    isrel = True
                                    break
                            
                            if not isrel:
                                dict_temp = {}
                                dict_temp["token"] = inst["text"].lower()
                                dict_temp["token"] = re.sub(r"[(]", r" -lrb- ", dict_temp["token"])
                                dict_temp["token"] = re.sub(r"[)]",r" -rrb- ",dict_temp["token"])
                                dict_temp["token"] = dict_temp["token"].split()
                                head = inst["entities"][j][0]
                                dict_temp["h"] = {}
                                dict_temp["h"]["name"] = head.lower()
                                dict_temp["h"]["id"] = ent[1]
                                dict_temp["h"]["pos"] = getEntityPos(dict_temp["token"],
                                                                         dict_temp["h"]["name"].split())
                                    # print(dict_temp["h"]["pos"])
                                head = inst["entities"][k][0]
                                dict_temp["t"] = {}
                                dict_temp["t"]["name"] = head.lower()
                                dict_temp["t"]["id"] = ent2[1]
                                dict_temp["t"]["pos"] = getEntityPos(dict_temp["token"],
                                                                         dict_temp["t"]["name"].split())
                                dict_temp["relation"] = "NA"
                                count2 += 1

                                
                                final_texts_na.append(dict_temp)
    print(count)
    print(count2)
    print("Found {} sentences".format(count))
    final_texts_na = final_texts_na[:3*count]
    final_texts.extend(final_texts_na)
    return final_texts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CODE FOR: Distant Supervision Relation Extraction with Intra-Bag and Inter-Bag Attentions')
    parser.add_argument('--text_file', default='text/train.en.tok', help='path to pre-training file')
    parser.add_argument('--entities_file', default='DBP-5L/entity_lists/en.tsv', help='path to training file')
    parser.add_argument('--kg_file', default='DBP-5L/kgs/en_complete.txt', help='path to test file')
    parser.add_argument('--relations_file', default='DBP-5L/relations.txt', help='path to test-one file')
    parser.add_argument('--num_sentences', default=-1, help='path to test-one file')
    parser.add_argument('--lang', default='english',choices=['italian','english','french','spanish','dutch','chinese','greek','arabic'], help='path to test-one file')
    parser.add_argument('--resume',action = 'store_true',help='path to test-one file')
    parser.add_argument('--start',default = 0,help='path to test-one file')
    parser.add_argument('--end',default = -1,help='path to test-one file')
    parser.add_argument('--num_threads',default = 1,help = 'number of threads to run in parallel')
    args = parser.parse_args()
    
   

    try:
       _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
       pass
    else:
       ssl._create_default_https_context = _create_unverified_https_context
    
    global stop_set
    if(args.lang == 'chinese'):
       words = set()
       files = os.listdir("stopwords-zh")
       for file in files:
          temp = os.path.join("stopwords-zh",file)
          instances = open(temp,"r").read().split("\n")
          for instance in instances:
             words.add(instance)
       stop_set = words
    else:   
       nltk.download("stopwords")
       stop_set = set(stopwords.words(args.lang))
     
    if(args.resume):
       outfile_en = open("targets_" + args.lang + "_wiki.pkl","rb")
       sources_tagged = pickle.load(outfile_en)
       print(sources_tagged[0])
    else:
       	
       file = open(args.text_file,"r",encoding = "utf-8") ##Change later based on name of input file.
       if(int(args.num_sentences) > 0):
          sources = []
          s = file.readline()
          while int(args.num_sentences) > len(sources):
             sources.append(s)
             s = file.readline()
       else:
          sources = file.read().split("\n") 
       print(sources[0]) 
       #sources = sources[:10000]
       if(args.lang == 'arabic'):
           
           num_threads = int(args.num_threads)
           batch_size = (len(sources) // num_threads) + 1
           threads = list()
           batches = []
           chunkMappingList = [[]]*num_threads
           start = time.time()
           for i in range(num_threads):
               start = i*batch_size
               end = min((i+1)*batch_size,len(sources))
               sources_part = sources[start:end]
               t = threading.Thread(target= getEntitiesArabic,args = (i,sources_part,) , name= "thread "+str(i))
               threads.append(t)
               t.start()
           
           for t in threads:
               t.join()
           
           sources_tagged = []
           for source in chunkMappingList:
               sources_tagged.extend(source) 
           
           print("Time taken for labelling entities is {} seconds".format(time.time() - start))
           
       else:
           sources_tagged = getEntities(sources,lang = args.lang)
       print(sources_tagged[0])
       outfile_en = open("targets_" + args.lang + "_wiki.pkl","wb")
       pickle.dump(sources_tagged,outfile_en)
       outfile_en.close()
    kg_entities = getKGEntities(args.entities_file)
    kg_entities = PreprocessKGEntities(kg_entities)
    sources_tagged = AlignEntities(sources_tagged,kg_entities)
    print(sources_tagged[0])
    kg = getKG(args.kg_file)     
    final_texts = alignRelations(sources_tagged,kg,args.relations_file)
    print(final_texts[0])
    
    file_final = open("final_texts_" + args.lang + "_wiki.txt","w",encoding = "utf-8")
    for text in final_texts:
      str_temp = json.dumps(text,ensure_ascii = False)

      file_final.write(str_temp + "\n")

    file_final.close()

