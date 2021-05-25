import sys
import re
def clean(text):
  text = re.sub(r"[.',;:?!()/]", " ", text)
  text = re.sub(r'"'," ",text)
  text = re.sub(r"\s+"," ",text)
  return text
def token_entity(unclean_sentence):
  clean_sentence = []
  s_1=-1
  e_1=-1
  s_2=-1
  e_2=-1
  true_index=-1
  for i,a in enumerate(unclean_sentence):
    if a.startswith('<e1>'):
      s_1=true_index+1
    elif a.startswith('<e2>'):
      s_2=true_index+1
    elif a.startswith('</e1>'):
      e_1=true_index+1
    elif a.startswith('</e2>'):
      e_2=true_index+1
    else:
      true_index+=1
      clean_sentence.append(clean(a))
  return clean_sentence,s_1,s_2,e_1,e_2
name = set()
f=open(sys.argv[1],"r")
lines=f.readlines()
i=0
while(i<len(lines)):
  l=lines[i]
  r=lines[i+1]
  i+=4
  unclean_sentence = l.split('\t')[1][1:-1].split(' ')
  clean_sentence,s_1,s_2,e_1,e_2 = token_entity(unclean_sentence)
  name.add(" ".join(clean_sentence[s_1:e_1]))
  name.add(" ".join(clean_sentence[s_2:e_2]))

f.close()
name_dict = {key: i for i,key in enumerate(name)}
dataset=[]
f = open(sys.argv[1],"r")
lines=f.readlines()
i=0
while(i<len(lines)):
  l=lines[i]
  r=lines[i+1]
  # print(l)
  # print(r)
  i+=4
  unclean_sentence = l.split('\t')[1][1:-1].split(' ')
  clean_sentence,s_1,s_2,e_1,e_2 = token_entity(unclean_sentence)

  # if not r.startswith('no_relation'):
  #   if r.split('(')[1].startswith('e2'):
  #     head={}
  #     nh = " ".join(clean_sentence[s_2:e_2])
  #     head['name']=nh
  #     head['pos'] = [s_2,e_2]
  #     head['id'] = name_dict[head['name']]
  #     tail={}
  #     nt = " ".join(clean_sentence[s_1:e_1])
  #     tail['name']=nt
  #     tail['pos'] = [s_1,e_1]
  #     tail['id'] = name_dict[tail['name']]
  #   elif r.split('(')[1].startswith('e1'):
  #     head={}
  #     nh = " ".join(clean_sentence[s_1:e_1])
  #     head['name']=nh
  #     head['pos'] = [s_1,e_1]
  #     head['id'] = name_dict[head['name']]
  #     tail={}
  #     nt = " ".join(clean_sentence[s_2:e_2])
  #     tail['name']=nt
  #     tail['pos'] = [s_2,e_2]
  #     tail['id'] = name_dict[tail['name']]
  #   else:
  #     print('error')
  #     break
  # else:
  head={}
  nh = " ".join(clean_sentence[s_1:e_1])
  head['name']=nh
  head['pos'] = [s_1,e_1]
  head['id'] = name_dict[head['name']]
  tail={}
  nt = " ".join(clean_sentence[s_2:e_2])
  tail['name']=nt
  tail['pos'] = [s_2,e_2]
  tail['id'] = name_dict[tail['name']]

  final={}
  final['token']=clean_sentence
  final['h']=head
  final['t']=tail
  final['relation']=r.split('\n')[0]
  dataset.append(final)
f.close()

import json
f = open(sys.argv[2],"w",encoding = "utf-8")

for text in dataset:
  str_temp = json.dumps(text,ensure_ascii = False)

  f.write(str_temp + "\n")

f.close()
