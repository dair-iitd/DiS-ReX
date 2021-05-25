import sys
import re

def processEntity(ent):
  
  text = ""
  tokens = ent.split("/")
  text2 = "/".join(tokens[:-1]) + "/"
  ent = tokens[-1]
  length = len(ent)
  count = 0 
  while count < length:
    if(count <  (length - 1) and ent[count:count + 2] == r"\u"):
       character = chr(int(ent[count + 2:count + 6],16))
       text += character
       count += 6
    else:
       text += ent[count]
       count += 1
  text = text2 + text
  return text
       
   	


file = open(sys.argv[1],"r")
s = file.readline()

triples = []

dict_entities = {}

dict_rel = {}

kg = []
count  = 0
count2 = 0 
count3 = 0 
while s:
    count3 += 1
    try:
      s = re.sub("%[0-9]+","",s)
      s = re.sub("[<>]"," ",s)
      tokens = s.strip().split()
      head = tokens[0]
      rel = tokens[1]
      tail = tokens[2]
    except:
      count += 1
      s = file.readline()
      continue
    #print(rel)
    if not (head.startswith("http://ar.dbpedia.org") and tail.startswith("http://ar.dbpedia.org") and rel.startswith("http://dbpedia.org")):
        count += 1
        s = file.readline()
        continue
    #print("TEST")
    if not head in  dict_entities.keys():
        dict_entities[head] = len(dict_entities)
    if not tail in  dict_entities.keys():
        dict_entities[tail] = len(dict_entities)
    
    rel = re.sub("http://ar.dbpedia.org","http://dbpedia.org",rel)
    if not rel in dict_rel.keys():
        dict_rel[rel] = len(dict_rel)

    head_id = dict_entities[head]
    tail_id = dict_entities[tail]
    rel_id = dict_rel[rel]

    kg.append((head_id,rel_id,tail_id))

    s = file.readline()
    count2 += 1
    if(count2 % 50000 == 0):
    	print(count2,count3)

print("Found {} triples and ignores {} triples".format(count,count2))
file_ent = open(sys.argv[2],"w")
file_rel= open(sys.argv[3],"w")

temp = []
for k in dict_entities.keys():
    str_temp = processEntity(k)
    k2 = re.sub(r"\\u",r"\u",k)
    temp.append(k)
    file_ent.write(str_temp + "\n")

print(temp[:10])
    
    
file_ent.close()

for k in dict_rel.keys():
    file_rel.write(processEntity(str(k)) + "\n")

file_rel.close()

file_kg = open(sys.argv[4],"w")

for a,b,c in kg:
    file_kg.write(str(a) + "\t" + str(b) + "\t" + str(c) + "\n")
    
file_kg.close()






