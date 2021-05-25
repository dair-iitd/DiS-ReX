import re
import sys

input_file = open(sys.argv[1],"r",encoding = "utf-8")
out_file = open(sys.argv[2],"w",encoding = "utf-8")


s = input_file.readline()
count = 0 
while s:
    s = re.sub(r"[,=:;'*]"," ",s)
    s = re.sub(r"'"," ",s)
    s = re.sub("\s+"," ",s)
    sents = re.split(r"(?<![0-9])[.]",s)
    for sent in sents:
    
        out_file.write(sent + "\n")
        count += 1
        if(count % 500000 == 0):
            print(count)
    
    s = input_file.readline()
    
input_file.close()
out_file.close()
