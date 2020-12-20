def open_to_list(tang_or_song, number):
  import json
  doc = 'data/json/poet.'+ tang_or_song +'.'+str(number)+'.json'
  with open(doc, 'r') as f:
    data = json.load(f)
    f.close() 
  data_look = []
  for i, poet in enumerate(data):
    poet_temp=[]
    for j, sentence in enumerate(poet['paragraphs']):
      poet_temp.append(sentence) 
    data_look.append(poet_temp)
  return data_look

def five_word_filter(tang_or_song,number):
  import json
  import re
  doc = 'data/json/poet.'+ tang_or_song +'.'+str(number)+'.json'
  with open(doc, 'r',encoding='UTF-8') as f:
    data = json.load(f)
    f.close() 
  data_look = []
  for i, poet in enumerate(data):
    is_five = 0
    temp_poet=[]
    for j, sentence in enumerate(poet['paragraphs']):
      temp_poet.append(sentence)
      if re.match('[\u4e00-\u9fa5]{5}，[\u4e00-\u9fa5]{5}。',sentence) is  None: 
        is_five += 1
    if is_five == 0 and len(temp_poet) >= 1:
      data_look.append(temp_poet)
  return data_look

def seven_word_filter(tang_or_song, number):
  import json
  import re
  doc = 'data/json/poet.'+ tang_or_song +'.'+str(number)+'.json'
  with open(doc, 'r',encoding='UTF-8') as f:
    data = json.load(f)
    f.close() 
  data_look = []
  for i, poet in enumerate(data):
    is_five = 0
    temp_poet=[]
    for j, sentence in enumerate(poet['paragraphs']):
      temp_poet.append(sentence)
      if re.match('[\u4e00-\u9fa5]{7}，[\u4e00-\u9fa5]{7}。',sentence) is  None: 
        is_five += 1
    if is_five == 0 and len(temp_poet) >= 1:
      data_look.append(temp_poet)
  return data_look


five_word_song = []
for i in range(0,254000,1000):
  five_word_song = five_word_song + five_word_filter('song',i)
  
seven_word_song = []
for i in range(0,254000,1000):
  seven_word_song = seven_word_song + seven_word_filter('song',i)
  
five_word_tang = []
for i in range(0,57000,1000):
  five_word_tang = five_word_tang + five_word_filter('tang',i)
  
seven_word_tang = []
for i in range(0,57000,1000):
  seven_word_tang = seven_word_tang + seven_word_filter('tang',i)
  
five_word = five_word_song + five_word_tang
seven_word = seven_word_song + seven_word_tang  


import pickle
file=open("data/five","wb")
pickle.dump(five_word,file) 
file.close()  
  
import pickle
file=open("data/seven","wb")
pickle.dump(seven_word,file) #保存list到文件
file.close() 
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  