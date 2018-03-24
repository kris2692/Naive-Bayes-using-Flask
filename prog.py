# -*- coding: utf-8 -*-
import os
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug import secure_filename
import math
import nltk
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt


UPLOAD_FOLDER=r"C:\Users\krish\Desktop\New folder (2)\uploads"
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
          
          
training_data=[]
test=[]                                                  
names=[]
people={}
doc_count={}
prior_count={}
words = {}
word_count = {}
length=0
probabilities_train={}
probabilities_test={}
plot_dict=''
plot_dict_1=[]

  
def counts():
  for name in names:
    temp1=[]
    temp2=[]
    temp3=[]
    word_count.setdefault(name,0)
    words.setdefault(name,[])
    prior_count.setdefault(name,0)
    for array in people.get(name):
      temp2.append(array)
      for word in array:
        temp1.append(word)
        words[name].append(word)
      word_count[name]=len(temp1)
      doc_count[name]=len(temp2)
      prior_count[name]=doc_count[name]/(sum(doc_count.values()))
    for array in words.values():
      for word in array:
        temp3.append(word)
    global length
    length=len(list(set(temp3)))
    

def preprocessing(data):
  tokenizer=RegexpTokenizer(r'\w+')
  for line in data:
    tokenized_line=tokenizer.tokenize(line)
    if tokenized_line[0] not in names:
      names.append(tokenized_line[0])
    if(tokenized_line[0] not in people):
      people.setdefault(tokenized_line[0],[])
    people[tokenized_line[0]].append(tokenized_line[1:])

def training():

  for name in names:
    probabilities_train.setdefault(name,{})
    word_prob={}
    for word in list(set(words[name])):
      a=words[name].count(word)+0.2
      b=word_count[name]+(length*0.2)
      
      prob_count=a/b
      word_prob[word]=prob_count
    word_prob[' ']=1/(word_count[name]+(length*0.2))
    probabilities_train[name]=word_prob                      
                    
def testing():
  tokenizer=RegexpTokenizer(r'\w+')
  for name in names:
    probabilities_test.setdefault(name,[])
    for line in test:
      temp=[]
      tokenized_line=tokenizer.tokenize(line)
      for test_word in tokenized_line[1:]:
        if(test_word in probabilities_train.get(name)):
          temp.append(math.log(probabilities_train.get(name)[test_word]))
        else:
          temp.append(math.log(probabilities_train.get(name)[' ']))
      a=(sum(temp)+math.log(prior_count[name]))
      probabilities_test[name].append(a)

@app.route('/results', methods=['POST','GET'])      
def prediction():
  sentence_speaker=''
  predicted_speakers=[]
  for i in range(len(test)):
    highest=-99999
    for k,v in probabilities_test.items():
      if v[i]>=highest:
        highest=v[i]
        sentence_speaker=k
    predicted_speakers.append(sentence_speaker)
    
  total_test_doc_count=0
  correct_prediction_count=0
  for line in test:
    line=line.split(' ')
    if(line[0]==predicted_speakers[total_test_doc_count]):
      correct_prediction_count+=1
    total_test_doc_count+=1
  accuracy=(correct_prediction_count/total_test_doc_count)*100
  info={'accuracy':accuracy,'stats':stats}
  return render_template('results.html',info=info)

def statistics():
  global stats
  stats=[]
  temp=plot_dict.split(' ')
  for word in temp:
    if word not in stopwords.words("english") and word not in names and word not in [',','.','?',':',';','!','\n']:
      plot_dict_1.append(word)
  fd=FreqDist(plot_dict_1)
  temp2=fd.most_common(20)
  for word in temp2:
    stats.append(word[0])



@app.route('/', methods=['POST','GET'])
def user_input():
  if request.method == 'POST':
      f1 = request.files['training-file']
      f1.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f1.filename)))
      f_train=open(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f1.filename)),encoding="utf8",mode="r")
      for line in f_train:
        training_data.append(line)
        global plot_dict
        plot_dict+=''.join(line)
      f_train.close()
      f2 = request.files['test-file']
      f2.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f2.filename)))
      f_test=open(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f2.filename)), encoding="utf8",mode="r")
      for line in f_test:
        test.append(line)
      f_test.close()
      preprocessing(training_data)
      counts()
      training()
      statistics()
      testing()
      return redirect(url_for("prediction"))
  return render_template('input_user.html')      

if __name__=='__main__':
  app.secret_key = 'super secret key'
  app.config['SESSION_TYPE'] = 'filesystem'
  app.run()
  