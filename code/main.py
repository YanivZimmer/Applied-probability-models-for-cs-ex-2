Output1="develop.txt"
Output2="test.txt"
Output3="INPUT_WORD"
Output4="output.txt"
Output5=300000 
Output6=3.333333E-6
def events_in_file(filename):
    events=[]
    with open(filename) as f:
        lines=f.read().split('\n')[::2]
        content_lines=lines[1::2]
        for content in content_lines:
            events= events+content.split(" ")
    return events
events_development_file=events_in_file("../dataset/{0}".format(Output1))
Output7=len(events_development_file)
def split_train_validation(events,split_rate):
    index=round(split_rate*len(events))
    train=events[:index+1]
    validation=events[index+1:]
    return train,validation
train_dev,val_dev=split_train_validation(events_development_file,0.9)
Output8=len(val_dev)
Output9=len(train_dev)
def count_unique_events(events):
    vocab=set(events)
    return len(vocab)
Output10=count_unique_events(train_dev)
def count_event_in_events(event,events):
    return events.count(event)
Output11=count_event_in_events(Output3,train_dev)
#MLE=occurrences of w in events/number of events
Output12=Output11/Output9
Output13=0
#lindMLE=(occurrences of w in events+lamda)/(number of event+lamda*|vocab|)
def lind_mle(lamda, w_in_events,num_of_events,vocab_size):
    Numerator=w_in_events+lamda
    Denominator=num_of_events+lamda*vocab_size
    return Numerator/Denominator
Output14=lind_mle(lamda=0.1,w_in_events=Output11,num_of_events=Output9,vocab_size=Output10)
Output15=lind_mle(lamda=0.1,w_in_events=0,num_of_events=Output9,vocab_size=Output10)
#calc prod and lind for dataset
from collections import Counter
def calc_prob_unigram(data):
    prob={}
    data_len=len(data)
    counter=Counter(data)
    for item in counter.items:
        prob[item[0]]=item[1]/data_len
    return prob
def calc_prob_lind(data,lamda):
    prob={}
    data_len=len(data)
    counter=Counter(data)
    vocab_size=count_unique_events(data)
    for item in counter.items():
        lind=lind_mle(lamda=lamda,w_in_events=item[1],num_of_events=data_len,vocab_size=vocab_size)
        prob[item[0]]=lind
    return prob
#calc preplexity and helper methods
import math
#TODO- ask what log base should be used?
def calc_log_event(event,prob_dict,prob_unseen):
    if event not in prob_dict.keys():
        return prob_unseen
    return math.log2(prob_dict[event])

def calc_log_sum(data,prob_dict,prob_unseen):
    sum=0
    for event in data:
        sum=sum+calc_log_event(event=event,prob_dict=prob_dict,prob_unseen=prob_unseen)     
    return sum

def calc_preplexity(data,prob_dict,prob_unseen):
    power=-1*calc_log_sum(data=data,prob_dict=prob_dict,prob_unseen=prob_unseen)/(len(data))
    base=2
    result=math.pow(base,power)
    return result

#calc preplexity for validation set
def preplexity(valid_data,train_data,lamda):
    prob_dict=calc_prob_lind(data=train_data,lamda=lamda)
    prob_unseen=lind_mle(lamda=lamda,w_in_events=0,num_of_events=len(train_data)
    ,vocab_size=count_unique_events(train_data))
    prepl=calc_preplexity(data=valid_data,prob_dict=prob_dict,prob_unseen=prob_unseen)
    return prepl
#NOT TESTED YET!
Output16=preplexity(valid_data=val_dev,train_data=train_dev,lamda=0.01)
Output17=preplexity(valid_data=val_dev,train_data=train_dev,lamda=0.1)
Output18=preplexity(valid_data=val_dev,train_data=train_dev,lamda=1.0)
print(Output16,Output17,Output18)