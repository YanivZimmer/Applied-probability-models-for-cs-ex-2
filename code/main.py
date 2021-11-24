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
