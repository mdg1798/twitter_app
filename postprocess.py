"""[summary]
this module provide preprocessing functions for analyzing tweet
"""
import pandas as pd
import numpy as np


def post_edit(pred):
    y_argmax=np.argmax(pred)
    topic=''
    if y_argmax==0:
        topic="Digital and Fintech"
    elif y_argmax==1:
        topic="Client Experience"
    elif y_argmax==2:
        topic="Cryptocurrency"
    elif y_argmax==3:
        topic="Investment"
    elif y_argmax==4:
        topic="Insurance"
    elif y_argmax==5:
        topic="Advisors"
    else: 
        topic="Stocks and Trading"
    return topic

def predict(ar):
    t=[]
    for i in ar:
        t.append(post_edit(i))
    return t