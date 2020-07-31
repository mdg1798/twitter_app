
import numpy as np
from flask import Flask, request, jsonify, render_template,redirect,url_for
import preedit
from keras.models import load_model
import postprocess
import tensorflow as tf
from textblob import TextBlob
import get_tweets
import pandas as pd
import os
import numpy as np
global graph
graph = tf.get_default_graph()
model=load_model("trend_prediction_70.h5")
app = Flask(__name__)
#app.secret_key=os.environ.get('secret_key', str(np.random.randint(0, 1000000)))
IMG_FOLDER = os.path.join('static', 'img1')
app.config['UPLOAD_FOLDER'] = IMG_FOLDER

def sentiment(text):
    blob=TextBlob(text)
    return blob.sentiment.polarity

def threshold(p):
    if p>0.1:
        a='positive'
    elif p<=0.1 and p>=-0.1:
        a='neutral'
    else:
        a='negative'
    return a


def predict_results(name):
    with graph.as_default():
            tweets=get_tweets.get_tweets(name)
            clean=preedit.pipeline(tweets,"tweet_text")
            preds=model.predict(clean)
            tweets['Topic']=postprocess.predict(preds)
            tweets['Tweet Polarity']=tweets['tweet_text'].apply(sentiment)
            tweets['Tweet Sentiment']=tweets['Tweet Polarity'].apply(threshold)
            v1=pd.DataFrame([tweets.Topic.value_counts()])
            v2=pd.DataFrame([tweets['Tweet Sentiment'].value_counts()])
    return tweets,v1,v2


@app.route('/')
def home():
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'img.jpg')
    return render_template('index1.html', user_image = full_filename)

@app.route('/',methods=['POST'])

def get_data():
    user = request.form['search']
    return redirect(url_for('success', name=user))

@app.route('/success/<name>')
def success(name):
   # return "<xmp>" + str(predict_results(name)) + " </xmp> "
   return render_template('index.html',name=name, name1='Theme Count', name2='Sentiment Count',data=predict_results(name)[0].to_html(),data2=predict_results(name)[1].to_html(),data3=predict_results(name)[2].to_html())
   # return render_template('index.html',name=name,data=predict_results(name)[0].to_html())
# def predict():
# #     '''
# #     For rendering results on HTML GUI
# #     '''
#     for x in request.form.values():
#         int_features=str(x)
#     with graph.as_default():
#     #final_features = [np.array(int_features)]
#         prediction = model.predict(pre(int_features))

#         output = pe(prediction)
#         sen=sentiment(int_features)
#         sen_1=threshold(sen)

#     return render_template('index.html', prediction_text='Topic is {} and sentiment is {}'.format(output,sen_1))

# @app.route('/predict_api',methods=['POST'])
# def predict_api():
# #     '''
# #     For direct API calls trought request
# #     '''
#     data = request.get_json(force=True)
#     prediction = model.predict([np.array(list(data.values()))])

#     output = prediction[0]
#     return jsonify(output)

# if __name__ == "__main__":
#     app.run(debug=False)

