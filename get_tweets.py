import GetOldTweets3 as got
import numpy as  np
import pandas as pd

def get_tweets(keyword):
    tweetCriteria = got.manager.TweetCriteria().setQuerySearch(keyword)\
                                               .setMaxTweets(100)\
                                               .setEmoji("unicode")\
                                               .setTopTweets(True)
    tweet_df = pd.DataFrame({'got_criteria':got.manager.TweetManager.getTweets(tweetCriteria)})
    tweet_df["tweet_text"] = tweet_df["got_criteria"].apply(lambda x: x.text)
    tweet_df.drop(["got_criteria"],inplace=True,axis=1)
    return tweet_df