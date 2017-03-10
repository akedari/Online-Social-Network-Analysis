"""
collect.py
"""
from TwitterAPI import TwitterAPI
import json
import sys
import time

consumer_key = 'YMeXJH8KmiZwTOMpYeSmivWuA'
consumer_secret = 'HxdxrDCzOl3eCu9rDUO20gUGj0DBKETb9qTNB3xV9vclJFxmkP'
access_token = '772473409880072192-Vfz4zl9CxWKJ3zLIVPMBJ5iX9Wfjppy'
access_token_secret = 'afERLrTGOKmlRSBrUqIkgWJkne31YXjdwcHqOz6MQ5SQ3'

def get_twitter():
    """ Construct an instance of TwitterAPI using the tokens you entered above.
    Returns:
      An instance of TwitterAPI.
    """
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)

def extractdata(jsondata):
    sinceid = 12345
    if len(jsondata['text'])>0:
        if not jsondata['text'].startswith('RT') or jsondata['text'].startswith('rt'):
            with open('data/tweets.txt', 'a') as f:
                tweet = json.dumps(str(jsondata['text']))
                f.write(tweet)
                f.write('\n')

            twitter = get_twitter()

            user_follower = []
            user_follower.append(str(jsondata['user']['screen_name']))
            response = robust_request(twitter, 'friends/ids', {'screen_name': jsondata['user']['screen_name'], 'count': '100'})
            friends_ids = [friends_id for friends_id in response]
            user_follower.append(friends_ids)

            with open('data/followers.txt', 'a') as f:
                followers = json.dumps(user_follower)
                f.write(followers)
                f.write('\n')

    sinceid = jsondata['id']
    return sinceid

def robust_request(twitter, resource, params, max_tries=5):
    """ If a Twitter request fails, sleep for 15 minutes.
    Do this at most max_tries times before quitting.
    Args:
      twitter .... A TwitterAPI object.
      resource ... A resource string to request; e.g., "friends/ids"
      params ..... A parameter dict for the request, e.g., to specify
                   parameters like screen_name or count.
      max_tries .. The maximum number of tries to attempt.
    Returns:
      A TwitterResponse object, or None if failed.
    """
    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)

def getdata(twitter):
    tweets = []
    sinceid = 1234
    counter = 0
    flag = True
    while flag:
        r = twitter.request('search/tweets', {'q': 'BMW', 'lang': 'en', 'count': 100, 'since_id':sinceid})
        if r.status_code != 200:  # error
            print("Error")
        else:
            for item in r.get_iterator():
                counter =counter +1
                sinceid = extractdata(item)
                if counter > 1000:
                    flag= False

def main():
    twitter = get_twitter()
    getdata(twitter)

if __name__ == '__main__':
    main()
