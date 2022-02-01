from pathlib import Path
from collections import defaultdict
from itertools import combinations
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import re
import pandas as pd
import praw
from praw.models import MoreComments
import nltk
nltk.download('vader_lexicon')


class SentimentModel:
    def __init__(self, games, teams, rosters, reddit, sa_type='vader'):
        self.sid = SentimentIntensityAnalyzer()
        self.sa_type = sa_type
        self.games = games
        self.teams = [x.lower() for x in teams['TEAM']]
        self.rosters = rosters
        self.rosters['PLAYER'] = self.rosters['PLAYER'].str.lower()
        self.rosters['TEAM'] = self.rosters['TEAM'].str.lower()
        self.reddit = reddit

    #
    def analysis(self):
        '''
        Run sentiment analysis by preprocessing the data, appending polarity scores, and filtering posts
        Params: None
        Returns: None
        '''
        # csv file exists
        if Path('data/posts.csv').is_file():
            df = pd.read_csv('data/posts.csv')

        # Read in top posts from reddit and generate dataframe to save as a csv if it does not exist
        else:
            print('Generating Dataframe ...')
            ids = [x for x in self.games['ID']]
            data = defaultdict(list)
            i = 0
            for game in ids:
                submission = self.reddit.submission(id=game)
                for top_level_comment in submission.comments[1:]:
                    if isinstance(top_level_comment, MoreComments):
                        continue
                    data[i].append(top_level_comment.body)
                    data[i].append(game)
                    i += 1

            df = pd.DataFrame.from_dict(
                data, orient='index', columns=['Body', 'ID'])
            df['TEAM A'] = df['ID'].map(self.games.set_index('ID')['TEAM A'])
            df['TEAM B'] = df['ID'].map(self.games.set_index('ID')['TEAM B'])
            df.to_csv('data\posts.csv', index=False)

        df = self.preprocess_data(df)
        self.df = self.append_polarity_scores(df)
        self.filtered = self.filter_posts()

    def preprocess_data(self, posts):
        '''
        Clean the data
        Params: Dataframe containing reddit posts
        Returns: Dataframe containing preprocessed reddit posts
        '''
        posts['Body'] = posts['Body'].str.lower()
        posts = posts.drop(posts[posts.Body == '[deleted]'].index)
        posts['Body'] = posts['Body'].apply(lambda x: re.sub(r'\B#\S+', '', x))
        posts['Body'] = posts['Body'].apply(
            lambda x: re.sub(r'http\S+', '', x))
        posts['Body'] = posts['Body'].apply(
            lambda x: ' '.join(re.findall(r'\w+', x)))
        posts['Body'] = posts['Body'].apply(
            lambda x: re.sub(r'\s+', ' ', x, flags=re.I))
        posts['Body'] = posts['Body'].apply(lambda x: re.sub('@[^\s]+', '', x))
        posts = posts.drop(posts[posts.Body == ''].index)
        return posts

    def append_polarity_scores(self, posts):
        '''
        Modify dataframe by appending polarity scores
        Params: Dataframe containing preprocessed reddit posts
        Returns: Dataframe with posts and polarity scores
        '''
        if self.sa_type == 'vader':
            posts['Scores'] = posts['Body'].apply(
                lambda body: self.sid.polarity_scores(body))

            posts['compound'] = posts['Scores'].apply(
                lambda score_dict: score_dict['compound'])
            posts['comp_score'] = posts['compound'].apply(
                lambda x: 'pos' if x >= 0 else 'neg')

        else:
            posts['Scores'] = posts['Body'].apply(
                lambda body: TextBlob(body).sentiment.polarity)

            posts['Polarity'] = posts['Scores'].apply(
                lambda x: 'Pos' if x >= 0 else 'Neg')
        return posts

    def filter_posts(self):
        ''''
        Filter posts containing references to teams and players
        Params: None
        Returns: Filtered dataframe
        '''
        df = self.df
        df['TEAM A'] = df['TEAM A'].str.lower()
        df['TEAM B'] = df['TEAM B'].str.lower()

        data = defaultdict(list)
        i = 0
        for team in self.teams:
            roster = self.rosters['PLAYER'][self.rosters['TEAM'] == team]
            roster = roster.tolist()
            data, i = self.filter_team(df, roster, team, data, i)
        df = pd.DataFrame.from_dict(data, orient='index', columns=[
                                    'Target', 'Team', 'Body', 'Score'])
        return df

    def split_names(self, team, roster):
        player_names = []
        team_name = []
        for player in roster:
            player_names.append(player.split())
        team_name = team.split()
        return player_names, team_name

    def filter_team(self, df, roster, team, data, i):
        '''
        Helper function for filtering posts of a specific team and players in the roster
        Params: Polarity score dataframe, roster dataframe, team dataframe, dictionary, counter
        Returns: Dictionary and counter
        '''
        players, team_name = self.split_names(team, roster)
        posts = df['Body'][(df['TEAM A'] == team) | (df['TEAM B'] == team)]
        posts = posts.unique()
        posts = posts.tolist()

        for post in posts:
            for name in players:
                for n in name:
                    if n in post and post not in [x for v in data.values() for x in v]:
                        if n == 'al' or n == 'ed':
                            continue
                        data[i].append(' '.join(name))
                        data[i].append(team)
                        data[i].append(post)
                        if self.sa_type == 'vader':
                            score = df['comp_score'][df['Body'] == post]
                        else:
                            score = df['Polarity'][df['Body'] == post]

                        score = score.unique()
                        data[i].append(score.tolist())
                        i += 1
            for t in team_name:
                if t in post and post not in [x for v in data.values() for x in v]:
                    data[i].append(team)
                    data[i].append(team)
                    data[i].append(post)
                    if self.sa_type == 'vader':
                        score = df['comp_score'][df['Body'] == post]
                    else:
                        score = df['Polarity'][df['Body'] == post]
                    score = score.unique()
                    data[i].append(score.tolist())
                    i += 1

        return data, i
