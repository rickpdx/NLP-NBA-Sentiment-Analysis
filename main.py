import pandas as pd
import praw
from collections import defaultdict
from sa_model import SentimentModel


def team_results(model):
    '''
    Target references to individual teams
    Params: Sentiment Analysis model
    Return: Dataframe including targets
    '''
    values = defaultdict(list)
    i = 0
    for team in model.teams:
        target = model.filtered['Score'][model.filtered['Team'] == team]
        counts = target.value_counts()
        counts = counts.tolist()
        values[i].append(team)
        values[i].append(counts[0])
        values[i].append(counts[1])
        values[i].append(round(counts[0]/(counts[0]+counts[1]), 2))
        values[i].append(round(counts[1]/(counts[0]+counts[1]), 2))
        i += 1
    values = pd.DataFrame.from_dict(values, orient='index', columns=[
        'Team', 'Pos', 'Neg', 'PPerc', 'NPerc'])
    if model.sa_type == 'vader':
        values.to_csv('data/team_results_VADER.csv', index=False)
    else:
        values.to_csv('data/team_results_TB.csv', index=False)
    return values


def top_teams(df, n):
    '''
    Read out top n teams (limit is 20) for each feature
    Params: Dataframe
    Return: None
    '''
    if n > 20:
        return

    neg = df.nlargest(n, 'Neg')
    pos = df.nlargest(n, 'Pos')
    n_p = df.nlargest(n, 'NPerc')
    p_p = df.nlargest(n, 'PPerc')
    print("---Top 10 negative sentiment---")
    print(neg)
    print("---Top 10 positive sentiment---")
    print(pos)

    print("---Top 10 negative percentage---")
    print(n_p)
    print("---Top 10 positive percentage---")
    print(p_p)


def player_results(model):
    '''
    Target references to individual players
    Params: Sentiment Analysis model
    Return: Dataframe including targets
    '''
    players = [x for x in model.rosters['PLAYER']]
    # print(players)
    print(model.filtered)
    values = defaultdict(list)
    i = 0
    for player in players:
        target = model.filtered['Score'][model.filtered['Target'] == player]
        counts = target.value_counts()
        title = counts.keys()
        size = counts.shape[0]
        if size == 0:
            continue

        if size == 1:
            counts = counts.tolist()
            if title[0] == '[\'pos\']':
                print('1')
                print(title[0])
                values[i].append(player)
                values[i].append(counts[0])
                values[i].append(0)
                values[i].append(1)
                values[i].append(0)

            else:
                print('2')
                values[i].append(player)
                values[i].append(0)
                values[i].append(counts[0])
                values[i].append(0)
                values[i].append(1)
        else:
            values[i].append(player)
            values[i].append(counts[0])
            values[i].append(counts[1])
            values[i].append(round(counts[0]/(counts[0]+counts[1]), 2))
            values[i].append(round(counts[1]/(counts[0]+counts[1]), 2))
        i += 1
    values = pd.DataFrame.from_dict(values, orient='index', columns=[
        'Player', 'Pos', 'Neg', 'PPerc', 'NPerc'])
    if model.sa_type == 'vader':
        values.to_csv('data/player_results_VADER.csv', index=False)
    else:
        values.to_csv('data/player_results_TB.csv', index=False)
    return values


def top_players(df, n):
    '''
    Read out top n players (limit is 100) for each feature
    Params: Dataframe
    Return: None
    '''

    if n > 100:
        return

    neg = df.nlargest(n, 'Neg')
    pos = df.nlargest(n, 'Pos')
    n_p = df.nlargest(n, 'NPerc')
    p_p = df.nlargest(n, 'PPerc')
    print("---Top 10 negative sentiment---")
    print(neg)
    print("---Top 10 positive sentiment---")
    print(pos)
    print("---Top 10 negative percentage---")
    print(n_p)
    print("---Top 10 positive percentage---")
    print(p_p)


def count_distribution(df1, df2):
    '''
    Calculate the distribution and compare classification of two Sentiment Analysis models
    Params: Vader Dataframe and TextBlob Dataframe
    Return: Dataframe with classification and body
    '''
    data = defaultdict(list)

    vader = [x for x in df1['comp_score']]
    tb = [x for x in df2['Polarity']]
    i = 0
    for x, y in zip(vader, tb):
        data[i].append(x)
        data[i].append(y)
        i += 1

    df = pd.DataFrame.from_dict(
        data, orient='index', columns=['VADER', 'TextBlob'])
    df['Post'] = df1['Body']
    return df


if __name__ == "__main__":

    # Reddit Credentials
    ua = ""
    c_id = ""
    c_sec = ""
    reddit = praw.Reddit(user_agent=ua, client_id=c_id, client_secret=c_sec)

    # Read in rosters and game threads
    rosters = pd.read_csv('data/rosters.csv')
    games = pd.read_csv('data/games.csv')
    teams = pd.read_csv('data/teams.csv')

    # Sentiment Analysis model for Vader
    sa = SentimentModel(games, teams, rosters, reddit, 'vader')
    sa.analysis()
    t_results = team_results(sa)
    p_results = player_results(sa)
    top_teams(t_results)
    top_players(p_results)

    # Sentiment Analysis model for TextBlob
    sa2 = SentimentModel(games, teams, rosters, reddit, 'textblob')
    sa2.analysis()
    t2_results = team_results(sa2)
    p2_results = player_results(sa2)
    top_teams(t2_results)
    top_players(p2_results)

    # Comparing Results
    df = count_distribution(sa.df, sa2.df)
    df.to_csv('data/comparison_of_scores.csv', index=False)
