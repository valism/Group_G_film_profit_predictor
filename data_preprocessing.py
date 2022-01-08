import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import raw data
imdb_df = pd.read_csv('IMDb movies.csv')

#list(imdb_df.columns.values)
# drop columns
drop_cols = ['title',
             'original_title',
             'country',
             'description',
             'avg_vote','votes',
             'usa_gross_income',
             'metascore',
             'reviews_from_users',
             'reviews_from_critics',
             'director',
             'writer',
             'actors']

imdb_df.drop(columns=drop_cols, axis=1, inplace=True)

# drop rows that contain any NaN
imdb_df.dropna(inplace=True)

# filtering for films released post 2015
imdb_df[["year"]] = imdb_df[["year"]].apply(pd.to_numeric)
imdb_df = imdb_df[imdb_df['year'] > 2015]

# filtering for films measured in $USD
imdb_df = imdb_df[imdb_df['budget'].str.contains('$', regex=False)]
imdb_df = imdb_df[imdb_df['worlwide_gross_income'].str.contains('$', regex=False)]

# convert 'budget' and 'worlwide_gross_income' columns to integers
imdb_df['budget'] = imdb_df['budget'].str.extract('(\d+)').astype(int)
imdb_df['worlwide_gross_income'] = imdb_df['worlwide_gross_income'].str.extract('(\d+)').astype(int)

# reading showed that largest markets are US, China and Korean so films with one of these languages are of interest
def language_enc(imdb_df):
    # has film been released in english, chinese or korean. one-hot encoding
    imdb_numeric_language_df = imdb_df['language'].str.get_dummies(sep=', ')
    imdb_numeric_language_df = imdb_numeric_language_df[['English','Chinese','Mandarin','Cantonese','Korean']]
    imdb_numeric_language_df['Chinese'] = imdb_numeric_language_df[['Chinese','Mandarin','Cantonese']].max(axis=1)
    imdb_numeric_language_df.drop(columns=['Mandarin','Cantonese'], inplace=True)
    imdb_numeric_language_df = imdb_numeric_language_df.add_prefix('LANGUAGE_')
    #print(imdb_numeric_language_df.sum())

    imdb_df = imdb_df.merge(imdb_numeric_language_df, left_index=True, right_index=True)
    imdb_df.drop(columns=['language'], inplace=True)
    
    return imdb_df


# Top 5 Production companies defined as highest grossers. 20th Century Fox also added in as extra due to high number of films
        # 1) Universal Pictures
        # 2) Warner Bros
        # 3) Columbia Pictures
        # 4) Walt Disney Pictures
        # 5) Marvel Studios
        # 7) 20th Century Fox
def production_enc(imdb_df):
    # standardise spellings of named production companies of interest
    imdb_df.loc[imdb_df['production_company'].str.contains('marvel', case=False), 'production_company'] = 'Marvel Studios'
    imdb_df.loc[imdb_df['production_company'].str.contains('universal', case=False), 'production_company'] = 'Universal Pictures'
    imdb_df.loc[imdb_df['production_company'].str.contains('warner', case=False), 'production_company'] = 'Warner Bros'
    imdb_df.loc[imdb_df['production_company'].str.contains('Twentieth', case=False), 'production_company'] = 'Twentieth Century Fox'
    imdb_df.loc[imdb_df['production_company'].str.contains('20th', case=False), 'production_company'] = 'Twentieth Century Fox'
    imdb_df.loc[imdb_df['production_company'].str.contains('disney', case=False), 'production_company'] = 'Walt Disney Pictures'

    imdb_numeric_pc_df = imdb_df['production_company'].str.get_dummies(sep=', ')
    imdb_numeric_pc_df = imdb_numeric_pc_df[['Marvel Studios','Columbia Pictures','Universal Pictures','Warner Bros','Twentieth Century Fox','Walt Disney Pictures']]
    imdb_numeric_pc_df = imdb_numeric_pc_df.add_prefix('PC_')

    imdb_df = imdb_df.merge(imdb_numeric_pc_df, left_index=True, right_index=True)
    imdb_df.drop(columns=['production_company'], inplace=True)
    
    return imdb_df

def date_enc(imdb_df):
    # split out published date column and group by season
    date = imdb_df['date_published'].str.split('-', expand=True)
    spring = ["01","02","03","04"]
    summer = ["05","06","07","08"]
    autumn = ["09","10"]
    winter = ["11","12"]

    date[1] = date[1].replace(spring,'spring',regex=True)
    date[1] = date[1].replace(summer,'summer',regex=True)
    date[1] = date[1].replace(autumn,'autumn',regex=True)
    date[1] = date[1].replace(winter,'winter',regex=True)

    imdb_df['date_published'] = date[1]
    imdb_df.rename(columns={"date_published": "SEASON_RELEASED"}, inplace=True)
    imdb_numeric_season_df = imdb_df['SEASON_RELEASED'].str.get_dummies()
    imdb_df = imdb_df.merge(imdb_numeric_season_df, left_index=True, right_index=True)
    imdb_df.drop(columns=['SEASON_RELEASED'], inplace=True)

    # year released one-hot encoding
    year = imdb_df['year'].astype(str).replace(' ','')
    year = year.str.get_dummies()
    year = year.add_prefix('YEAR_')
    imdb_df = imdb_df.merge(year, left_index=True, right_index=True)
    imdb_df.drop(columns=['year'], inplace=True)
    
    return imdb_df

# genre grouping and one-hot encoding
def genre_enc(imdb_df):
    # create new genres dataframe and split out by comma seperation
    genres_df = imdb_df['genre'].str.split(',', expand=True)
    genres_df.rename(columns={0: 'genre_1', 1: 'genre_2', 2: 'genre_3'}, inplace=True)
    genres_df['genre_1'] = genres_df['genre_1'].str.strip()
    genres_df['genre_2'] = genres_df['genre_2'].str.strip()
    genres_df['genre_3'] = genres_df['genre_3'].str.strip()

    # calculate the number of films that fall into each genre
    genre_counts = pd.DataFrame(genres_df['genre_1'].value_counts())
    genre_counts = genre_counts.merge(pd.DataFrame(genres_df['genre_2'].value_counts()), how='outer', left_index=True, right_index=True)
    genre_counts = genre_counts.merge(pd.DataFrame(genres_df['genre_3'].value_counts()), how='outer', left_index=True, right_index=True)
    genre_counts['Count']= genre_counts.sum(axis=1)
    genre_counts["Percentage"] = (genre_counts["Count"] / genre_counts["Count"].sum() * 100).round(1)
    print("The table below shows the occurence of each genre in the imdb database and the total percentage of films they occur in")
    print("95.7% of films are covered by first 14 genres. the final 6 genres are only found in 4.3 percent of films, so are aggregated to form an 'Other' genre grouping")
    print(genre_counts.sort_values(by=['Percentage'], ascending=False))

    # as shown, 95.7% of films are covered by first 14 genres. the final 6 genres are only found in 4.3% of films, so are aggregated to form an 'Other' genre grouping
    other = ['Western','Musical','War','Sport','Music','History']
    # replace the 4.3% 'other' genres with 'Other' string in the genre column of the main imdb_df
    imdb_df["genre"] = imdb_df["genre"].replace(other,'Other',regex=True)

    # one-hot encoding of 14 named genres and 'Other' genres
    imdb_numeric_genres_df = imdb_df['genre'].str.get_dummies(sep=', ')
    imdb_numeric_genres_df = imdb_numeric_genres_df.add_prefix('GENRE_')
    imdb_df = imdb_df.merge(imdb_numeric_genres_df, left_index=True, right_index=True)
    imdb_df.drop(columns=['genre'], inplace=True)

    return imdb_df

# film_pair_selection selects as close to 10,000 pairs as possible that ensures equal number of films across each of the 15 genre categories are included in dataset, and that only films with at least one genre in common are compared
def film_overlapping_genre_selection(imdb_df):
    # get a list of GENRE_ column names to be used to iterate through for selection process. Must remove first n=18 columns as not GENRE related
    n = 22
    cols = imdb_df.columns.tolist()
    del cols[:n]

    # m=36 defined as the number of films possible to pick for 15 genre categories to allow for 10,000 pairs
    m = 36
    genres_films_lst = []
    # for each GENRE_ column, select films that are of that genre (i.e. =1)
    for col in cols:
        genre_sel = (imdb_df.loc[imdb_df[col] == 1]).head(m)
    
        # genre matrix to assist pairwise comparison and remove duplicate comparisons
        genre_mtx = pd.DataFrame(np.diag(genre_sel["duration"]), columns=genre_sel["imdb_title_id"], index=genre_sel["imdb_title_id"])
        genre_tri = genre_mtx.mask(np.triu(np.ones(genre_mtx.shape)).astype(bool)).stack()
    
        # create and tidy up dataframe indexes
        genre_films = genre_tri.to_frame()
        genre_films.index.names = ['title_x','title_y']
        genre_films = genre_films.reset_index()
        genre_films.drop(columns=[0], axis=1, inplace=True)
    
        # append to genres_films_lst
        genres_films_lst.append(genre_films)

    # concatenate list together to form one big dataframe
    genres_films = pd.concat(genres_films_lst, ignore_index=True)

    # merge in film_x variables
    genres_films = genres_films.merge(imdb_df, how='left', left_on='title_x', right_on='imdb_title_id')
    # merge in film_y variables
    genres_films = genres_films.merge(imdb_df, how='left', left_on='title_y', right_on='imdb_title_id')
    genres_films.drop(columns=['imdb_title_id_x','imdb_title_id_y'], axis=1, inplace=True)

    # create profit columns
    genres_films["profit_x"] = genres_films["worlwide_gross_income_x"] - genres_films["budget_x"]
    genres_films["profit_y"] = genres_films["worlwide_gross_income_y"] - genres_films["budget_y"]

    # classifying if film_x has made more profit than film_y. 1=yes, 0=no
    conditions = [
        (genres_films['profit_x'] > genres_films['profit_y']),
        (genres_films['profit_x'] < genres_films['profit_y']),
        (genres_films['profit_x'] == genres_films['profit_y'])
        ]
     
    values = ['1','0','0']
     
    genres_films['profit_xy'] = np.select(conditions, values)

    return genres_films


imdb_df = language_enc(imdb_df)
imdb_df = production_enc(imdb_df)
imdb_df = date_enc(imdb_df)
imdb_df = genre_enc(imdb_df)

# select derived overlapping genre dataset (i.e. pairwise comparison of roughly 10,000 pairs)
genres_films = film_overlapping_genre_selection(imdb_df)
genres_films.drop(columns=['worlwide_gross_income_x','worlwide_gross_income_y','profit_x','profit_y'], inplace=True)

# histogram plot of film_x duration
plt.hist(genres_films['duration_x'], density=True)
plt.savefig("duration_x.jpg")

# histogram plot of film_x budget
plt.hist(genres_films['budget_x'], density=True)
plt.savefig("budget_x.jpg")

genres_films.to_csv('profit_x_y.csv')

#print(imdb_df)