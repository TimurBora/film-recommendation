import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join('..')))
from src.data_processing.data_loader import MovieLensDataLoader
md = MovieLensDataLoader("/home/pajalone/film-recommendation/data/raw/movielens/ml-latest-small")
md.load_data()

#print(md.get_movie_info(1))
#md.get_movie_genres(1)
#print(md.links_df.head())

    

        