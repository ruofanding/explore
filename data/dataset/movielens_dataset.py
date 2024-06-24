import torch
import pickle
from tqdm import tqdm
class MovieLensDataset(torch.utils.data.Dataset):

    def __init__(self, input_file):
        if not isinstance(input_file, list):
            input_file = [input_file]
        super(MovieLensDataset).__init__()
        self.examples = []
        for file in input_file:
            with open(file) as input:
                for i, line in enumerate(input):
                    col = line.split(',')
                    user_id, movie_id, rating = int(col[0]), int(
                        col[1]), float(col[2])
                    self.examples.append((user_id, movie_id, rating))
            print(f'Load {len(self.examples)} examples from {input_file}')

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


import torch

class MovieLensDatasetGenre(torch.utils.data.Dataset): #has genre information

    def __init__(self, input_file):
        if not isinstance(input_file, list):
            input_file = [input_file]
        super(MovieLensDataset).__init__()
        self.examples = []
        self.movie_to_genre = {}


        for file in input_file:
            with open(file) as input:
                for i, line in enumerate(input):
                    col = line.split(',')
                    user_id, movie_id, rating = int(col[0]), int(
                        col[1]), float(col[2])
                    self.examples.append((user_id, movie_id, rating))
            print(f'Load {len(self.examples)} examples from {input_file}')

        with open("data/movie_map_store", "rb") as f:
            self.movie_id_map = pickle.load(f)

        genre_count = 0
        genre_map = {}
        with open("data/ml-20m/movies.csv") as input:
            for i, line in enumerate(input):
                if i == 0:
                    continue
                items = line.split(",")
                movie_id = int(items[0])
                title = ",".join(items[1:-1])
                genres = items[-1]
                if movie_id not in self.movie_id_map: #not seen in ratings
                    continue
                our_id = self.movie_id_map[movie_id]
                genre_list = genres.split("|")
                for g in genre_list:
                    if g not in genre_map:
                        genre_map[g] = genre_count
                        genre_count +=1
                self.movie_to_genre[our_id] = [genre_map[g] for g in genre_list]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        user_id, movie_id, rating = self.examples[idx]
        genre = self.movie_to_genre[movie_id]
        padded_genre = genre + [-1] * (10 - len(genre))
        return (user_id, movie_id, rating, padded_genre)
