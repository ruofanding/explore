import torch


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