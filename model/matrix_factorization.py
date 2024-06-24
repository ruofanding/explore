import collections
import torch
import torch.nn.functional as F
import torch.nn as nn
import pickle

class InbatchMF(torch.nn.Module):

    def __init__(self,
                 emb_dim=8,
                 user_num=-1,
                 movie_num=-1,
                 init_factor=(1 / 64),
                 use_bias=False,
                 normalize=False,
                 tmp=0.04,
                 eval_normalize=False,
                 counter_file=None,
                 sample_correction_weight=1.0,
                 important_sampling=False,
                 reweight=False):
        super().__init__()
        self.user_embedding = torch.nn.Embedding(user_num,
                                                 emb_dim,
                                                 sparse=False)
        self.movie_embedding = torch.nn.Embedding(movie_num,
                                                  emb_dim,
                                                  sparse=False)
        self.use_bias = use_bias
        self.normalize = normalize
        self.eval_normalize = eval_normalize
        self.tmp = tmp
        self.user_embedding.weight.data.normal_(0, init_factor)
        self.movie_embedding.weight.data.normal_(0, init_factor)

        if use_bias:
            self.user_bias = torch.nn.Embedding(user_num, 1)
            self.movie_bias = torch.nn.Embedding(movie_num, 1)
            self.user_bias.weight.data.zero_()
            self.movie_bias.weight.data.zero_()

        self.sample_correction_weight = sample_correction_weight
        self.important_sampling = important_sampling
        self.reweight = reweight
        if counter_file:
            self.frequency = {}
            with open(counter_file) as input:
                for line in input:
                    movie_id, cnt = [int(x) for x in line.split(':')]
                    self.frequency[movie_id] = cnt / 9900550.0 * 512.0
        else:
            self.frequency = None

        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, user_id, movie_id, rating):
        bsz = user_id.shape[0]
        user_embedding = self.user_embedding(user_id)
        movie_embedding = self.movie_embedding(movie_id)

        if self.normalize:
            user_embedding = F.normalize(user_embedding)
            movie_embedding = F.normalize(movie_embedding)

        logits = torch.mm(user_embedding, movie_embedding.t())
        if self.use_bias:
            user_bias = self.user_bias(user_id).tile(1, bsz)
            movie_bias = self.movie_bias(movie_id).tile(1, bsz)
            logits = logits + (user_bias + movie_bias.t())

        if self.frequency is not None:
            f = []
            for id in movie_id.cpu().numpy():
                f.append(self.frequency[id])
            f = torch.Tensor(f).cuda()
            f = torch.log(f) * self.sample_correction_weight
            logits = logits - f.t()

        loss = self.criterion(
            logits / self.tmp,
            torch.arange(user_id.shape[0], device=user_id.device))
        if self.reweight:
            loss = loss * (rating - 3) / 2.0
        return loss.mean()

    def recall(self, user_id, topk):
        user_embedding = self.user_embedding(user_id)
        movie_embedding = self.movie_embedding.weight
        if self.eval_normalize:
            user_embedding = F.normalize(user_embedding)
            movie_embedding = F.normalize(movie_embedding)
        logits = torch.mm(user_embedding, movie_embedding.t())
        if self.use_bias:
            movie_bias = self.movie_bias.weight
            logits = logits + movie_bias.t()

        _, topk_indices = torch.topk(logits, topk, dim=-1, sorted=True)
        return topk_indices, [
            torch.var_mean(logits[0]), logits[0].max(), logits[0].min()
        ]


class LRMF(torch.nn.Module):

    def __init__(self,
                 emb_dim,
                 user_num,
                 movie_num,
                 init_factor,
                 use_bias=False,
                 normalize=False,
                 eval_normalize=True, neg_num=10):
        super().__init__()
        self.user_embedding = torch.nn.Embedding(user_num, emb_dim)
        self.movie_embedding = torch.nn.Embedding(movie_num, emb_dim)
        self.user_embedding.weight.data.uniform_(0, init_factor)
        self.movie_embedding.weight.data.uniform_(0, init_factor)
        self.movie_num = movie_num
        self.neg_num = neg_num
        self.normalize = normalize
        self.eval_normalize = eval_normalize
        self.use_bias = use_bias
        if self.use_bias:
            self.user_bias = torch.nn.Embedding(user_num, 1)
            self.movie_bias = torch.nn.Embedding(movie_num, 1)
            self.user_bias.weight.data.zero_()
            self.movie_bias.weight.data.zero_()

        self.bce_criterion = torch.nn.BCELoss()

    def forward(self, user_id, movie_id, rating):
        bs = user_id.shape[0]

        user_id = torch.reshape(user_id, [bs, 1])
        movie_id = torch.reshape(movie_id, (-1, 1))
        if self.neg_num > 0:
            neg_id = torch.randint(0,
                                self.movie_num, [bs, self.neg_num],
                                device=user_id.device)
            all_movie_id = torch.cat([movie_id, neg_id], dim=1)
        else:
            all_movie_id = movie_id

        user_embedding = self.user_embedding(
            user_id)  # [bsz, 1] -> [bsz, 1, emb_dim]
        movie_embedding = self.movie_embedding(
            all_movie_id)  # [bsz, 1 + neg_num] -> [bsz, 1 + neg_num, emb_dim]

        if self.normalize:
            user_embedding = F.normalize(user_embedding, dim=2)
            movie_embedding = F.normalize(movie_embedding, dim=2)

        dot_product = torch.sum(user_embedding * movie_embedding,
                                dim=2)  # [bsz, 1 + neg_num]
        if self.use_bias:
            user_bias = self.user_bias(user_id)
            movie_bias = self.movie_bias(all_movie_id)
            dot_product = dot_product + user_bias.reshape(
                bs, 1) + movie_bias.reshape(bs, 1 + self.neg_num)

        logits = F.sigmoid(dot_product)
        if self.neg_num > 0:
            targets = torch.cat([
                torch.reshape(torch.ones_like(user_id), [bs, 1]),
                torch.zeros_like(neg_id)
            ],
                            dim=1)
        else:
            targets = torch.reshape(torch.ones_like(user_id), [bs, 1])
        return self.bce_criterion(logits, targets.float())

    def recall(self, user_id, topk):
        user_embedding = self.user_embedding(user_id)
        movie_embedding = self.movie_embedding.weight
        if self.eval_normalize:
            user_embedding = F.normalize(user_embedding)
            movie_embedding = F.normalize(movie_embedding)
        logits = torch.mm(user_embedding, movie_embedding.t())
        if self.use_bias:
            movie_bias = self.movie_bias.weight
            logits = logits + movie_bias.t()

        _, topk_indices = torch.topk(logits, topk, dim=-1, sorted=True)
        return topk_indices, [
            torch.var_mean(logits[0]), logits[0].max(), logits[0].min()
        ]

class DeepLRMF(torch.nn.Module):

    def __init__(self,
                 emb_dim,
                 user_num,
                 movie_num,
                 init_factor,
                 use_bias=False,
                 normalize=True,
                 eval_normalize=True,
                 neg_num=10,
                 layers=[64,32,1])\
            :
        super().__init__()
        self.emb_dim = emb_dim

        self.user_embedding = torch.nn.Embedding(user_num, emb_dim)
        self.movie_embedding = torch.nn.Embedding(movie_num, emb_dim)
        self.user_embedding.weight.data.uniform_(0, init_factor)
        self.movie_embedding.weight.data.uniform_(0, init_factor)
        self.movie_num = movie_num
        self.neg_num = neg_num
        self.normalize = normalize
        self.eval_normalize = eval_normalize
        self.use_bias = use_bias
        if self.use_bias:
            self.user_bias = torch.nn.Embedding(user_num, 1)
            self.movie_bias = torch.nn.Embedding(movie_num, 1)
            self.user_bias.weight.data.zero_()
            self.movie_bias.weight.data.zero_()

        self.bce_criterion = torch.nn.BCELoss()

        self.fc_stack = nn.ModuleList()
        prior = 2 * emb_dim
        for dim in layers:
            self.fc_stack.append(nn.Linear(prior, dim))
            prior = dim

    def forward(self, user_id, movie_id, rating, inference = False):

        bs = user_id.shape[0]

        user_id = torch.reshape(user_id, [bs, 1])
        movie_id = torch.reshape(movie_id, (-1, 1))
        if self.neg_num > 0 and not inference:
            neg_id = torch.randint(0,
                                self.movie_num, [bs, self.neg_num],
                                device=user_id.device)
            all_movie_id = torch.cat([movie_id, neg_id], dim=1)
        else:
            all_movie_id = movie_id

        user_embedding = self.user_embedding(
            user_id)  # [bsz, 1] -> [bsz, 1, emb_dim]
        movie_embedding = self.movie_embedding(
            all_movie_id)  # [bsz, 1 + neg_num] -> [bsz, 1 + neg_num, emb_dim]

        if self.normalize:
            user_embedding = F.normalize(user_embedding, dim=-1)
            movie_embedding = F.normalize(movie_embedding, dim=-1)

        # print(user_embedding.shape, movie_embedding.shape)
        x = torch.concat((user_embedding.repeat([1, 1 + (self.neg_num if not inference else 0), 1]), movie_embedding), axis = -1)

        for i, fc in enumerate(self.fc_stack):
            x = fc(x)
            if i < len(self.fc_stack) - 1:
                x = F.relu(x)

        x = x.view(-1, 1 + (self.neg_num if not inference else 0))

        if self.use_bias:
            user_bias = self.user_bias(user_id)
            movie_bias = self.movie_bias(all_movie_id)
            x = x + user_bias.reshape(
                bs, 1) + movie_bias.reshape(bs, 1 + (self.neg_num if not inference else 0))

        logits = F.sigmoid(x)
        if inference:
            return logits
        if self.neg_num > 0:
            targets = torch.cat([
                torch.reshape(torch.ones_like(user_id), [bs, 1]),
                torch.zeros_like(neg_id)
            ],
                            dim=1)
        else:
            targets = torch.reshape(torch.ones_like(user_id), [bs, 1])
        return self.bce_criterion(logits, targets.float())

    def recall(self, user_id, topk):
        batch_dim = len(user_id)

        user_embedding = self.user_embedding(user_id)
        movie_embedding = self.movie_embedding.weight

        movie_cnt = len(movie_embedding)

        if self.eval_normalize:
            user_embedding = F.normalize(user_embedding)
            movie_embedding = F.normalize(movie_embedding)

        uembed_stack = user_embedding.view(len(user_id), 1, -1).repeat(1, len(movie_embedding), 1) #bsz x emd_dim
        movie_embedding = movie_embedding.unsqueeze(dim = 0).repeat(len(uembed_stack), 1, 1)
        # print(uembed_stack.shape, movie_embedding.shape)
        x = torch.concat((uembed_stack.view(-1, self.emb_dim), movie_embedding.view(-1, self.emb_dim)), axis = -1)
        for i, fc in enumerate(self.fc_stack):
            x = fc(x)
            if i < len(self.fc_stack) - 1:
                x = F.relu(x)
        logits = x.view(batch_dim, movie_cnt)

        if self.use_bias:
            movie_bias = self.movie_bias.weight
            logits = logits + movie_bias.t()

        _, topk_indices = torch.topk(logits, topk, dim=-1, sorted=True)
        return topk_indices, [
            torch.var_mean(logits[0]), logits[0].max(), logits[0].min()
        ]

# class DeepLRMFGenre(torch.nn.Module):
#
#     def __init__(self,
#                  emb_dim,
#                  user_num,
#                  movie_num,
#                  init_factor,
#                  use_bias=False,
#                  normalize=False,
#                  eval_normalize=True,
#                  neg_num=10,
#                  layers=[64,32,1],
#                  genre_emb_dim = 8)\
#             :
#         super().__init__()
#         self.emb_dim = emb_dim
#         self.emb_dim = emb_dim
#
#         self.user_embedding = torch.nn.Embedding(user_num, emb_dim)
#         self.movie_embedding = torch.nn.Embedding(movie_num, emb_dim)
#         self.user_embedding.weight.data.uniform_(0, init_factor)
#         self.movie_embedding.weight.data.uniform_(0, init_factor)
#         self.movie_num = movie_num
#         self.neg_num = neg_num
#         self.normalize = normalize
#         self.eval_normalize = eval_normalize
#         self.use_bias = use_bias
#         self.g_emb_dim = genre_emb_dim
#         if self.use_bias:
#             self.user_bias = torch.nn.Embedding(user_num, 1)
#             self.movie_bias = torch.nn.Embedding(movie_num, 1)
#             self.user_bias.weight.data.zero_()
#             self.movie_bias.weight.data.zero_()
#
#         self.bce_criterion = torch.nn.BCELoss()
#
#         self.fc_stack = nn.ModuleList()
#         prior = 2 * emb_dim + genre_emb_dim
#         for dim in layers:
#             self.fc_stack.append(nn.Linear(prior, dim))
#             prior = dim
#         self.init_genre()
#
#         self.genre_embeddings = torch.nn.Embedding(self.genre_count, genre_emb_dim)
#
#
#
#     def get_movie_embed(self, movie_id):
#         movie_embedding = self.movie_embedding(movie_id)
#         genre_stack = []
#         flat_movies = movie_id.view(-1)
#         # print(movie_id.shape, movie_id)
#         for mid in flat_movies:
#             genres = self.movie_to_genre[mid.item()]
#             g_embs = self.genre_embeddings(torch.Tensor(genres).long())
#             genre_stack.append(g_embs.mean(dim = 0))
#         genre_stack = torch.stack(genre_stack)
#         # reshape_g = genre_stack.view(-1, self.neg_num + 1, self.g_emb_dim)
#         reshape_g = torch.zeros(movie_embedding.shape)
#         # print(reshape_g.shape)
#         final_embed = torch.cat((movie_embedding, reshape_g), axis = -1)
#         # print(final_embed.shape)
#         return final_embed
#
#     def forward(self, user_id, movie_id, rating, inference=False):
#         bs = user_id.shape[0]
#
#         user_id = torch.reshape(user_id, [bs, 1])
#         movie_id = torch.reshape(movie_id, (-1, 1))
#         if self.neg_num > 0 and not inference:
#             neg_id = torch.randint(0,
#                                 self.movie_num, [bs, self.neg_num],
#                                 device=user_id.device)
#             all_movie_id = torch.cat([movie_id, neg_id], dim=1)
#         else:
#             all_movie_id = movie_id
#
#         user_embedding = self.user_embedding(
#             user_id)  # [bsz, 1] -> [bsz, 1, emb_dim]
#         movie_embedding = self.get_movie_embed(
#             all_movie_id)  # [bsz, 1 + neg_num] -> [bsz, 1 + neg_num, emb_dim]
#
#         if self.normalize:
#             user_embedding = F.normalize(user_embedding, dim=2)
#             movie_embedding = F.normalize(movie_embedding, dim=2)
#
#         # print(user_embedding.shape, movie_embedding.shape)
#         x = torch.concat((user_embedding.repeat([1, 1 + (self.neg_num if not inference else 0), 1]), movie_embedding), axis = -1)
#
#         for i, fc in enumerate(self.fc_stack):
#             x = fc(x)
#             if i < len(self.fc_stack) - 1:
#                 x = F.relu(x)
#
#         x = x.view(-1, 1 + (self.neg_num if not inference else 0))
#
#         if self.use_bias:
#             user_bias = self.user_bias(user_id)
#             movie_bias = self.movie_bias(all_movie_id)
#             x = x + user_bias.reshape(
#                 bs, 1) + movie_bias.reshape(bs, 1 + self.neg_num)
#
#         logits = F.sigmoid(x)
#         if inference:
#             return logits
#         if self.neg_num > 0:
#             targets = torch.cat([
#                 torch.reshape(torch.ones_like(user_id), [bs, 1]),
#                 torch.zeros_like(neg_id)
#             ],
#                             dim=1)
#         else:
#             targets = torch.reshape(torch.ones_like(user_id), [bs, 1])
#         return self.bce_criterion(logits, targets.float())
#
#     def recall(self, user_id, topk):
#         batch_dim = len(user_id)
#
#         user_embedding = self.user_embedding(user_id)
#         movie_embedding = self.movie_embedding.weight
#
#         movie_cnt = len(movie_embedding)
#
#         if self.eval_normalize:
#             user_embedding = F.normalize(user_embedding)
#             movie_embedding = F.normalize(movie_embedding)
#
#         uembed_stack = user_embedding.view(len(user_id), 1, -1).repeat(1, len(movie_embedding), 1) #bsz x emd_dim
#         movie_embedding = movie_embedding.unsqueeze(dim = 0).repeat(len(uembed_stack), 1, 1)
#         # print(uembed_stack.shape, movie_embedding.shape)
#         x = torch.concat((uembed_stack.view(-1, self.emb_dim), movie_embedding.view(-1, self.emb_dim)), axis = -1)
#         for i, fc in enumerate(self.fc_stack):
#             x = fc(x)
#             if i < len(self.fc_stack) - 1:
#                 x = F.relu(x)
#         logits = x.view(batch_dim, movie_cnt)
#
#         if self.use_bias:
#             movie_bias = self.movie_bias.weight
#             logits = logits + movie_bias.t()
#
#         _, topk_indices = torch.topk(logits, topk, dim=-1, sorted=True)
#         return topk_indices, [
#             torch.var_mean(logits[0]), logits[0].max(), logits[0].min()
#         ]
#
#     def init_genre(self):
#         genre_count = 0
#         genre_map = {}
#         # self.movie_to_genre = [[0] * 10 for _ in range(len(movie_id_map))]
#         self.movie_to_genre = {}
#         with open("data/movie_map_store", "rb") as f:
#             self.movie_id_map = pickle.load(f)
#         with open("data/ml-20m/movies.csv") as input:
#             for i, line in enumerate(input):
#                 if i == 0:
#                     continue
#                 items = line.split(",")
#                 movie_id = int(items[0])
#                 title = ",".join(items[1:-1])
#                 genres = items[-1]
#                 if movie_id not in self.movie_id_map:  # not seen in ratings
#                     continue
#                 our_id = self.movie_id_map[movie_id]
#                 genre_list = genres.split("|")
#                 for g in genre_list:
#                     if g not in genre_map:
#                         genre_map[g] = genre_count
#                         genre_count += 1
#                 self.movie_to_genre[our_id] = [genre_map[g] for g in genre_list]
#         self.genre_count = genre_count