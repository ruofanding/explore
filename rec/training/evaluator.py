import os
import torch

from data.movielens.movielens_dataset import MovieLensDataset, MovieLenseCollateFn


class RecallEvaluator:

    def __init__(self, path, config, topks=[10, 100, 200]):
        self.config = config
        test_ds = MovieLensDataset(path)
        sampler = torch.utils.data.distributed.DistributedSampler(
            test_ds, shuffle=False, drop_last=False) if config.device == 'cuda' else None
        self.test_data_loader = torch.utils.data.DataLoader(test_ds,
                                                            sampler=sampler,
                                                            num_workers=0,
                                                            batch_size=64,
                                                            shuffle=False,
                                                            collate_fn=MovieLenseCollateFn)
        self.rank = int(os.environ.get('RANK', 0))
        self.best_result = None
        self.best_epoch = -1
        self.topks = topks

    @torch.no_grad()
    def eval(self, model: torch.nn.Module, epoch: int):
        model.eval()
        if not hasattr(model, 'recall'):
            model = model.module
        recalled_num = torch.zeros(len(self.topks),
                                   dtype=torch.long,
                                   device=next(model.parameters()).device)
        total_samples = torch.zeros(1,
                                    dtype=torch.long,
                                    device=next(model.parameters()).device)
        device = torch.device(self.config.device)
        for samples in self.test_data_loader:
            user_id, movie_id, rating = samples
            total_samples += user_id.shape[0]
            user_id = user_id.to(device)
            topk_indices, var_mean = model.recall(user_id,
                                                  max(self.topks))
            for i, topk in enumerate(self.topks):
                movie_id = movie_id.to(device)
                target = movie_id.reshape(-1, 1).repeat(1, topk)
                recalled_num[i] += torch.sum(
                    torch.eq(target, topk_indices[:, :topk]).int()).cpu()

        if self.config.device == 'cuda':
            torch.distributed.all_reduce(recalled_num)
            torch.distributed.all_reduce(total_samples)
        recalled_num = recalled_num.cpu().numpy()
        total_samples = total_samples.item()
        if self.rank == 0:
            for i in range(len(self.topks)):
                print(
                    f'Recall@{self.topks[i]:3d} = {recalled_num[i]/total_samples:.3f} ({recalled_num[i]} / {total_samples})'
                )
        if (self.best_result is None
                ) or self.best_result[-1] < recalled_num[-1] / total_samples:
            self.best_result = recalled_num / total_samples
            self.best_epoch = epoch

    def best_result(self):
        return self.best_result
