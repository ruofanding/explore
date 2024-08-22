import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim


class Trainer():

    def __init__(self, epoch, config, log_freq=5000):
        self.total_epoch = epoch
        self.config = config
        self.log_freq = log_freq
        self.rank = int(os.environ.get('RANK', 0))

        if self.config.device == 'cuda':
            self.world_size = int(os.environ.get('WORLD_SIZE', 1))
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            self.setup_distributed()

    def __move_to_device(self, data):
        device = torch.device(self.config.device)
        if isinstance(data, list) or isinstance(data, tuple):
            data = [d.to(device) for d in data]
        else:
            data = {key: data[key].to(device) for key in data}
        return data

    def broadcast(self, model: torch.nn.Module, src=0) -> None:
        """
        将model的参数做broadcast
        """
        for v in model.state_dict().values():
            torch.distributed.broadcast(v, src)

    def setup_distributed(self):
        master_address = os.environ.get('MASTER_ADDR', "127.0.0.1")
        master_port = int(os.environ.get('MASTER_PORT', 34171))
        torch.cuda.set_device(self.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='tcp://{}:{}'.format(
                                                 master_address, master_port),
                                             world_size=self.world_size,
                                             rank=self.rank,
                                             group_name='mtorch')

    def fit(self, model, optimizer, train_loader, evaluator=None):
        if self.config.device == 'cuda':
            self.broadcast(model)
            model = DDP(model,
                        device_ids=[self.local_rank],
                        find_unused_parameters=False)

        total_step = 0
        self.last_loss = None
        for epoch in range(self.total_epoch):
            self.last_epoch = epoch
            running_loss = 0
            steps = 0
            for batch_idx, samples in enumerate(train_loader):
                optimizer.zero_grad()

                samples = self.__move_to_device(samples)
                if isinstance(samples, list):
                    loss = model(*samples)
                else:
                    loss = model(**samples)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                steps += 1
                if batch_idx % self.log_freq == 0 and batch_idx != 0 and self.rank == 0:
                    self.last_loss = running_loss / steps
                    print(
                        f'[{epoch + 1}, {batch_idx + 1:5d}] loss: {self.last_loss:.3f}'
                    )
                    running_loss = 0.0
                    steps = 0
                total_step += 1
            self.last_loss = running_loss / steps

            # torch.save(model.state_dict(), f'model_{total_step}.th')
            # print(f'user_var_mean={torch.var_mean(model.user_embedding.weight)}')
            # print(f'movie_var_mean={torch.var_mean(model.movie_embedding.weight)}')

            if evaluator:
                evaluator.eval(model, epoch)
