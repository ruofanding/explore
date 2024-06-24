'''
$ torchrun --nproc_per_node 4 train.py
'''
from datetime import datetime
import sys
import torch
import os
from training.trainer import Trainer
from training.evaluator import RecallEvaluator
from data.dataset.movielens_dataset import MovieLensDataset, MovieLensDatasetGenre
from model.matrix_factorization import InbatchMF, LRMF, DeepLRMF, DeepLRMFGenre
from gv.generative_video import GVRecall
import torch.optim as optim

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('model', 'InbatchMF', 'LRMF/InbatchMF/DeepLRMF')
flags.DEFINE_string('optimizer', 'SGD', 'SGD/RSMP')
flags.DEFINE_string('data_dir', 'data/',
                    'path of data directory')
flags.DEFINE_integer('emb_dim', 8, 'embedding dimension')
flags.DEFINE_integer('bsz', 512, 'batch size')
flags.DEFINE_integer('num_epoch', 5, 'num of epoch')
flags.DEFINE_integer('neg_num', 10,
                     'number of random sampled negatives for LR.')
flags.DEFINE_bool('use_bias', False, 'use bias')
flags.DEFINE_bool('normalize', True, 'use bias')
flags.DEFINE_bool('sample_correction', False, 'use sampling correction')
flags.DEFINE_bool('eval_normalize', True, 'eval normalize')
flags.DEFINE_bool('important_sampling', False, '')
flags.DEFINE_bool('reweight', False, 'if reweight with rating')
flags.DEFINE_float('lr', 1e-3, 'lr')
flags.DEFINE_float('init_factor', 1 / 64.0, 'init factor')
flags.DEFINE_float('tmp', 0.04, 'temperature')
flags.DEFINE_float('weight_decay', 0.0, 'weight decay')
flags.DEFINE_float('sample_correction_weight', 1.0, '')
flags.DEFINE_bool('with_genre', False, 'use genre embeddings')

class ResultLogger():

    def __init__(self, trainer, output, rank, argv=[], evaluator=None):
        self.trainer = trainer
        self.rank = rank
        self.argv = argv
        self.evaluator = evaluator
        self.output = output

    def write_result(self):
        if self.rank != 0:
            return
        if self.trainer.last_loss is None:
            return

        loss = self.trainer.last_loss
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        best_recall = None
        best_epoch = None
        last_epoch = self.trainer.last_epoch
        if self.evaluator.best_result is not None:
            best_recall = self.evaluator.best_result
            best_epoch = self.evaluator.best_epoch
        argv = str(self.argv[1:])
        result = ','.join([
            timestamp, f'{loss:.4f}',
            str(best_recall),
            str(best_epoch),
            str(last_epoch), argv
        ])
        with open(self.output, 'a') as file:
            file.write(result + '\n')


def main(argv):
    trainer = Trainer(FLAGS.num_epoch,False, log_freq=1000)

    rank = int(os.environ.get('RANK', 0))
    if not FLAGS.with_genre:
        ds = MovieLensDataset(os.path.join(FLAGS.data_dir, 'train.csv' if rank == 0 else f'train_{rank}.txt'))
    else:
        ds = MovieLensDatasetGenre(os.path.join(FLAGS.data_dir, 'train.csv' if rank == 0 else f'train_{rank}.txt'))

    data_loader = torch.utils.data.DataLoader(ds,
                                              num_workers=0,
                                              batch_size=FLAGS.bsz,
                                              shuffle=True)

    if FLAGS.model == 'InbatchMF':
        model = InbatchMF(
            emb_dim=FLAGS.emb_dim,
            user_num=129797,
            movie_num=20709,
            init_factor=FLAGS.init_factor,
            use_bias=FLAGS.use_bias,
            normalize=FLAGS.normalize,
            tmp=FLAGS.tmp,
            eval_normalize=FLAGS.eval_normalize,
            counter_file=(os.path.join(FLAGS.data_dir, 'counter.csv')
                          if FLAGS.sample_correction else None),
            sample_correction_weight=FLAGS.sample_correction_weight,
            important_sampling=FLAGS.important_sampling,
            reweight=FLAGS.reweight)
    elif FLAGS.model == 'LRMF':
        model = LRMF(FLAGS.emb_dim,
                     129797,
                     20709,
                     FLAGS.init_factor,
                     use_bias=FLAGS.use_bias,
                     normalize=FLAGS.normalize,
                     eval_normalize=FLAGS.eval_normalize,
                     neg_num=FLAGS.neg_num)
    elif FLAGS.model == 'DeepLRMF':
        model = DeepLRMF(FLAGS.emb_dim,
                     129797,
                     20709,
                     FLAGS.init_factor,
                     use_bias=FLAGS.use_bias,
                     normalize=FLAGS.normalize,
                     eval_normalize=FLAGS.eval_normalize,
                     neg_num=FLAGS.neg_num)
    elif FLAGS.model == 'DeepLRMFGenre':
        model = DeepLRMFGenre(FLAGS.emb_dim,
                     129797,
                     20709,
                     FLAGS.init_factor,
                     use_bias=FLAGS.use_bias,
                     normalize=FLAGS.normalize,
                     eval_normalize=FLAGS.eval_normalize,
                     neg_num=FLAGS.neg_num)
    model = model.cpu()

    if FLAGS.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=FLAGS.lr)
    elif FLAGS.optimizer == 'RMSP':
        optimizer = optim.RMSprop(model.parameters(),
                                  lr=FLAGS.lr,
                                  weight_decay=FLAGS.weight_decay)
    evaluator = RecallEvaluator(os.path.join(FLAGS.data_dir, 'test.csv'))
    result_logger = ResultLogger(trainer,
                                 'result.txt',
                                 rank,
                                 argv=sys.argv,
                                 evaluator=evaluator)
    try: #use flags for model loading
        model_save_title = "./trained_models/" + str(sys.argv[1:])
        if os.path.exists(model_save_title):
            model.load_state_dict(torch.load(model_save_title))
        else:
            trainer.fit(model, optimizer, data_loader, evaluator=evaluator)
            result_logger.write_result()
            torch.save(model.state_dict(), model_save_title)

    except KeyboardInterrupt:
        pass
    GVRecall(model, data_loader)



if __name__ == '__main__':
    app.run(main)
