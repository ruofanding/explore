from itertools import chain
from random import shuffle

from absl import app
from absl import flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('num_shards', 1, 'number of shards')

def main(argv):
    with open('user_action.csv') as input:
        current_id = 0
        user_actions = []
        train_actions = []
        test_actions = []
        for line in chain(input, ['999999,999999,5,0']):
            col = line.split(',')
            user_id = int(col[0])

            if current_id != user_id:
                if (current_id % 129 == 0):
                    user_actions.sort(key=lambda x: int(x.split(',')[-1]))
                    half = len(user_actions) // 2
                    train_actions.extend(user_actions[:half])
                    test_actions.extend(user_actions[half:])
                else:
                    train_actions.extend(user_actions)
                user_actions = []
                current_id = user_id

            user_actions.append(line)
    shuffle(train_actions)
    shuffle(test_actions)


    lines_per_shard = len(train_actions) // FLAGS.num_shards
    for i in range(FLAGS.num_shards):
        with open('train_{}.csv'.format(i), 'w') as train_output:
            train_output.writelines(train_actions[lines_per_shard*i: min(lines_per_shard*(i+1), len(train_actions))])

    with open('test.csv',
                                                    'w') as test_output:
        test_output.writelines(test_actions)


if __name__ == '__main__':
    app.run(main)