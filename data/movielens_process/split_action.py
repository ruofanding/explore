from itertools import chain
from random import shuffle

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
with open('train.csv', 'w') as train_output, open('test.csv',
                                                  'w') as test_output:
    train_output.writelines(train_actions)
    test_output.writelines(test_actions)
