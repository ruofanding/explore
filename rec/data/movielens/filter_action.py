next_id = 0
current_user_id = 1
user_actions = []
user_cnt = 0
action_cnt = 0
movie_id_map = {}


def write_to_output(user_actions, output):
    global next_id, user_cnt, action_cnt, movie_id_map
    if len(user_actions) >= 10:
        for action in user_actions:
            _, movie_id, remaining = action.split(',', 2)
            movie_id = int(movie_id)
            if movie_id not in movie_id_map:
                movie_id_map[movie_id] = len(movie_id_map)
            new_line = ','.join(
                [str(next_id),
                 str(movie_id_map[movie_id]), remaining])
            output.writelines([new_line])
        user_cnt += 1
        action_cnt += len(user_actions)

        next_id += 1


with open('ml-20m/ratings.csv') as input, open('user_action.csv',
                                               'w') as output:
    for i, line in enumerate(input):
        # skip column name
        if i == 0:
            continue

        col = line.split(',')
        user_id, rating = int(col[0]), float(col[2])

        if current_user_id != user_id:
            write_to_output(user_actions, output)
            user_actions = []
            current_user_id = user_id

        if rating >= 4.0:
            user_actions.append(line)

        if i % 1000000 == 99999:
            print('total user:{}, total action:{}, total_movie:{}'.format(
                user_cnt, action_cnt, len(movie_id_map)))
    write_to_output(user_actions, output)
    print('total user:{}, total action:{}, total_movie:{}'.format(
        user_cnt, action_cnt, len(movie_id_map)))
