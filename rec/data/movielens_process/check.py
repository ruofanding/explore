with open('ml-20m/ratings.csv') as input:
    movie_ids = set()
    cnt = 0
    for i, line in enumerate(input):
        if i == 0:
            continue
        cnt += 1
        movie_id = int(line.split(',')[1])
        movie_ids.add(movie_id)
    print(len(movie_ids), cnt)