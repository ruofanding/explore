import collections

cnt = collections.defaultdict(int)
l = 0
with open('data/train.csv', 'r') as input:
    for line in input:
        l += 1
        movie_id = int(line.split(',')[1])
        cnt[movie_id] += 1

with open('data/counter.csv', 'w') as output:
    for id in cnt:
        output.write(f'{id}:{cnt[id]}\n')
print(f'Read {l} line, and count {len(cnt)} movie ids.')
