import numpy as np
import cPickle

for filename in ('vectors',):
    print 'Converting {}.txt'.format(filename)
    word2id = dict()
    embeddings = []
    with open('../data/w2v/{}.txt'.format(filename)) as f:
        n, k = map(int, f.readline().split())
        i = 0
        for line in f:
            word, vect = line[:line.find(' ')], line[line.find(' ') + 1:]
            word2id[unicode(word.decode('utf-8'))] = i
            embeddings.append(np.array(map(np.float32, vect.split())))
            i += 1
    embeddings = np.array(embeddings, dtype=np.float32)
    assert n == embeddings.shape[0]
    assert k == embeddings.shape[1]
    cPickle.dump((word2id, embeddings), open('../data/w2v/{}.pkl'.format(filename), 'wb'))
