__author__ = 'Jingzhang'

from seg_rnn import parse_file, parse_embedding

reader = './data/format/eng.testa.format'
embedding = parse_embedding('./data/embed/glove.6B.300d.txt')

parse_file(reader, embedding)