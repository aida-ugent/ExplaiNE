import json
import numpy as np
import scipy.sparse as sparse
from collections import defaultdict
from os.path import join


from tqdm import *

from utils import (to_cache, from_cache)

def load_dblp_data(path_to_cache_file, recompute=False):
    """
    load DBLP data.
    """
    data = from_cache(path_to_cache_file)
    if data is not None and recompute is False:
        return (data['A'], data['E'], data['coauthor_dict'],
                data['author_id_dict'], data['id_author_dict'],
                data['paper_author_dict'])
    data_folder = '../data/dblp-ref/'
    target_venues = ['knowledge discovery and data mining',
                    'european conference on principles of data mining and knowledge discovery',
                    'data mining and knowledge discovery',
                    'neural information processing systems',
                    'journal of machine learning research',
                    'international conference on machine learning',
                    'machine learning',
                    'international conference on learning representations']
    records = []
    for i in range(4):
        data_file = 'dblp-ref-{:d}.json'.format(i)
        with open(join(data_folder, data_file)) as f:
            for i, line in tqdm(enumerate(f)):
                r = json.loads(line)
                venue = r['venue'].lower().strip()
                if venue in target_venues:
                    records.append(r)

    # compile indices
    coauthor_dict = defaultdict(dict)
    paper_author_dict = defaultdict(list)
    for record in records:
        author_list = record['authors']
        year = int(record['year'])
        n_authors = len(author_list)
        paper_author_dict[record['id']] = author_list
        for i in range(n_authors-1):
            author_i = author_list[i]
            for j in range(i+1, n_authors):
                author_j = author_list[j]
                if author_j not in coauthor_dict[author_i]:
                    coauthor_dict[author_i][author_j] = list()
                    coauthor_dict[author_j][author_i] = list()
                coauthor_dict[author_i][author_j].append(record['id'])
                coauthor_dict[author_j][author_i].append(record['id'])
    author_id_dict = {name: i for i,name in enumerate(coauthor_dict.keys())}
    id_author_dict = {i: name for i,name in enumerate(coauthor_dict.keys())}
    print('#authors: {:d}'.format(len(coauthor_dict.keys())))
    print('#papers: {:d}'.format(len(records)))

    # compute adjacency matrix
    E = []
    for author in coauthor_dict.keys():
        co_authors = coauthor_dict[author]
        for co_author in co_authors:
            E.append([author_id_dict[author], author_id_dict[co_author]])
    E = np.array(E)
    n = len(coauthor_dict.keys())
    A = sparse.csr_matrix((np.ones(len(E)), (E[:,0], E[:,1])), shape=(n, n))

    # persist data
    data = {
        'A': A,
        'E': E,
        'coauthor_dict': coauthor_dict,
        'author_id_dict': author_id_dict,
        'id_author_dict': id_author_dict,
        'paper_author_dict': paper_author_dict
    }
    to_cache(path_to_cache_file, data)
    return (data['A'], data['E'], data['coauthor_dict'], data['author_id_dict'],
           data['id_author_dict'], data['paper_author_dict'])
