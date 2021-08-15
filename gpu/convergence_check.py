import pickle, os
import cupy

def get_similarity(run1, run2, top_words, tolerance):
    similarity = sum([any([len(set(list(run1[i].keys())[-top_words:]) - set(list(run2[j].keys())[-top_words:])) <= tolerance for i in range(n_topics)]) for j in range(n_topics)])
    return similarity



#print('loading whitened...')
filenames  = ['convergence_check/whitened/' + name for name in os.listdir('convergence_check/whitened')]
whiteneds = []
for name in filenames:
    with open(name, 'rb') as f:
        whiteneds.append(pickle.load(f))


#print('comparing whitened...')

tolerance = 1e-10
converges = all([all([bool(cupy.allclose(run2, run1, rtol=0, atol=tolerance)) for run1 in whiteneds]) for run2 in whiteneds])
assert converges
#print('converges:', converges)

#print()


#print('loading distributions...')
filenames  = ['convergence_check/distributions/' + name for name in os.listdir('convergence_check/distributions')]
distributions = []
for name in filenames:
    with open(name, 'rb') as f:
        distributions.append(pickle.load(f))

print('comparing distributions...')

n_topics = len(distributions[0])

top_words = 10
word_tolerance = 5
similarity_thresh = n_topics/2

convergence = all([all([get_similarity(run1, run2, top_words, word_tolerance) >= similarity_thresh  for run1 in distributions]) for run2 in distributions])

print('top_words word_tolerance similarity_thresh |', top_words, word_tolerance, similarity_thresh)
print('converges:', convergence)
