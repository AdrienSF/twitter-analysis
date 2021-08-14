import pickle, os
import cupy

def are_similar(run1, run2, top_words, tolerance):
    similarity = all([any([len(set(list(run1[i].keys())[-top_words:]) - set(list(run2[j].keys())[-top_words:])) <= tolerance for i in range(n_topics)]) for j in range(n_topics)])
    return similarity



print('loading whitened...')
filenames  = ['convergence_check/whitened/' + name for name in os.listdir('convergence_check/whitened')]
whiteneds = []
for name in filenames:
    with open(name, 'rb') as f:
        whiteneds.append(pickle.load(f))


print('comparing whitened...')

tolerance = 1e-10
converges = all([all([bool(cupy.allclose(run2, run1, rtol=0, atol=tolerance)) for run1 in whiteneds]) for run2 in whiteneds])

print('converges:', converges)

print()


print('loading distributions...')
filenames  = ['convergence_check/distributions/' + name for name in os.listdir('convergence_check/distributions')]
distributions = []
for name in filenames:
    with open(name, 'rb') as f:
        distributions.append(pickle.load(f))

print('comparing distributions...')

n_topics = len(distributions[0])

top_words = 10
tolerance = 7

run1 = distributions[0]
run2 = distributions[1]
converges = all([all([are_similar(run1, run2, top_words, tolerance) for run1 in distributions]) for run2 in distributions])


print('converges:', converges)
