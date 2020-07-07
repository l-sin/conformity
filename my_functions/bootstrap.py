def bootstrap(func,N,sample,weights=None):
    if weights is None:
        return [ func(np.random.choice(sample, size=len(sample),replace=True)) for i in range(N) ]
    else:
        pass