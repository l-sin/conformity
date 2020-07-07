from joblib import load
import sys
import numpy as np

fname = str(sys.argv[1])
optimization_results = load(fname)
                            
print('Showing the top 10 optimization results')
print('(B, rscaleMod, veldispMod), BAS, ARI, FMI')

#print(sorted(optimization_results,
#                                 key=lambda x: x[1] if not np.isnan(x[1]) else 0,
#                                 reverse=True)[:10])

for params,bas,ari,fmi in sorted(optimization_results,
                                 key=lambda x: x[1] if not np.isnan(x[1]) else 0,
                                 reverse=True)[:10]:
    print('({0:.1f},{1:.1f},{2:.1f}),{3:.3f},{4:.3f},{5:.3f}'.format(*params,bas,ari,fmi))