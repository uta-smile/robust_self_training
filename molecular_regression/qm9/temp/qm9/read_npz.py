import numpy as np

a = np.load('strain.npz')
# print(len(a['num_atoms']))
print(a.keys())
# print(a['Cv'])
# print(a['Cv_thermo'])
print(a['lumo'])
print(a['mu'])

