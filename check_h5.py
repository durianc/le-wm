import h5py
import numpy as np

f = h5py.File('/mnt/data/szeluresearch/stable-wm/tworoom.h5', 'r')

print('Keys in file:', list(f.keys()))
print('proprio shape:', f['proprio'].shape)
print('pos_agent shape:', f['pos_agent'].shape)
print('action shape:', f['action'].shape if 'action' in f else 'No action key')
print('\nFirst sample:')
print('proprio[0]:', f['proprio'][0])
print('pos_agent[0]:', f['pos_agent'][0])
if 'action' in f:
    print('action[0]:', f['action'][0])

f.close()
