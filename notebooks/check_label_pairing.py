import pickle

with open(r'data\MFCAD\dataset\step\0-0-0-0-0-23.face_truth', 'rb') as f:
    labels = pickle.load(f)
print('Label count:', len(labels))
print('Labels:', labels)

from occwl.io import load_step
solid = load_step(r'data\MFCAD\dataset\step\0-0-0-0-0-23.step')[0]
faces = list(solid.faces())
print('Face count from occwl:', len(faces))