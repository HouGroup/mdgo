import numpy as np


def atom_vec(atom1, atom2, dimension):
    vec = [0, 0, 0]
    for i in range(3):
        diff = atom1.position[i]-atom2.position[i]
        if diff > dimension[i]/2:
            vec[i] = diff - dimension[i]
        elif diff < - dimension[i]/2:
            vec[i] = diff + dimension[i]
        else:
            vec[i] = diff
    return np.array(vec)


def position_vec(pos1, pos2, dimension):
    vec = [0, 0, 0]
    for i in range(3):
        diff = pos1[i]-pos2[i]
        if diff > dimension[i]/2:
            vec[i] = diff - dimension[i]
        elif diff < - dimension[i]/2:
            vec[i] = diff + dimension[i]
        else:
            vec[i] = diff
    return np.array(vec)
