"""
Trace the exact nesting structure of People.mat[0,0]['Sen']
"""
import scipy.io as sio
import numpy as np

PEOPLE_MAT = r'C:\Users\2005g\Downloads\4691144\People.mat'

mat = sio.loadmat(PEOPLE_MAT, struct_as_record=True, squeeze_me=False)
people = mat['people']

person = people[0, 0]
print(f"person: shape={person.shape}, dtype={person.dtype}")
# person is (1,1) structured array with field 'Sen'

sen = person['Sen'][0, 0]  # extract the single element
print(f"Sen: type={type(sen)}, shape={sen.shape}, dtype={sen.dtype}")
# Sen is (1, 5) object array -> 5 speed conditions

for speed_idx in range(sen.shape[1]):
    cell = sen[0, speed_idx]
    print(f"\n  Speed {speed_idx}: type={type(cell)}, shape={getattr(cell, 'shape', 'N/A')}, dtype={getattr(cell, 'dtype', 'N/A')}")
    
    if isinstance(cell, np.ndarray):
        if cell.dtype == object:
            print(f"    Object array. Inspecting elements...")
            for i in range(min(cell.shape[0], 3)):
                for j in range(cell.shape[1] if cell.ndim > 1 else 1):
                    if cell.ndim > 1:
                        elem = cell[i, j]
                    else:
                        elem = cell[i]
                    print(f"    [{i},{j}]: type={type(elem)}, shape={getattr(elem, 'shape', 'N/A')}, dtype={getattr(elem, 'dtype', 'N/A')}")
                    if isinstance(elem, np.ndarray) and elem.dtype != object:
                        flat = elem.flatten()
                        print(f"      *** NUMERIC DATA: {len(flat)} samples, first 3: {flat[:3]}")
                    elif isinstance(elem, np.ndarray) and elem.dtype == object:
                        print(f"      Still object. size={elem.size}")
                        if elem.size > 0:
                            inner = elem.flat[0]
                            print(f"      inner: type={type(inner)}, shape={getattr(inner, 'shape', 'N/A')}, dtype={getattr(inner, 'dtype', 'N/A')}")
                            if isinstance(inner, np.ndarray) and inner.dtype != object:
                                flat = inner.flatten()
                                print(f"        *** NUMERIC DATA: {len(flat)} samples, first 3: {flat[:3]}")
        elif cell.dtype != object:
            flat = cell.flatten()
            print(f"    Numeric: {len(flat)} samples, first 3: {flat[:3]}")
    
    if speed_idx >= 1:
        break
