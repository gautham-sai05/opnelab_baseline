"""
Phase 2A: Proper HDF5 structural investigation of People.mat
This script correctly handles MATLAB v7.3 (HDF5) files by resolving
object references rather than treating them as plain arrays.
"""
import h5py
import numpy as np
import os
import json
import sys

PEOPLE_MAT = r'C:\Users\2005g\Downloads\4691144\People.mat'

def inspect_hdf5(filepath):
    print(f"=== HDF5 Inspection of {filepath} ===")
    print(f"File size: {os.path.getsize(filepath) / 1e6:.1f} MB")
    
    with h5py.File(filepath, 'r') as f:
        print(f"\nTop-level keys: {list(f.keys())}")
        
        for key in f.keys():
            if key.startswith('#'):
                continue
            obj = f[key]
            print(f"\n--- Key: '{key}' ---")
            print(f"  Type: {type(obj)}")
            if isinstance(obj, h5py.Dataset):
                print(f"  Shape: {obj.shape}")
                print(f"  Dtype: {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"  Sub-keys: {list(obj.keys())}")
                # Recurse one level
                for sk in obj.keys():
                    sobj = obj[sk]
                    print(f"    '{sk}': type={type(sobj)}, ", end='')
                    if isinstance(sobj, h5py.Dataset):
                        print(f"shape={sobj.shape}, dtype={sobj.dtype}")
                    elif isinstance(sobj, h5py.Group):
                        print(f"sub-keys={list(sobj.keys())}")
        
        # Now specifically investigate 'people' if it exists
        if 'people' in f:
            people = f['people']
            print(f"\n=== Detailed investigation of 'people' ===")
            print(f"Type: {type(people)}")
            if isinstance(people, h5py.Dataset):
                print(f"Shape: {people.shape}, Dtype: {people.dtype}")
                # If it's an object reference array, resolve refs
                if people.dtype == h5py.ref_dtype or 'object' in str(people.dtype).lower():
                    print("Contains object references. Resolving...")
                    refs = people[()]
                    print(f"Refs shape: {refs.shape}")
                    # Resolve first reference
                    for i in range(min(refs.shape[0], 3)):
                        for j in range(min(refs.shape[1] if refs.ndim > 1 else 1, 3)):
                            if refs.ndim > 1:
                                ref = refs[i, j]
                            else:
                                ref = refs[i]
                            try:
                                target = f[ref]
                                print(f"  Ref [{i},{j}]: type={type(target)}")
                                if isinstance(target, h5py.Dataset):
                                    print(f"    Shape: {target.shape}, Dtype: {target.dtype}")
                                    if target.size < 20:
                                        print(f"    Data: {target[()]}")
                                elif isinstance(target, h5py.Group):
                                    print(f"    Keys: {list(target.keys())}")
                                    # Dig into the group
                                    for gk in target.keys():
                                        gobj = target[gk]
                                        print(f"      '{gk}': type={type(gobj)}, ", end='')
                                        if isinstance(gobj, h5py.Dataset):
                                            print(f"shape={gobj.shape}, dtype={gobj.dtype}")
                                            # If this is also references
                                            if gobj.dtype == h5py.ref_dtype:
                                                print(f"        Contains refs, resolving first...")
                                                grefs = gobj[()]
                                                print(f"        Refs shape: {grefs.shape}")
                                                # Resolve first inner ref
                                                for ii in range(min(grefs.size, 3)):
                                                    idx = np.unravel_index(ii, grefs.shape)
                                                    inner_ref = grefs[idx]
                                                    try:
                                                        inner_target = f[inner_ref]
                                                        print(f"          Inner ref {idx}: type={type(inner_target)}", end='')
                                                        if isinstance(inner_target, h5py.Dataset):
                                                            print(f", shape={inner_target.shape}, dtype={inner_target.dtype}")
                                                            # Print first few values
                                                            data = inner_target[()]
                                                            if data.size > 0:
                                                                flat = data.flatten()
                                                                print(f"            First 5 values: {flat[:5]}")
                                                                print(f"            Min={flat.min():.4f}, Max={flat.max():.4f}, Mean={flat.mean():.4f}")
                                                        elif isinstance(inner_target, h5py.Group):
                                                            print(f", keys={list(inner_target.keys())}")
                                                    except Exception as e:
                                                        print(f"          Failed to resolve inner ref: {e}")
                                            elif gobj.size < 10:
                                                print(f"        Data: {gobj[()]}")
                                        elif isinstance(gobj, h5py.Group):
                                            print(f"keys={list(gobj.keys())}")
                            except Exception as e:
                                print(f"  Ref [{i},{j}]: Failed to resolve: {e}")
            elif isinstance(people, h5py.Group):
                print(f"Keys: {list(people.keys())}")
                # Walk the whole group tree
                def walk_group(grp, prefix='', depth=0, max_depth=5):
                    if depth > max_depth:
                        return
                    for k in grp.keys():
                        obj = grp[k]
                        full = f"{prefix}/{k}"
                        if isinstance(obj, h5py.Dataset):
                            print(f"{'  '*depth}{full}: Dataset shape={obj.shape}, dtype={obj.dtype}")
                            if obj.dtype == h5py.ref_dtype and depth < 3:
                                refs = obj[()]
                                print(f"{'  '*(depth+1)}Object refs shape: {refs.shape}")
                        elif isinstance(obj, h5py.Group):
                            print(f"{'  '*depth}{full}: Group keys={list(obj.keys())}")
                            walk_group(obj, full, depth+1, max_depth)
                walk_group(people, 'people')

if __name__ == '__main__':
    if not os.path.exists(PEOPLE_MAT):
        print(f"ERROR: {PEOPLE_MAT} not found")
        sys.exit(1)
    
    # Check if it's actually HDF5
    with open(PEOPLE_MAT, 'rb') as f:
        magic = f.read(8)
        print(f"File magic bytes: {magic[:8]}")
        is_hdf5 = magic[:4] == b'\x89HDF'
        print(f"Is HDF5: {is_hdf5}")
    
    if is_hdf5:
        inspect_hdf5(PEOPLE_MAT)
    else:
        print("\nFile is NOT HDF5 (MATLAB v7.3). Trying scipy.io.loadmat...")
        import scipy.io as sio
        mat = sio.loadmat(PEOPLE_MAT, struct_as_record=False, squeeze_me=False)
        print(f"Top-level keys: {[k for k in mat.keys() if not k.startswith('__')]}")
        for key in mat.keys():
            if key.startswith('__'):
                continue
            obj = mat[key]
            print(f"\n'{key}': type={type(obj)}, shape={getattr(obj, 'shape', 'N/A')}, dtype={getattr(obj, 'dtype', 'N/A')}")
            if hasattr(obj, 'shape') and hasattr(obj, 'dtype'):
                if obj.dtype == object:
                    print(f"  Object array. Inspecting first elements...")
                    for i in range(min(obj.shape[0], 2)):
                        for j in range(min(obj.shape[1] if obj.ndim > 1 else 1, 2)):
                            if obj.ndim > 1:
                                item = obj[i, j]
                            else:
                                item = obj[i]
                            print(f"  [{i},{j}]: type={type(item)}")
                            if hasattr(item, '_fieldnames'):
                                print(f"    Fields: {item._fieldnames}")
                                for fn in item._fieldnames:
                                    field = getattr(item, fn)
                                    print(f"      '{fn}': type={type(field)}, shape={getattr(field, 'shape', 'N/A')}, dtype={getattr(field, 'dtype', 'N/A')}")
                                    if hasattr(field, 'dtype') and field.dtype == object:
                                        # Go deeper
                                        if field.size > 0:
                                            inner = field.flat[0]
                                            print(f"        First element: type={type(inner)}, shape={getattr(inner, 'shape', 'N/A')}, dtype={getattr(inner, 'dtype', 'N/A')}")
                                            if hasattr(inner, '_fieldnames'):
                                                print(f"        Fields: {inner._fieldnames}")
                                                for ifn in inner._fieldnames:
                                                    ifield = getattr(inner, ifn)
                                                    print(f"          '{ifn}': type={type(ifield)}, shape={getattr(ifield, 'shape', 'N/A')}, dtype={getattr(ifield, 'dtype', 'N/A')}")
                                                    if hasattr(ifield, 'dtype') and ifield.dtype != object and hasattr(ifield, 'shape'):
                                                        if ifield.ndim >= 1 and ifield.size > 5:
                                                            flat = ifield.flatten()
                                                            print(f"            First 5: {flat[:5]}, shape={ifield.shape}")
