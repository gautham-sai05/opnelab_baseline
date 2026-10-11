import h5py, json, os, numpy as np, scipy.io as sio

mat_path = r"c:/Users/2005g/Downloads/4691144/People.mat"

# Helper to recursively explore HDF5 groups

def explore(name, obj, depth=0, out=None):
    if out is None:
        out = []
    indent = "  " * depth
    if isinstance(obj, h5py.Group):
        out.append(f"{indent}Group: {name}/")
        for k, v in obj.items():
            explore(k, v, depth+1, out)
    elif isinstance(obj, h5py.Dataset):
        shape = obj.shape
        dtype = obj.dtype
        # Detect if dataset is a reference (object or region)
        if obj.dtype.kind == 'O':
            out.append(f"{indent}Dataset (ref): {name}, shape={shape}, dtype={dtype}")
            # Try to resolve first reference
            try:
                ref = obj[0]
                if isinstance(ref, h5py.Reference):
                    target = obj.file[ref]
                    out.append(f"{indent}  -> points to {target.name}, shape={target.shape}, dtype={target.dtype}")
            except Exception:
                pass
        else:
            out.append(f"{indent}Dataset: {name}, shape={shape}, dtype={dtype}")
    return out

with h5py.File(mat_path, 'r') as f:
    structure = explore('/', f, depth=0)
    # Print a summary of top-level keys
    top_keys = list(f.keys())
    summary = {
        "top_keys": top_keys,
        "structure": structure[:200]  # first 200 lines to avoid huge output
    }
    # Save to JSON for easier reading
    out_path = os.path.join(os.path.dirname(mat_path), "people_mat_structure.json")
    with open(out_path, "w") as jf:
        json.dump(summary, jf, indent=2)
    print("Structure written to", out_path)

# Also load with scipy.io to see what keys it exposes
try:
    mat = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    keys = list(mat.keys())
    print("scipy.io.loadmat keys:", keys)
    # Show type and maybe shape of People variable if present
    if 'People' in mat:
        people = mat['People']
        print("People type:", type(people))
        if isinstance(people, np.ndarray):
            print("People array shape:", people.shape)
except Exception as e:
    print("loadmat error:", e)
