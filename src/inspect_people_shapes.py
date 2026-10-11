import scipy.io as sio, json, os, numpy as np, sys

mat_path = r"c:/Users/2005g/Downloads/4691144/People.mat"
out_path = r"c:/Users/2005g/Downloads/4691144/people_structure_summary.json"

def safe_len(x):
    try:
        return int(len(x))
    except Exception:
        return None

try:
    mat = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    # Identify main variable (non-__ key)
    var_name = next((k for k in mat.keys() if not k.startswith('__')), None)
    if var_name is None:
        raise RuntimeError('No primary variable found')
    people = mat[var_name]
    summary = {
        'variable_name': var_name,
        'people_count': safe_len(people),
        'persons': []
    }
    for p_idx, person in enumerate(people):
        # each person should have Sen attribute (numpy.ndarray of sensors)
        sen = getattr(person, 'Sen', None)
        if sen is None:
            continue
        person_rec = {
            'person_index': p_idx+1,
            'sensor_count': safe_len(sen),
            'sensors': []
        }
        # sen is an array of length 5 (sensors)
        for s_idx, sensor in enumerate(sen):
            sensor_rec = {'sensor_index': s_idx+1, 'speeds': {}}
            for speed in range(1,9):
                field = f'S{speed}'
                if not hasattr(sensor, field):
                    continue
                arr = getattr(sensor, field)
                # arr expected shape (num_traces, 3) where 3 amplifications
                shape = getattr(arr, 'shape', None)
                sensor_rec['speeds'][field] = {
                    'shape': shape if shape is not None else None,
                    'num_traces': shape[0] if shape else None,
                    'num_amplifications': shape[1] if shape else None
                }
            person_rec['sensors'].append(sensor_rec)
        summary['persons'].append(person_rec)
    # write JSON
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print('Summary written to', out_path)
except Exception as e:
    print('Error:', e)
    sys.exit(1)
