import scipy.io as sio, json, os, numpy as np, sys

mat_path = r"c:/Users/2005g/Downloads/4691144/People.mat"
out_path = r"c:/Users/2005g/Downloads/4691144/people_structure.json"

def safe_len(x):
    try:
        return int(len(x))
    except Exception:
        return None

try:
    mat = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    # Determine variable name (case-insensitive)
    var_name = next((k for k in mat.keys() if not k.startswith('__')), None)
    if not var_name:
        raise RuntimeError('No data variable found')
    people = mat[var_name]
    report = {
        'variable_name': var_name,
        'people_len': safe_len(people),
        'persons': []
    }
    for p_idx, person in enumerate(people):
        # Each person should have attribute 'Sen'
        sen = getattr(person, 'Sen', None)
        if sen is None:
            continue
        person_info = {
            'person_index': p_idx+1,
            'num_sensors': safe_len(sen),
            'sensors': []
        }
        for s_idx, sensor in enumerate(sen):
            sensor_info = {'sensor_index': s_idx+1, 'speeds': {}}
            for speed in range(1,9):
                field_name = f'S{speed}'
                if not hasattr(sensor, field_name):
                    continue
                arr = getattr(sensor, field_name)
                # Expected shape (num_traces, 3) where 3 = amplification levels
                shape = getattr(arr, 'shape', None)
                sensor_info['speeds'][field_name] = {
                    'shape': shape if shape is not None else None,
                    'num_traces': shape[0] if shape else None,
                    'num_amplifications': shape[1] if shape else None
                }
            person_info['sensors'].append(sensor_info)
        report['persons'].append(person_info)
    # Write JSON
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print('Structure saved to', out_path)
except Exception as e:
    print('Error loading People.mat:', e)
    sys.exit(1)
