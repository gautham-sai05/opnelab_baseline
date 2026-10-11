import scipy.io as sio, json, os, numpy as np, sys

mat_path = r"c:/Users/2005g/Downloads/4691144/People.mat"
out_path = r"c:/Users/2005g/Downloads/4691144/people_mat_deep.json"

try:
    mat = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    people = mat.get('people') or mat.get('People')
    if people is None:
        raise RuntimeError('People variable not found')
    result = {'people_len': int(len(people)), 'person_details': []}
    for idx, person in enumerate(people):
        # person is a mat_struct with attribute Sen
        sen = getattr(person, 'Sen', None)
        if sen is None:
            continue
        sen_len = int(len(sen))
        # sen is array of struct with fields S1..S8
        speed_fields = []
        for s_idx in range(1, 9):
            field_name = f'S{s_idx}'
            if hasattr(sen[0], field_name):
                speed_fields.append(field_name)
        # For each speed, we can inspect the shape of the cell array (trace x amplification)
        speed_info = {}
        for s_idx in range(1, 9):
            field = f'S{s_idx}'
            if not hasattr(sen[0], field):
                continue
            arr = getattr(sen[0], field)
            # arr is a numpy.ndarray of shape (num_traces, 3) where 3 amplifications
            try:
                shape = tuple(arr.shape)
            except Exception:
                shape = None
            speed_info[field] = {'shape': shape}
        result['person_details'].append({
            'person_index': idx+1,
            'sen_len': sen_len,
            'speed_fields': speed_fields,
            'speed_info': speed_info,
        })
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print('Deep inspection written to', out_path)
except Exception as e:
    print('Error:', e)
    sys.exit(1)
