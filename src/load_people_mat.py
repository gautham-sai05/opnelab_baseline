import scipy.io as sio, json, os, sys, numpy as np

mat_path = r"c:/Users/2005g/Downloads/4691144/People.mat"

try:
    mat = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    # Top-level keys
    top_keys = list(mat.keys())
    # Find the primary variable (likely 'People')
    primary = None
    for k in top_keys:
        if not k.startswith('__'):
            primary = k
            break
    result = {'top_keys': top_keys, 'primary_key': primary}
    if primary and primary in mat:
        people = mat[primary]
        # Examine type and shape
        result['people_type'] = str(type(people))
        try:
            length = len(people)
            result['people_len'] = length
        except Exception:
            result['people_len'] = None
        # Inspect first person
        if isinstance(people, (list, tuple, np.ndarray)) and len(people) > 0:
            first = people[0]
            attrs = [a for a in dir(first) if not a.startswith('_')]
            result['first_person_attrs'] = attrs
            # Attempt to access Sen field if exists
            if hasattr(first, 'Sen'):
                sen = first.Sen
                result['sen_type'] = str(type(sen))
                if isinstance(sen, (list, tuple, np.ndarray)):
                    result['sen_len'] = len(sen)
        # Save JSON for inspection
    out_path = os.path.join(os.path.dirname(mat_path), 'people_mat_inspection.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print('Inspection written to', out_path)
except Exception as e:
    print('Error loading People.mat:', e)
    sys.exit(1)
