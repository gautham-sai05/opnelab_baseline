"""
Deep investigation of People.mat structure.
It's MATLAB v5 format. 'people' is (1,10) object array.
Each element is itself an ndarray. Need to understand the nesting.
"""
import scipy.io as sio
import numpy as np
import sys

PEOPLE_MAT = r'C:\Users\2005g\Downloads\4691144\People.mat'

mat = sio.loadmat(PEOPLE_MAT, struct_as_record=True, squeeze_me=False)
people = mat['people']
print(f"people: shape={people.shape}, dtype={people.dtype}")

for person_idx in range(people.shape[1]):
    person = people[0, person_idx]
    print(f"\n=== Person {person_idx} ===")
    print(f"  type={type(person)}, shape={getattr(person, 'shape', 'N/A')}, dtype={getattr(person, 'dtype', 'N/A')}")
    
    if isinstance(person, np.ndarray):
        if person.dtype.names:
            print(f"  Structured array with fields: {person.dtype.names}")
            for fn in person.dtype.names:
                field = person[fn]
                print(f"    '{fn}': type={type(field)}, shape={field.shape}, dtype={field.dtype}")
                if field.dtype == object and field.size > 0:
                    inner = field.flat[0]
                    print(f"      First element: type={type(inner)}, shape={getattr(inner, 'shape', 'N/A')}, dtype={getattr(inner, 'dtype', 'N/A')}")
                    if isinstance(inner, np.ndarray) and inner.dtype.names:
                        print(f"      Struct fields: {inner.dtype.names}")
                        for ifn in inner.dtype.names:
                            ifield = inner[ifn]
                            print(f"        '{ifn}': type={type(ifield)}, shape={getattr(ifield, 'shape', 'N/A')}, dtype={getattr(ifield, 'dtype', 'N/A')}")
                    elif isinstance(inner, np.ndarray) and inner.dtype == object:
                        print(f"      Another object array shape={inner.shape}")
                        if inner.size > 0:
                            inner2 = inner.flat[0]
                            print(f"        inner2: type={type(inner2)}, shape={getattr(inner2, 'shape', 'N/A')}, dtype={getattr(inner2, 'dtype', 'N/A')}")
                            if isinstance(inner2, np.ndarray) and inner2.dtype != object:
                                flat = inner2.flatten()
                                print(f"        First 5 values: {flat[:5]}")
                                print(f"        Total samples: {len(flat)}")
                    elif isinstance(inner, np.ndarray) and inner.dtype != object:
                        flat = inner.flatten()
                        print(f"      First 5 values: {flat[:5]}")
                        print(f"      Total samples: {len(flat)}")
        elif person.dtype == object:
            print(f"  Object array with shape {person.shape}")
            if person.size > 0:
                inner = person.flat[0]
                print(f"    First element: type={type(inner)}, shape={getattr(inner, 'shape', 'N/A')}, dtype={getattr(inner, 'dtype', 'N/A')}")
                if isinstance(inner, np.ndarray) and inner.dtype.names:
                    print(f"    Struct fields: {inner.dtype.names}")
                    for fn in inner.dtype.names:
                        field = inner[fn]
                        print(f"      '{fn}': shape={getattr(field, 'shape', 'N/A')}, dtype={getattr(field, 'dtype', 'N/A')}")
                        if isinstance(field, np.ndarray) and field.dtype == object and field.size > 0:
                            elem = field.flat[0]
                            print(f"        elem: type={type(elem)}, shape={getattr(elem, 'shape', 'N/A')}, dtype={getattr(elem, 'dtype', 'N/A')}")
                            if isinstance(elem, np.ndarray) and elem.dtype == object and elem.size > 0:
                                elem2 = elem.flat[0]
                                print(f"          elem2: type={type(elem2)}, shape={getattr(elem2, 'shape', 'N/A')}, dtype={getattr(elem2, 'dtype', 'N/A')}")
                                if isinstance(elem2, np.ndarray) and elem2.dtype != object:
                                    flat = elem2.flatten()
                                    print(f"          WAVEFORM? First 5: {flat[:5]}, total={len(flat)}")
                            elif isinstance(elem, np.ndarray) and elem.dtype != object:
                                flat = elem.flatten()
                                if len(flat) > 100:
                                    print(f"          WAVEFORM? First 5: {flat[:5]}, total={len(flat)}")
                elif isinstance(inner, np.ndarray) and inner.dtype == object:
                    print(f"    Still object array shape={inner.shape}")
                    if inner.size > 0:
                        inner2 = inner.flat[0]
                        print(f"      inner2: type={type(inner2)}, shape={getattr(inner2, 'shape', 'N/A')}, dtype={getattr(inner2, 'dtype', 'N/A')}")
        else:
            # Numeric array
            flat = person.flatten()
            print(f"  Numeric array: shape={person.shape}, first 5: {flat[:5]}")
    
    if person_idx >= 1:
        print("\n(Stopping after 2 persons for brevity)")
        break
