import re

def postprocess(a, et):
    rs, vs = a
    newrs, newvs = [], []
    ref = 0
    for (i, (r, v)) in enumerate(zip(rs, vs)):
        r = int(r)
        if i == len(rs) - 1:
            newrs += list(range(r, et))
            newvs += [v for _ in range(r, et)]
        else:
            next_r = int(rs[i + 1])
            newrs += list(range(r, next_r))
            newvs += [v for _ in range(r, next_r)]
    return newrs, newvs

def zero_cost(txt):
    z_pattern = f"(.*): t = -1, r = ([0-9]*), c = ([0-9.]*)"
    z_res = results = list(zip(*re.findall(z_pattern, txt)))
    zerocost = float(z_res[2][0])
    return zerocost

def convert_to_dict(txt):
    z_pattern = f"(.*): t = -1, r = ([0-9]*), c = ([0-9.]*)"
    z_res = results = list(zip(*re.findall(z_pattern, txt)))
    zerocost = float(z_res[2][0])
    pattern = f"(.*): t = [0-9]*, r = ([0-9]*), c = ([0-9.]*)"
    results = list(zip(*re.findall(pattern, txt)))
    return [[0., 1.] + [float(r) for r in results[1]], [zerocost, zerocost] + [float(r) for r in results[2]]]

def process_for_plot(txt, et):
    a = convert_to_dict(txt)
    return postprocess(a, et)