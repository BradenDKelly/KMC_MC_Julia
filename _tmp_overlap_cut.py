import math

def f(r):
    return 4*((1.0/r)**12 - (1.0/r)**6) - 100.0

lo, hi = 0.2, 1.2
flo, fhi = f(lo), f(hi)
if flo * fhi > 0:
    raise SystemExit('root not bracketed')

for _ in range(100):
    mid = 0.5*(lo+hi)
    fm = f(mid)
    if abs(fm) < 1e-12:
        break
    if flo*fm > 0:
        lo, flo = mid, fm
    else:
        hi, fhi = mid, fm

print(mid)
