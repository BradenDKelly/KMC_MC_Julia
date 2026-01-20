import math
N=864
rho=0.7
L=(N/rho)**(1/3)
rc=L/2
invr2=(1/rc)**2
invr6=invr2**3
invr12=invr6**2
u_rc=4*(invr12-invr6)
npairs=N*(N-1)/2
print('L',L,'rc',rc,'u_rc',u_rc,'npairs',npairs,'npairs*u_rc',npairs*u_rc)
