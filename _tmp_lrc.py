import math
N=864
rho=0.7
L=(N/rho)**(1/3)
rc=L/2
rc3=rc**3
rc9=rc3**3
u_tail=(8*math.pi*rho/3)*((1/(3*rc9))-(1/rc3))
print('rc',rc,'u_tail',u_tail,'N*u_tail',N*u_tail)
