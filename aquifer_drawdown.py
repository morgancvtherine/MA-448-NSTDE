import numpy as np 
import matplotlib.pyplot as plt 

def ftcs(u1, dr, dt, D, r_in, r_out, alpha, beta, p, q, rho, s):
    N = len(u1)
    u = np.zeros(N)
    
    mu = D*dt/dr
    nu = D*dt/dr**2 
    
    # Updating the internal spacial nodes
    for i in range(1, N-1):
        rj = r_in + i*dr
        u[i] = (nu-.5*mu/rj)*u1[i-1] + (1 - 2*nu - dt*beta)*u1[i] + (nu + .5*mu/rj)*u1[i+1] + dt*s 
    
    # inserting the boundary conditions 
    u[0] = (1 - 2*nu - dt*beta)*u1[0] + 2*nu*u1[1] + dt*s - 2*dr*alpha*(nu - .5*mu/r_in) 
    u[N-1] = 2*nu*u1[N-2] + (1 - 2*nu - dt*beta - 2*dr*p*(nu + .5*mu/r_out)/q)*u1[N-1]  + dt*s + 2*dr*rho*(nu + .5*mu/r_out)/q
    
    return u 
#####################################################################################

if __name__ == "__main__":
    T = 150
    S = .2
    D = T/S 
    Q = 100
    w = .003 
    source = 0
    beta = 0 
    p = 0 
    q = 1 
    rho = 0 
    alpha = Q/(np.pi*T)
    t0 = 0 
    tend = 15 # days 
    rin = .5 
    rout = 100 
    
    N = 6 # number steps in radial direction 
    M = 100 # number of time steps 
    out_num = M // 10
    dr = (rout - rin)/N 
    dt = tend/M 
    
    h = np.zeros(N+1) 
    sol = np.zeros((M+1,N+1))
    
    for i in range(N+1): 
        h[i] = 0
        sol[0,i] = h[i]

                
    P=M//out_num
    diff=P-1 
    for n in range(1,M+1): # begin time stepping loop   
        h = ftcs(h, dr, dt, D, rin, rout, alpha, beta, p, q, rho, source)
        if np.mod(n,P)==0 or n==M:
            for i in range(0,N+1):
                sol[n-diff,i] = h[i] 
            diff = diff+P-1 
            
    t = np.linspace(t0, tend, M+1)
    r = np.linspace(rin, rout, N+1)
    for n in range(0,len(sol[:,0])):
        if n==0:
            plt.plot(r,sol[n,:],'o-')
            plt.legend(['t=0'],loc='best')
            plt.xlabel(r'$r$ radius')
            plt.ylabel(r'$h(r,t)$')
            plt.title(r'Numerical Solution to $\frac{\partial u}{\partial t} = \frac{\partial^2 u}{\partial x^2}$')
            plt.grid()
        else:
            plt.plot(r,sol[n,:],'-o')
    plt.show()
