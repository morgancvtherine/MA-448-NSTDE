import numpy as np 
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp
from scipy.integrate import odeint


def f(t, x):
    L = x[0] 
    P = x[1] 
    Z = x[2]
    
    rhs = np.zeros(len(x)) 
    rhs[0] = -3.6*L + 1.2*(P*(1-P**2) - 1)
    rhs[1] = -1.2*P + 6*(L + (2/(1 + Z)))
    rhs[2] = -0.12*Z + 12*P 
    
    return rhs 


def rk4(f, x0, t): 
    N = len(x0)
    M = len(t)    
    x = np.zeros((M, N))
    
    x[0] = x0
    h = (t[-1]-t[0])/M
    
    for n in range(M-1):       
        K1 = f(t[n],x[n] )
        K2 = f(t[n] + .5*h, x[n] + np.multiply(K1, h*.5))
        K3 = f(t[n] + .5*h, x[n]  + np.multiply(K2, h*.5))
        K4 = f(t[n] + h, x[n]  + np.multiply(K3, h))
        phi = np.multiply(K1 + np.multiply(K2, 2) + np.multiply(K3, 2) + K4, 1/6)
        x[n+1] = x[n] + np.multiply(phi,h)   
       
    return x

def rk4_once (tn, yn, h):
    
     K1 = f(tn, yn)
     K2 = f(tn + .5*h, yn + np.multiply(K1, .5*h))
     K3 = f(tn + .5*h, yn + np.multiply(K2,.5*h))
     K4 = f(tn + h, yn + np.multiply(K3, h))
    
     ynew = yn + np.multiply((K1 + np.multiply(K2,2) + np.multiply(K3,2) + K4), h/6)
     return ynew

def abm4(f, x0, t):
    N = len(x0)
    M = len(t)    
    x = np.zeros((M, N))
    x[0] = x0
    
    h = (t[-1]-t[0])/M
    
    for n in range(3):
        x[n+1] = rk4_once(t[n], x[n], h)        
     
    for n in range(3,M-1):
       F1 = f(t[n],x[n])
       F2 = f(t[n-1],x[n-1])
       F3 = f(t[n-2],x[n-2])
       F4 = f(t[n-3],x[n-3])
                 
       x[n+1] = x[n] + np.multiply(np.multiply(F1, 55) - np.multiply(F2, 59) \
                    + np.multiply(F3, 37) - np.multiply(F4, 9), h/24)
       G1 = f(t[n+1], x[n+1])
       G2 = F1 
       G3 = F2 
       G4 = F3 
        
       x[n+1] = x[n] +  np.multiply(np.multiply(G1, 9) \
            + np.multiply(G2, 19) - np.multiply(G3, 5) + G4, h/24)
        
    return x

if __name__ == '__main__':

    x0 = [0, 0, 0]
    T = 21 # years 
    N = 15001
    
    t, h = np.linspace(0, T, N, retstep = True)
    x_rk4 = rk4(f, x0, t)
    x_abm4 = abm4(f, x0, t)
   
    # RK4 Method 
    L = x_rk4[:, 0]
    P = x_rk4[:, 1]
    Z = x_rk4[:, 2] 
    
    fig = plt.figure()
    plt.plot(L, P, '-')
    plt.title("Petrarch's Love Relative to Laura's \n Fourth-order Runge-Kutta")
    plt.xlabel("Laura's Love")
    plt.ylabel("Petrarch's Love")
    plt.grid()
    plt.show()
    fig.savefig("L_P.png")
    
    fig = plt.figure()
    plt.plot(P, Z, '-')
    plt.title("Petrarch's inspiration realitive to his love \n Fourth-order Runge-Kutta")
    plt.xlabel("Petrarch's love")
    plt.ylabel("Petrarch's inspo")
    plt.grid()
    plt.show()
    fig.savefig("P_Z.png")
    
    fig = plt.figure()
    plt.plot(t, L, '-.', t, P, '-.', t, Z, '-.')
    plt.title("Laura and Petarch's Cylical Dynamic of Love \n Fourth-order Runge-Kutta")
    plt.xlabel("time (years)")
    plt.ylabel("love & inspiration")
    plt.legend(["Laura's love", "Petrarch's love", "Petrarch's inspo"], loc = 'best')
    plt.grid()
    plt.show()
    fig.savefig("LP_love_rk4.png")
    
    # ABM4 Method
    L = x_abm4[:, 0]
    P = x_abm4[:, 1]
    Z = x_abm4[:, 2] 
    
    plt.plot(L, P, '-')
    plt.title("Petrarch's Love Relative to Laura's \n Fourth-order Adam Bashforth-Moulton")
    plt.xlabel("Laura's Love")
    plt.ylabel("Petrarch's Love")
    plt.grid()
    plt.show()
    
    plt.plot(P, Z, '-')
    plt.title("Petrarch's inspiration realitive to his love \n Fourth-order Adam Bashforth-Moulton")
    plt.xlabel("Petrarch's love")
    plt.ylabel("Petrarch's inspo")
    plt.grid()
    plt.show()
    
    fig = plt.figure()
    plt.plot(t, L, '-.', t, P, '-.', t, Z, '-.')
    plt.title("Laura and Petarch's Cylical Dynamic of Love \n Fourth-order Adam Bashforth-Moulton")
    plt.xlabel("time (years)")
    plt.ylabel("love & inspiration")
    plt.legend(["Laura's love", "Petrarch's love", "Petrarch's inspo"], loc = 'best')
    plt.grid()
    plt.show()
    fig.savefig("LP_love_abm4.png")
    
    # Python's in-built ode solvers
    tspan=[0,21] 
    x = solve_ivp(f, tspan, x0, method='RK45',t_eval = t) #LSODA, BDF
    fig = plt.figure()
    plt.plot(x.t, x.y[0], '-', x.t, x.y[1], '-', x.t, x.y[2], '-')
    plt.title("Laura and Petarch's Cylical Dynamic of Love \n solve_ivp")
    plt.xlabel("time (years)")
    plt.ylabel("love & inspiration")
    plt.legend(["Laura's love", "Petrarch's love", "Petrarch's inspo"], loc = 'best')
    plt.grid()
    plt.show()
    fig.savefig("LP_love_solve_ivp.png")
    
    fig = plt.figure()
    plt.plot(x.t, x.y[1], '-', x.t, x.y[2], '-')
    plt.title("Petrarch's Love and Inspiration Over the Years")
    plt.xlabel("time (years)")
    plt.ylabel("love & inspiration")
    plt.legend(["Petrarch's love", "Petrarch's inspo"], loc = 'best')
    plt.grid()
    plt.show()
    fig.savefig("loveninspo.png")
    
    fig = plt.figure()
    plt.plot(x.t, x.y[1], '-')
    plt.title("Petrarch's Love Over the Years")
    plt.xlabel("time (years)")
    plt.ylabel("love")
    plt.grid()
    plt.show()
    fig.savefig("pets_love.png")
    
    fig = plt.figure()
    x1 = odeint(f, x0,t,tfirst =True)
    plt.plot(x.t, x1[:,0], '-', x.t, x1[:,1], '-', x.t, x1[:,2], '-')
    plt.title("Laura and Petarch's Cylical Dynamic of Love \n odeint")
    plt.xlabel("time (years)")
    plt.ylabel("love & inspiration")
    plt.legend(["Laura's love", "Petrarch's love", "Petrarch's inspo"], loc = 'best')
    plt.grid()
    plt.show()
    fig.savefig("LP_love_odeint.png")
    
    fig = plt.figure()
    x1 = odeint(f, x0,t,tfirst =True)
    plt.plot(x.t, x1[:,0], '-', x.t, x1[:,1], '-')
    plt.title("Laura and Petarch's Love Over the Years")
    plt.xlabel("time (years)")
    plt.ylabel("love & inspiration")
    plt.legend(["Laura's love", "Petrarch's love"], loc = 'best')
    plt.grid()
    plt.show()
    fig.savefig("all_love.png")