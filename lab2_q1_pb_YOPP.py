import numpy as np
import matplotlib.pyplot as plt
"""
==========================================
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
M. Catherine Yopp - 6 March 2022
MA448 - Project 2: Part (b)
    Analyzing the Boundary Value Problem 
        x' = (4 - 2y)/t^3
        y' = -e^x
        x(1) = 0, y(2) = 0
    by solving the Initial Value Problem 
        x' = (4 - 2y)/t^3
        y' = -e^x
        x(0) = 1, y(0) = s
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
==========================================

"""

def f(t,y): 
    f1 = (4 - 2*y[1])/t**3
    f2 = -np.exp(y[0])
    dydt=[f1,f2]
    
    return dydt

def exact(t):
    x = np.log(t)
    y = 2 - (t**2)*.5
    real = [x, y]
    return real

def rk4sys(rhs,y,t): 
    M = len(y)
    N = len(t)    
    Y = np.zeros((N,M)) 
    
    Y[0,:] = y
    h = (t[-1]-t[0])/(N-1)
    for n in range(N-1):       
        K1 = rhs(t[n],Y[n,:] )
        K2 = rhs(t[n] + .5*h, Y[n,:] \
                 + np.multiply(K1, h*.5))
        K3 = rhs(t[n] + .5*h, Y[n,:] \
                 + np.multiply(K2, h*.5))
        K4 = rhs(t[n] + h, Y[n,:] \
                 + np.multiply(K3, h))
        phi = np.multiply(K1 + np.multiply(K2,2) \
                          + np.multiply(K3, 2)\
                              + K4, 1/6)
        Y[n+1, :] = Y[n,:] + np.multiply(phi,h)
             
    return Y
#################################################
        
if __name__ == '__main__':  
    a = 1
    b = 2
    alpha = 0
    beta = 0
    nsteps = 11

    
    s0 = 1
    s1 = 2
        
    y0 = [alpha, s0]
    y1 = [alpha, s1]
    
    
    
    t = np.linspace(a,b,nsteps)  
    
    phi0 = rk4sys(f,y0,t)    
    phi1 = rk4sys(f,y1,t)
 
    Fs0 = beta - phi0[-1,1]
    Fs1 = beta - phi1[-1,1] 
    print(" i\t\t sn\t    beta_-phi(sn(b)\n")
    print(' {0:.0f} & \t  {1:.10f}\t & '\
          '{2: .10f}\\\\'.format(0, s0, Fs0))
    print('\\hline')
    print(' {0:.0f} & \t  {1:.10f}\t '\
          '&{2: .10f}\\\\'.format(0, s1, Fs1))
    print('\\hline')
    

    Ea = 1
    tol = 10**-8 
    
    yexact = np.zeros((nsteps,2))
    for n in range(nsteps):
        yexact[n,:] = exact(t[n])
    
    i = 0
    
    while Ea > tol:
        s2 = s1 + (((beta - phi1[-1,1]))*\
          (s1 - s0)/(phi1[-1,1] - phi0[-1,1]))
        
        Ea = abs(s2 - s1)/abs(s2)
        s0 = s1
        s1 = s2 
        i += 1
        y0 = [alpha, s0]
        y1 = [alpha, s1]
        
        phi0 = rk4sys(f,y0,t)
        phi1 = rk4sys(f,y1,t)

        Fs0 = beta - phi0[-1,1]
        Fs1 = beta - phi1[-1,1]
        print('{0: .0f}\t & {1: .10f}\t '\
              '& {2: .10f}'.format(i, s2, Fs1)) 
        print('\\hline')
        
    
   
    yexact = np.zeros((nsteps,2))
    for i in range(nsteps):
        yexact[i,:] = exact(t[i])
    
    plt.plot(t,phi1[:,0],'o-',t, yexact[:,0],'-')
    plt.legend(['numerical sol', 'exact sol'], \
               loc='best')
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title('The Numerical Solution to the BVP')
    plt.grid()
    plt.show()
    
    plt.plot(t,phi1[:,1],'o-',t, yexact[:,1],'-')
    plt.legend(['numerical sol', 'exact sol']\
               , loc='best')
    plt.xlabel("t")
    plt.ylabel("y")
    plt.title('The Numerical Solution to the BVP')
    plt.grid()
    plt.show()
    
    print("====================================='\
          '=======================================\n")  
    print("       h        |(y(h)-Y(h)/y(h/2)-Y(h/2)| '\
          '|(Y(h)-Y(h/2)/Y(h/2)-Y(h/4)|   ",end='\n')
    for n in range(1,12):
        N = 2**n
        y1 = [alpha, s1]
        
        t , h = np.linspace(a,b,N, retstep = True)
        Y1 = rk4sys(f, y1, t)   
        
        time = np.linspace(a,b, 2*N)
        Y2 = rk4sys(f, y1, time)   
        
        error_ratio1 = abs(exact(t[1]) - \
                           Y1[1])/abs(exact(time[1])-Y2[1])
    
        time_t = np.linspace(a,b, 4*N)
        Y4 = rk4sys(f, y1, time_t)
        error_ratio2 = abs(Y1[-1]-Y2[-1])/abs(Y2[-1]-Y4[-1])
        print('\t{0:0.6f} &\t\t\t{1:0.15f} &\t \t '\
              '\t {2:0.15f}\\\\'.format\
                  (h, error_ratio1[1], error_ratio2[1]))
        print('\\hline')
    print("========================================='\
          '===================================\n") 