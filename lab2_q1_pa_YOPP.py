import numpy as np
import matplotlib.pyplot as plt
"""
==========================================
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
M. Catherine Yopp - 6 March 2022
MA448 - Project 2: Part (a)
    Analyzing the Boundary Value Problem 
        y" = -2y^2 + 8t^2y^3
        y(0) = 1, y(1) = 1/2
    by solving the Initial Value Problem 
        y" = -2y^2 + 8t^2y^3
        y(0) = 1, y'(0) = s
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
==========================================

"""

def f(t,y): 
    f1 = y[1]
    f2 = (-2*y[0]**2) + ((8*t**2)*(y[0]**3))
    dydt=[f1,f2]
    
    return dydt

def exact(t):
    return 1/ (1 + t**2)

def rk4sys(rhs,y,t): 
    M = len(y)
    N = len(t)    
    Y = np.zeros((N,M))
    
    Y[0,:] = y
    h = (t[-1]-t[0])/(N-1)
    for n in range(N-1):       
        K1 = rhs(t[n],Y[n,:] )
        K2 = rhs(t[n] + .5*h, Y[n,:] +  \
                 np.multiply(K1, h*.5))
        K3 = rhs(t[n] + .5*h, Y[n,:]  + \
                 np.multiply(K2, h*.5))
        K4 = rhs(t[n] + h, Y[n,:]  + \
                 np.multiply(K3, h))
        phi = np.multiply(K1 + np.multiply(K2,2)\
                + np.multiply(K3, 2) + K4, 1/6)
        Y[n+1, :] = Y[n,:] + np.multiply(phi,h)
             
    return Y
###########################################
        
if __name__ == '__main__':  
    a = 0
    b = 1
    alpha = 1
    beta = .5
    nsteps = 11
    
    s0 = 0
    s1 =1
        
    y0 = [alpha, s0]
    y1 = [alpha, s1]
    
    t = np.linspace(a,b,nsteps)  
    
    phi0 = rk4sys(f,y0,t)    
    phi1 = rk4sys(f,y1,t)

    Fs0 = beta - phi0[-1,0]
    Fs1 = beta - phi1[-1,0] 
    print("i\t\t sn\t\tbeta_-phi(sn(b)\n")
    print('{0:.0f}\t {1:.6f}\t {2: .6f}'\
          .format(0, s0, Fs0))
    print('{0:.0f}\t {1:.6f}\t {2: .6f}'\
          .format(0, s1, Fs1))
    
  
    Ea = 1
    tol = 10**-8 
    i = 0
    
    while Ea > tol:
        s2 = s1 + (((beta - phi1[-1,0]))* \
                   (s1 - s0)/(phi1[-1,0] - phi0[-1,0]))
     
        Ea = abs(s2 - s1)
        s0 = s1
        s1 = s2 
        i += 1
        y0 = [alpha, s0]
        y1 = [alpha, s1]
        
        phi0 = rk4sys(f, y0, t)        
        phi1 = rk4sys(f, y1, t)
          
        Fs0 = beta - phi0[-1,0]
        Fs1 = beta - phi1[-1,0]
        print('{0: .0f}\t {1: .6f}\t {2: .6f}'\
              .format(i, s2, Fs1)) 
        print(s1)
        
    yexact = np.zeros(nsteps)
    for i in range(nsteps):
        yexact[i] = exact(t[i])
    plt.plot(t, yexact, '-', t, phi1[:, 0], 'o')
    plt.legend(['exact', 'numericcal solution']\
               , loc='best')
    plt.xlabel('t')
    plt.ylabel('solution values')
    plt.title('RK 4 Approximation: \n'\
              'Final IVP Solution y')
    plt.grid()
    plt.show()
    
    print("============================"\
          "============================="\
          "===================\n")  
    print("       h        '\
          '|(y(h)-Y(h)/y(h/2)-Y(h/2)|   "\
          "|(Y(h)-Y(h/2)/Y(h/2)-Y(h/4)|   ",end='\n')
    for n in range(1,12):
        N = 2**n
        y1 = [alpha, s1]
        
        t , h = np.linspace(a,b,N, retstep = True)
        Y1 = rk4sys(f, y1, t)   
        
        time = np.linspace(a,b, 2*N)
        Y2 = rk4sys(f, y1, time)   
        
        error_ratio1 = abs(exact(t[-1]) - Y1[-1][0])\
            /abs(exact(t[-1])-Y2[-1][0])
    
        time_t = np.linspace(a,b, 4*N)
        Y4 = rk4sys(f, y1, time_t)
        error_ratio2=abs(Y1[-1][0]-Y2[-1][0])/\
            abs(Y2[-1][0]-Y4[-1][0])
        print('\t{0:0.6f}\t\t\t{1:0.15f}\t \t \t '\
              '{2:0.15f}'.format(h,error_ratio1,error_ratio2))
    print("======================================'\
          ======================================\n") 
    