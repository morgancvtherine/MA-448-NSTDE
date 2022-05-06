import numpy as np
import matplotlib.pyplot as plt

"""
M. Catherine Yopp - 5 Feb 2022
MA 448 - Lab 1 

THe following code analyzes the Kermack-McKendrick Epidemic Model 
in two forms: 
    (1) H' = -cHI, I' = cHI - mI, D' = -mI
    (2) D' = m(N - D - H0e^(-cD/m)), H = H0e^(-cD/m), I = N - H - D
    
Where H = number of healthy individuals, I =  number of infected individuals, 
D = number of dead as functions of time, c is the transmission rate of disease 
to healthy individuals andm is the mortality rate of infected individuals. 

"""


def f(t,x):
    
    f1 = -c*x[0]*x[1] #the healthy over time 
    f2 = (c*x[0]*x[1]) - (m*x[1]) #the infected over time 
    f3 = m*x[1] #the deceased over time     
    
    dxdt = [f1, f2, f3]
    return dxdt

def rk4sys(f, time, x):
    
    M = len(time) 
    N = len(x) 
    X = np.zeros((M,N))
    
    h = (time[-1] + time[0])/M 
    X[0,:] = x 
    
    for i in range(M -1):
        K1 = f(time[i], X[i,:] )
        K2 = f(time[i] + .5*h, X[i,:] + np.multiply(K1, h*.5))
        K3 = f(time[i] + .5*h, X[i,:]  + np.multiply(K2, h*.5))
        K4 = f(time[i] + h, X[i,:]  + np.multiply(K3, h))
        phi = np.multiply(K1 + np.multiply(K2,2) +  
                          np.multiply(K3, 2) + K4, 1/6)
        
        X[i+1, :] = X[i,:] + np.multiply(phi, h)
        
        if X[i+1,1] < ideal_infec :
            
            X.resize((i+1,N))
            newt = np.zeros(i+1)
            
            for j in range(i+1):
                newt[j] = time[j]
            break
    
    return newt, X
    
def dydt(time,y): 
    function = m*(cits - y - (H0*np.exp(-c*y/m)))
    return function

def rk4(t0, tmax, y0, steps):
    trk,h = np.linspace(t0,tmax,steps, retstep = True) 
    y = np.zeros(steps)
    H = np.zeros(steps)
    I = np.zeros(steps)
    y[0] = y0
    H[0] = H0 
    I[0] = I0 
         
    for n in range(steps-1):
          K1 = dydt(trk[n], y[n],) 
          K2 = dydt(trk[n] + .5*h, y[n] + .5*h*K1)
          K3 = dydt(trk[n] + .5*h, y[n] + .5*h*K2)
          K4 = dydt(trk[n] + h, y[n] + h*K3)
             
          y[n+1] = y[n] + ((h/6) * (K1 + (2*K2) + (2*K3) + K4))
          
          H[n+1] = H0 * np.exp(-c*y[n+1]/m)
          I[n+1] = cits - y[n+1] - H[n+1] 
          
          if I[n+1] < ideal_infec:
              y.resize(n+1)
              H.resize(n+1)
              I.resize(n+1)              
              newtrk = np.zeros(n+1)
              for j in range(n+1):
                  newtrk[j] = trk[j]
              break
                     
    return newtrk, y, H, I 
#############################################################################
if __name__ == "__main__": 
    
    # The triple ode system
    t0 = 0
    tmax = 20 #weeks
    steps = 1000
    #the solution converges by this many steps 
    time = np.linspace(t0, tmax, steps)
    
    m = 1.8 #the mortality rate of the infected per week 
    c = .001 #the transmission rate of disease to healthy individuals per week
    ideal_infec = 1
    x0 = [3350, 150, 0]
    [t, xsol] = rk4sys(f, time, x0)
      
    plt.plot(t,xsol[:,0],'g-',t,xsol[:,1] ,'y-',t,xsol[:,2], 'r-')
    plt.title("Disease's Effects on Citizens\n c = {0:.4f} & m = {1:.1f}\n \
               Triple ODE System".format(c,m), loc = 'center')
               
    plt.legend(['Healthy','Infected', 'Deceased'], loc = 'best')
    plt.xlabel('# of weeks')
    plt.ylabel('Affected Villagers')
    plt.grid() 
    plt.show()

    week = abs((t0 - t[-1])/len(t))
    print('week\t\thealthy\t\t infected\t   deceased\t\t')
    for k in range(len(t) -1):
        print('{0: .1f}\t\t{1: .0f}\t\t\t{2: .0f}\t\t\t{3: .0f}\t\t' \
              .format(week, xsol[k][0], xsol[k][1], xsol[k][2] ))
        
        week += abs((t0 - t[-1])/len(t))
    print('{0: .1f}\t\t{1: .0f}\t\t\t{2: .0f}\t\t\t{3: .0f}\t\t' \
              .format(week + abs((t0 - t[-1])/len(t)), xsol[-1][0] + 1, \
                      xsol[-1][1] -1 , xsol[-1][2] ))
        
    #The single ode system
    cits = 3500 
    H0 = 3350 
    I0 = 150 
    y0 = 0 
    
    [trk, Y, H, I] = rk4(t0, tmax, y0, steps)
    
    plt.plot(trk,H,'g-',trk,I ,'y-',trk,Y, 'r-')
    plt.title("Disease's Effects on Citizens\n c = {0:.4f} & m = {1:.1f}\n \
                  Single ODE System".format(c,m), loc = 'center')
              
    plt.legend(['Healthy','Infected', 'Deceased'], loc = 'best')
    plt.xlabel('# of weeks')
    plt.ylabel('Affected Villagers')
    plt.grid() 
    plt.show()


    
  
