import numpy as np
from pyBL import ThwaitesSimData, ThwaitesSim
from falkner_skan import falkner_skan
import sympy as sp
# import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline #for smoothed derivative experiment

#flags for what to run/plot
normal_sim = True
derivative_sim = False
#Inviscid Wedge Flow - LaPlace's Equation - inviscid solution to incompressible flow
alpha = .05 #beta = alpha*pi - 0 for flat plate
m = alpha/(2-alpha)
#m=.1
c = 1  # chord, in meters
r0 =0.00 #where the simulation will start
r = np.linspace(r0,c,100) #used as x - surface distance from origin


#rc = .2 #where clustering ends
#r = np.append(np.linspace(0,rc,55),np.linspace(rc+.001,c,50))
Vinf = 1 #U_0 
u_e = Vinf*pow(r,m)
nu = 1.45e-5  # kinematic viscosity

re = Vinf * r[-1] / nu

#y0 for sim (= theta0^2)
#y0 = .075*nu/tsd.du_edx(r0)  #from Moran
y0 = .45*pow(c,m)*nu*pow(r0,-1*m+1)/(Vinf*(5*m+1)) #theta^2 using analytical thwaites at r0

#y0 for theta^2 for flat plate
if m==0:
    y0 = 0

theta0 = np.sqrt(y0)
#set up pyBL data to use splining
tsd  =ThwaitesSimData(r*c,u_e,Vinf,nu,re,r0,theta0)

#Number of points on plot
n_plot = 200
plotr = np.linspace(r0,r[-1],n_plot)

#Analytical solution:
theta_analytical = np.sqrt(.45*pow(c,m)*nu*pow(r,-1*m+1)/(Vinf*(5*m+1)))    
du_edx_analytical = Vinf*m*pow(r/c,m-1)/c
#generate H values
h_analytical = []
c_f_analytical = []
for i in range(0,len(theta_analytical)):
    h_analytical+=[tsd.h(pow(theta_analytical[i],2)*du_edx_analytical[i]/nu)]
    c_f_analytical += [2 *nu*tsd.s(pow(theta_analytical[i],2)*du_edx_analytical[i]/nu) / (u_e[i]*theta_analytical[i])]
h_analytical = np.array(h_analytical)
c_f_analytical = np.array(c_f_analytical)

#Adjust y0 to attempt better results (current = 0)

#original y0 - only for use with r0=0 
#y0 = 0
#in general, would back out y0 from theta - this is based on the analytical solution
#y0 = pow(Vinf,5)*pow(r0,5*m+1)/((5*m+1)*pow(c,5*m))





lam0 = y0*tsd.du_edx(np.array([r0]))/nu

#Implementation with pyBL


ts = ThwaitesSim(tsd) #added adjustable y0
#initial_theta= 5E-4

if normal_sim==True:
    while ts.status=='running':
        ts.step()


#Running Thwaites again, but using the derivative of the velocity distribution instead
###Experiment with Smooth Derivative to establish spline
#r_der = r[np.isfinite(du_edx_analytical)] #removes x corresponding to infinite value
#du_edx = du_edx_analytical[np.isfinite(du_edx_analytical)] #removes infinite value


du_edx = np.copy(du_edx_analytical)
du_edx[np.isinf(du_edx_analytical)]=1E15 #replaces infinite value
du_edx[np.isnan(du_edx_analytical)]=0

due_dx_spline = CubicSpline(r,du_edx) #r_der
due_dx_spline_antiderivative = due_dx_spline.antiderivative()

#Create same simulation using u_e
tsd_der = tsd
# modify the u_e method to be the antiderivative of the duedx data + the first u_e
tsd_der.u_e = lambda x: due_dx_spline_antiderivative(x) +u_e[0]
#modify the du_edx method to be a curve fit to the smoothed derivative
tsd_der.du_edx = lambda x: due_dx_spline(x)
ts_der = ThwaitesSim(tsd_der)
if derivative_sim==True:
    while ts_der.status=='running': 
        ts_der.step()


    
#The following is stolen from the falkner-skan library's example
### Define constants
#m = 0.0733844181517584  # Exponent of the edge velocity ("a") #defined above
#Vinf = 7  # Velocity at the trailing edge
#nu = 1.45e-5  # kinematic viscosity
#c = 0.08  # chord, in meters
x = sp.symbols("x")  # x as a symbolic variable
ue = Vinf * (x / c) ** m

x_over_c = sp.symbols("x_over_c")

### Get the nondimensional solution
eta, f0, f1, f2 = falkner_skan(n_points=71,m=m)  # each returned value is a ndarray

### Get parameters of interest
Re_local = ue * x / nu
dFS = sp.sqrt(nu * x / ue)

theta_over_dFS = np.trapz(
    f1 * (1 - f1),
    dx=eta[1]
) * np.sqrt(2 / (m + 1))
dstar_over_dFS = np.trapz(
    (1 - f1),
    dx=eta[1]
) * np.sqrt(2 / (m + 1))
H = dstar_over_dFS / theta_over_dFS

Cf = 2 * np.sqrt((m+1)/2) * Re_local ** -0.5 * f2[0]

# Calculate the chord-normalized values of theta, H, and Cf
theta_x_over_c = (theta_over_dFS * dFS).subs(x, x_over_c * c).simplify()
H_x_over_c = H
Cf_x_over_c = (Cf).subs(x, x_over_c * c).simplify()

### Plot parameters of interest
plt.ion()

# Generate discrete values of parameters

#x_over_c_discrete = np.linspace(1 / 100, 1, 100)
#x_over_c_discrete = np.linspace(1 / n_plot, 1, n_plot)
x_over_c_discrete = np.linspace(0, 1, n_plot)
theta_x_over_c_discrete = sp.lambdify(x_over_c, theta_x_over_c, "numpy")(x_over_c_discrete)
H_x_over_c_discrete = sp.lambdify(x_over_c, H, "numpy")(x_over_c_discrete)
Cf_x_over_c_discrete = sp.lambdify(x_over_c, Cf_x_over_c, "numpy")(x_over_c_discrete)

# Plot it
fig, ax = plt.subplots()

#plt.subplot(311)
try:
    theta_x_over_c_discrete.shape
except AttributeError:
    theta_x_over_c_discrete = np.tile(theta_x_over_c_discrete, len(x_over_c_discrete))
    
plt.plot(x_over_c_discrete, theta_x_over_c_discrete,label=r'Falkner-Skan')
if normal_sim==True:
    plt.plot(plotr/c,ts.theta(plotr),label='Thwaites')
if derivative_sim==True:
    plt.plot(plotr/c,ts_der.theta(plotr),label='Thwaites (using derivative)')
plt.plot(r,theta_analytical,label='Thwaites (Analytical)')
plt.title(r"$\theta$ Distribution")
plt.xlabel(r"$x/c$")
plt.ylabel(r"$\theta$ (m)")
ax.legend(loc='center right', ncol=1)
plt.grid(True)

fig, ax = plt.subplots()
#plt.subplot(312)
plt.plot(x_over_c_discrete, np.tile(H_x_over_c_discrete, len(x_over_c_discrete)),label=r'Falkner-Skan')
if normal_sim==True:
    plt.plot(plotr/c,ts.h_x(plotr),label='Thwaites')
if derivative_sim==True:
    plt.plot(plotr/c,ts_der.h_x(plotr),label='Thwaites (using derivative)')
plt.plot(r/c,h_analytical,label='Thwaites (Analytical)')
plt.title(r"$H$ Distribution")
plt.xlabel(r"$x/c$")
plt.ylabel(r"$H$ (unitless)")
ax.legend(loc='upper right', ncol=1)
plt.grid(True)

fig, ax = plt.subplots()
#plt.subplot(313)
plt.plot(x_over_c_discrete, Cf_x_over_c_discrete,label=r'Falkner-Skan')
if normal_sim==True:
    plt.plot(plotr/c,ts.c_f(plotr),label='Thwaites')
if derivative_sim==True:
    plt.plot(plotr/c,ts_der.c_f(plotr),label='Thwaites (using derivative)')
plt.plot(r/c,c_f_analytical,label='Thwaites (Analytical)')
plt.title(r"$C_f$ Distribution")
ax.legend(loc='upper right', ncol=1)
#plt.xlim(0,.2)

plt.xlabel(r"$x/c$")
plt.ylabel(r"$C_f$ (unitless)")
plt.grid(True)


fig,ax = plt.subplots()
if normal_sim==True:
    plt.plot(plotr,ts.du_edx(plotr),label='Thwaites')
if derivative_sim==True:
    plt.plot(plotr,ts_der.du_edx(plotr),label='Thwaites (using derivative)')
plt.plot(r,du_edx_analytical,label='Thwaites (Analytical)')
plt.title(r"$\frac{du_e}{dx}$")
plt.xlabel('x(m)')
plt.ylabel(r"$\frac{du_e}{dx}$")

ax.legend(loc='upper right')

fig,ax = plt.subplots()
if normal_sim==True:
    plt.plot(r,ts.u_e(r),label='Thwaites')
if derivative_sim==True:
    plt.plot(r,ts_der.u_e(r),label='Thwaites (using derivative)')
#plt.plot(r,u_e,label='Actual Velocities')
plt.plot(r,u_e,label='Actual Velocity')
plt.title(r"$u_e$ spline ")
plt.xlabel('x(m)')
plt.ylabel(r"$u_e$")
ax.legend(loc='upper right')

fig,ax = plt.subplots()
if normal_sim==True:
    plt.plot(r,ts.u_e(r)-u_e,label='Thwaites')
if derivative_sim==True:
    plt.plot(r,ts_der.u_e(r)-u_e,label='Thwaites (using derivative)')
#plt.plot(r,u_e,label='Actual Velocities')
plt.title(r"$u_e$ spline error")
plt.xlabel('x(m)')
plt.ylabel(r"$u_e$")
ax.legend(loc='upper right')

fig,ax = plt.subplots()
if normal_sim==True:
    plt.plot(r,ts.du_edx(r)-du_edx_analytical,label='Thwaites')
if derivative_sim==True:
    plt.plot(r,ts_der.du_edx(r)-du_edx_analytical,label='Thwaites (using derivative)')
#plt.plot(r,u_e,label='Actual Velocities')
plt.title(r"$\frac{du_e}{dx}$ spline error")
plt.xlabel('x(m)')
plt.ylabel(r"$u_e$")
ax.legend(loc='upper right')

# -*- coding: utf-8 -*-

fig,ax = plt.subplots()

if normal_sim==True:
    plt.plot(plotr/c,ts.del_star(plotr),label='Thwaites')
if derivative_sim==True:
    plt.plot(plotr/c,ts_der.del_star(plotr),label='Thwaites (using derivative)')
plt.plot(r,theta_analytical*h_analytical,label='Thwaites (Analytical)')
plt.plot(x_over_c_discrete, theta_x_over_c_discrete*H_x_over_c_discrete,label=r'Falkner-Skan')
plt.title(r"$\delta*$ Distribution")
plt.xlabel(r"$x/c$")
plt.ylabel(r"$\delta*$")
ax.legend(loc='center right', ncol=1)
plt.grid(True)

#Compare Normal Thwaites approximation for 2()
if normal_sim==True:
    sim = ts
elif derivative_sim==True:
    sim = ts_der
    
plotlam = np.linspace(-.1,.1)
simple_thwaites = .45-6*plotlam
new_thwaites = np.zeros(plotlam.shape) #preallocate
for i in range(len(plotlam)):
    new_thwaites[i] = 2*(sim.s_lam(plotlam[i])-(2+sim.h_lam(plotlam[i]))*plotlam[i])
fig, ax = plt.subplots()
plt.plot(plotlam,simple_thwaites,label='Original')
plt.plot(plotlam,new_thwaites,label='New Method')
ax.legend(loc='upper right')
