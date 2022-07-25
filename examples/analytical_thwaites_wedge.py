import os
import numpy as np

from pyBL.thwaites_method import ThwaitesSimData, ThwaitesSim, ThwaitesSeparation

from falkner_skan import falkner_skan
import sympy as sp
import matplotlib.pyplot as plt

import tikzplotlib
from plot_BL_params import plot_BL_params, falkner_skan_linestyle,falkner_skan_label, thwaites_label, thwaites_linestyle, thwaites_lin_label,thwaites_lin_linestyle,thwaites_analytical_label,thwaites_analytical_linestyle
from plot_BL_params import theta_linestyle,theta_label,del_label,del_linestyle,c_f_label,c_f_linestyle,h_label,h_linestyle,error_label,x_label

def wedgeflow(alpha,r0,npts,Vinf,nu,n_plot,name,y0=None,theta0=None,use_analytical=False):
    #Inviscid Wedge Flow - LaPlace's Equation - inviscid solution to incompressible flow
    #alpha = angle*2/pi
    # r0 = distance from stagnation where the simulation will start
    # npts = number of points in velocity distribution
    # Vinf = freestream velocity
    # nu =  kinematic viscosity 
    #n_plot = number of points on plot
    
    m = alpha/(2-alpha)
    c = 1  # chord, in meters
    r1 = np.linspace(0.001,r0,15)
    r2 = np.linspace(r0,c,npts) #1d array of distances from origin
    if r0!=0:
        r = np.concatenate((r1[:-1],r2),axis=None)
    else:
        r = r2

    u_e = Vinf*pow(r,m) #edge velocity at each point in r
    
    

    re = Vinf * r[-1] / nu #freestream Reynolds Number
    
    if y0==None and theta0==None:
        # theta0 = None
        # if r0!=0:
            #uses analytic if not at x=0
            theta0 = np.sqrt(.45*pow(c,m)*nu*pow(r0,-1*m+1)/(Vinf*(5*m+1)))
    elif theta0!=None and y0!=None:
        print('theta0 and y0 overdefined, using y0')
        theta0 = np.sqrt(y0)
    elif theta0!=None:
        pass
    elif y0!=None:
        theta0 = np.sqrt(y0)
        
    tsd  =ThwaitesSimData(r*c,u_e,Vinf,nu,re,r0,theta0,linearize=False)


    plotr = np.linspace(r0,r[-1],n_plot)

    #Analytical solution:
    theta_analytical = np.sqrt(.45*pow(c,m)*nu*pow(plotr,-1*m+1)/(Vinf*(5*m+1))) #analytic momentum thickness solution (assumes linear F(lambda)) 
    du_edx_analytical = Vinf*m*pow(plotr/c,m-1)/c #analytic velocity derivative
    plot_u_e = Vinf*pow(plotr,m) 
    #generate H values
    h_analytical = []
    c_f_analytical = []
    for i in range(0,len(theta_analytical)):
        h_analytical+=[tsd.h_lam(pow(theta_analytical[i],2)*du_edx_analytical[i]/nu)]
        c_f_analytical += [2 *nu*tsd.s_lam(pow(theta_analytical[i],2)*du_edx_analytical[i]/nu) / (plot_u_e[i]*theta_analytical[i])]
    h_analytical = np.array(h_analytical)
    c_f_analytical = np.array(c_f_analytical)
    del_star_analytical = theta_analytical*h_analytical
        
        
    
    # lam0 = y0*tsd.du_edx(np.array([r0]))/nu
    
    #Implementation with pyBL
    
    
    ts = ThwaitesSim(tsd) 
    thwaites_sep = ThwaitesSeparation(ts)
    while ts.status=='running':
        ts.step()
    
    #Compare against linearized
    tsd_linear = tsd
    tsd_linear._linearize = True
    ts_linear = ThwaitesSim(tsd_linear) 
    while ts_linear.status=='running':
        ts_linear.step()
    
    
    

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
    # fig, ax = plt.subplots()
    
    #plt.subplot(311)
    try:
        theta_x_over_c_discrete.shape
    except AttributeError:
        theta_x_over_c_discrete = np.tile(theta_x_over_c_discrete, len(x_over_c_discrete))
        
    
    
    
    #New Thwaites Error w/ FS
    
    fig,ax = plt.subplots()
    plt.plot(x_over_c_discrete[x_over_c_discrete>r0],abs((ts.theta(x_over_c_discrete)-theta_x_over_c_discrete)/theta_x_over_c_discrete)[x_over_c_discrete>r0],linestyle=theta_linestyle,color='k',label=theta_label)
    plt.plot(x_over_c_discrete[x_over_c_discrete>r0],abs((ts.del_star(x_over_c_discrete)-theta_x_over_c_discrete*H_x_over_c_discrete)/(theta_x_over_c_discrete*H_x_over_c_discrete))[x_over_c_discrete>r0],color='k',linestyle=del_linestyle,label=del_label)
    plt.plot(x_over_c_discrete[x_over_c_discrete>r0],abs((ts.c_f(x_over_c_discrete)-Cf_x_over_c_discrete)/Cf_x_over_c_discrete)[x_over_c_discrete>r0],linestyle=c_f_linestyle,color='k',label=c_f_label)
    plt.plot(x_over_c_discrete[x_over_c_discrete>r0],abs((ts.h(x_over_c_discrete)-H_x_over_c_discrete)/H_x_over_c_discrete)[x_over_c_discrete>r0],linestyle=h_linestyle,color='k',label=h_label)
    plt.yscale('log')
    plt.ylim([.00001,5])
    plt.grid(True)
    plt.xlabel(x_label)
    plt.ylabel(error_label)
    ax.legend()
    
    if (not os.path.isdir('figures')):
        os.mkdir('figures')

    tikzplotlib.save(
        'figures/'+name+'_error.tex',
        axis_height = '\\figH',
        axis_width = '\\figW'
        )
    
    #Linear Thwaites Error w/ FS
    fig,ax = plt.subplots()
    plt.plot(x_over_c_discrete[x_over_c_discrete>r0],abs((ts_linear.theta(x_over_c_discrete)-theta_x_over_c_discrete)/theta_x_over_c_discrete)[x_over_c_discrete>r0],linestyle=theta_linestyle,color='k',label=theta_label)
    plt.plot(x_over_c_discrete[x_over_c_discrete>r0],abs((ts_linear.del_star(x_over_c_discrete)-theta_x_over_c_discrete*H_x_over_c_discrete)/(theta_x_over_c_discrete*H_x_over_c_discrete))[x_over_c_discrete>r0],color='k',linestyle=del_linestyle,label=del_label)
    plt.plot(x_over_c_discrete[x_over_c_discrete>r0],abs((ts_linear.c_f(x_over_c_discrete)-Cf_x_over_c_discrete)/Cf_x_over_c_discrete)[x_over_c_discrete>r0],linestyle=c_f_linestyle,color='k',label=c_f_label)
    
    plt.plot(x_over_c_discrete[x_over_c_discrete>r0],abs((ts_linear.h(x_over_c_discrete)-H_x_over_c_discrete)/H_x_over_c_discrete)[x_over_c_discrete>r0],color='k',label=h_label,linestyle=h_linestyle)
    plt.yscale('log')
    plt.ylim([.00001,5])
    plt.grid(True)
    plt.xlabel(x_label)
    plt.ylabel(error_label)
    ax.legend()
    
    tikzplotlib.save(
        'figures/'+name+'_linear_error.tex',
        axis_height = '\\figH',
        axis_width = '\\figW',
        
        )
    
    
    
    # fig,ax = plt.subplots()
    # plt.plot(plotr,ts.lam(plotr))
    
    fig,axs = plot_BL_params(theta=theta_x_over_c_discrete,
                             c_f=Cf_x_over_c_discrete,
                             h=np.tile(H_x_over_c_discrete, len(x_over_c_discrete)),
                             delta=theta_x_over_c_discrete*H_x_over_c_discrete,
                             x=x_over_c_discrete,
                             label=falkner_skan_label, 
                             linestyle=falkner_skan_linestyle,
                             ) 
    fig,axs = plot_BL_params(theta=theta_analytical,
                             c_f=c_f_analytical,
                             h=h_analytical,
                             delta=del_star_analytical,
                             x=plotr/c,
                             label=thwaites_analytical_label, 
                             linestyle=thwaites_analytical_linestyle,
                             fig=fig,
                             axs=axs
                             ) 
    fig,axs = plot_BL_params(x =plotr/c,
                             label=thwaites_label,
                             linestyle=thwaites_linestyle,
                             fig=fig,
                             axs=axs,
                             sim=ts
                             )
    fig,axs = plot_BL_params(x=plotr/c,
                              label=thwaites_lin_label,
                              linestyle=thwaites_lin_linestyle,
                              fig=fig,
                              axs=axs,
                              last=True,
                              sim=ts_linear,
                              file=name,
                              )  
    tsd2 = tsd
    # tsd2.theta0 = theta_x_over_c_discrete[0]
    # tsd2.theta0 = theta_x_over_c_discrete[0]*1.1
    # tsd2.theta0 = theta_x_over_c_discrete[0]*0.9

    # ts = ThwaitesSim(tsd2) 
    # thwaites_sep = ThwaitesSeparation(ts)
    # while ts.status=='running':
    #     ts.step()
    
    # # #Compare against linearized
    # # tsd_linear = tsd
    # # tsd_linear._linearize = True
    # # ts_linear = ThwaitesSim(tsd_linear) 
    # # while ts_linear.status=='running':
    # #     ts_linear.step()
    
    # fig,ax = plt.subplots()
    # plt.plot(x_over_c_discrete[x_over_c_discrete>r0],abs((ts.theta(x_over_c_discrete)-theta_x_over_c_discrete)/theta_x_over_c_discrete)[x_over_c_discrete>r0],linestyle=theta_linestyle,color='k',label=theta_label)
    # plt.plot(x_over_c_discrete[x_over_c_discrete>r0],abs((ts.del_star(x_over_c_discrete)-theta_x_over_c_discrete*H_x_over_c_discrete)/(theta_x_over_c_discrete*H_x_over_c_discrete))[x_over_c_discrete>r0],color='k',linestyle=del_linestyle,label=del_label)
    # plt.plot(x_over_c_discrete[x_over_c_discrete>r0],abs((ts.c_f(x_over_c_discrete)-Cf_x_over_c_discrete)/Cf_x_over_c_discrete)[x_over_c_discrete>r0],linestyle=c_f_linestyle,color='k',label=c_f_label)
    # plt.plot(x_over_c_discrete[x_over_c_discrete>r0],abs((ts.h(x_over_c_discrete)-H_x_over_c_discrete)/H_x_over_c_discrete)[x_over_c_discrete>r0],linestyle=h_linestyle,color='k',label=h_label)
    # plt.yscale('log')
    # plt.ylim([.00001,5])
    # plt.grid(True)
    # plt.xlabel(x_label)
    # plt.ylabel(error_label)
    # ax.legend()
    