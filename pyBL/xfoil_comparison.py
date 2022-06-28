# -*- coding: utf-8 -*-
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from pyBL import ThwaitesSim,ThwaitesSimData,ThwaitesSeparation, Michel,HeadSim,HeadSimData, HeadSeparation
import time
import tikzplotlib

from plot_BL_params import theta_linestyle,theta_label,del_label,del_linestyle,c_f_label,c_f_linestyle,h_label,h_linestyle,error_label,x_label
from plot_BL_params import plot_BL_params,pybl_label,pybl_linestyle,xfoil_label,xfoil_linestyle,michel_label,michel_linestyle,retheta_label,retheta_linestyle

#For consistent plotting
thetacolor = 'tab:blue'
hcolor = 'tab:orange'
delcolor = 'tab:green'
cfcolor = 'tab:red'

pyblcolor = 'tab:blue'
xfoilcolor = 'tab:red'
michelcolor = 'tab:purple'

#Place a copy of xfoil in the same folder as this script
# airfoil = '4412'
airfoil='0009'
aoa = 0
n_iter = 1
n_transition = 9 #for xfoil e^N, 9 is default
v_inf = 20 #m/s 
re = 2E6
force_trans = 1

le_sep_buffer = 0 #buffer to avoid separation at nonphysical leading edge
le_trans_buffer = 0 #buffer to avoid transition at nonphysical leading edge 

# batchfilename = 'runairfoil.txt'
# batchfile = open(batchfilename,'w')
invfile = 'inv.txt'
invcpfile = 'invcp.txt'
viscfile =  'visc.txt'

invisc_command_list = ("""NACA """+str(airfoil)+"""
OPER
a """+str(aoa)+"""
dump """+invfile+"""
cpwr """+invcpfile+"""

quit
""").encode('utf-8')

process = subprocess.Popen(['xfoil.exe'],
              stdin=subprocess.PIPE,
              stdout=None,
              stderr=None)
process.communicate(invisc_command_list)
process.wait()

#Get the inviscid sim u_e and coordinates
invdata = np.loadtxt(invfile)
s_data = invdata[:,0]
x_data = invdata[:,1]
u_e_over_v_inf_data = invdata[:,3]

#Get the inviscid cp
invcpdata = np.loadtxt(invcpfile,skiprows=2)
cp = invcpdata[:,2]
#Extract estimation of stagnation point
stagnation_ind = np.where(abs(cp-1) == min(abs(cp-1)))[0][0] -1 #stagnation based on cp being close to 1
# stagnation_ind = int(np.where(x_data==min(x_data))[0][0]) #location of minimum x coordinate
# stagnation_ind = np.where(u_e_over_v_inf_data==abs(u_e_over_v_inf_data))[-1][-1] #last place velocity is positive (first place on chord)


#flip the inviscid results
# s = np.flip(s_data[leading_edge_ind]-s_data[0:leading_edge_ind+1]) #
# u_e = np.flip(u_e_over_v_inf_data[0:leading_edge_ind+1])*v_inf
s = np.flip(s_data[stagnation_ind]-s_data) #s=0 is stagnation point
u_e = np.flip(u_e_over_v_inf_data)*v_inf
s0 = 0

#Perform laminar analysis
nu = v_inf*(s_data[stagnation_ind]-s_data[0])/re #use maybe different idea for chord length

def white_s(lam):
    return pow(lam+.09,.62)
    
def white_h(lam):
    z = .25-lam
    return 2+4.14*z-83.5*pow(z,2) +854*pow(z,3) -3337*pow(z,4) +4576*pow(z,5)

#Run viscous xfoil
visc_command_list = ("""NACA """+str(airfoil)+"""
OPER
VISC
"""+str(re)+"""
ITER 
"""+str(n_iter)+"""
vpar
n
"""+str(n_transition)+"""
xtr
"""+str(force_trans)+"""
1

a """+str(aoa)+"""
dump """+viscfile+"""

quit
""").encode('utf-8')

# visc_command_list = ("""NACA """+str(airfoil)+"""
# OPER
# VISC
# """+str(re)+"""
# ITER 
# """+str(n_iter)+"""
# vpar
# n
# """+str(n_transition)+"""
# xtr
# """+str(float(michel.x_tr))+"""
# 1

# a """+str(aoa)+"""
# dump """+viscfile+"""

# quit
# """).encode('utf-8')

process = subprocess.Popen(['xfoil.exe'],
              stdin=subprocess.PIPE,
              stdout=None,
              stderr=None)
process.communicate(visc_command_list)
process.wait()




#truncate viscous data (avoid added points)
viscdata = np.loadtxt(viscfile)
invlength = invdata.shape[0]
del_star_data = viscdata[0:invlength,4]
theta_data = viscdata[0:invlength,5]
c_f_data = viscdata[0:invlength,6]
h_data = viscdata[0:invlength,7]




# del_star = np.flip(del_star_data[0:leading_edge_ind+1])
# theta = np.flip(theta_data[0:leading_edge_ind+1])
# c_f = np.flip(c_f_data[0:leading_edge_ind+1])
# h = np.flip(h_data[0:leading_edge_ind+1])

#flip the viscous results
del_star = np.flip(del_star_data)
theta = np.flip(theta_data)
c_f = np.flip(c_f_data)*(v_inf**2)/(u_e**2)
h = np.flip(h_data)


#tsd = ThwaitesSimData(s,u_e,v_inf,nu,re,s0,0,white_s,white_h)
# tsd = ThwaitesSimData(s,u_e,v_inf,nu,re,s0,theta0=None) #entering theta0 as none uses moran for y0
# tsd = ThwaitesSimData(s,u_e,v_inf,nu,re,s0,theta0=theta_data[stagnation_ind],s=white_s,h=white_h) #entering theta0 as none uses moran for y0
tsd = ThwaitesSimData(s,u_e,v_inf,nu,re,s0,theta0=theta_data[stagnation_ind]) #entering theta0 as none uses moran for y0

# tsd.theta0 = np.sqrt(.075*nu/tsd.du_edx(s0)) 
ts = ThwaitesSim(tsd) 
while ts.status=='running':
    ts.step()
michel = Michel(ts,buffer = le_trans_buffer)
thwaites_sep = ThwaitesSeparation(ts,buffer=le_sep_buffer)
t_x_sep = thwaites_sep.x_sep
x_tr = michel.x_tr


try:
    hsd = HeadSimData(s,
                      u_e,
                      v_inf,
                      nu,
                      float(michel.x_tr), #x0
                      float(ts.theta(michel.x_tr)),
                      michel.h0)
# h0 = 1.4754/np.log(ts.rtheta(force_trans)) +.9698
# force_trans =np.array([force_trans])

# try:
#     hsd = HeadSimData(s,
#                       u_e,
#                       v_inf,
#                       nu,
#                       float(force_trans), #x0
#                       float(ts.theta(force_trans)),
#                       h0)
    hs = HeadSim(hsd)
    while hs.status=='running':
        hs.step()
    theta_rel_err = abs((np.append(ts.theta(s[s<=michel.x_tr]),hs.theta(s[s>michel.x_tr]))-theta)/theta )
    h_rel_err = abs((np.append(ts.h(s[s<=michel.x_tr]),hs.h(s[s>michel.x_tr]))-h)/h )
    del_star_rel_err = abs((np.append(ts.del_star(s[s<=michel.x_tr]),hs.del_star(s[s>michel.x_tr]))-del_star)/del_star )
    c_f_rel_err = abs((np.append(ts.c_f(s[s<=michel.x_tr]),hs.c_f(s[s>michel.x_tr]))-c_f)/c_f )
    
    
    # hs.dense_output_vec[-1](1.03)
    head_sep = HeadSeparation(hs)
    if head_sep.separated==True:
        print('Turbulent boundary layer has separated at x={}'.format(head_sep.x_sep))
    h_x_sep =  head_sep.x_sep   
    s_lam = np.linspace(s[0],float(michel.x_tr))
    s_turb = np.linspace(float(michel.x_tr),s[-1])
    s_tot = np.append(s_lam,s_turb)
except TypeError: #happens when flow has not transitioned
    s_lam = s
    s_turb = np.empty(0)
    s_tot =s
    hs = ts
    theta_rel_err = abs((ts.theta(s)-theta)/theta)
    h_rel_err = abs((ts.h(s)-h)/h)
    del_star_rel_err = abs((ts.del_star(s)-del_star)/del_star)
    c_f_rel_err = abs((ts.c_f(s)-c_f)/c_f)
    h_x_sep = None
    
# fig,ax = plt.subplots()
# plt.plot(s,del_star,label='XFOIL',color=xfoilcolor)
# plt.plot(s_tot,np.append(ts.del_star(s_lam),hs.del_star(s_turb)),label='pyBL',color=pyblcolor)
# plt.xlabel('x(m)')
# plt.ylabel(r'$\delta$* (m)')
# plt.xlim([0,max(s)])
# ax.legend(loc='upper left')
# plt.grid(True)
# tikzplotlib.save(
#     'figures/full_xfoil_del.tex',
#     axis_height = '\\figH',
#     axis_width = '\\figW'
    # )

# fig,ax = plt.subplots()
# plt.plot(s,h,label='XFOIL',color=xfoilcolor)
# plt.plot(s_tot,np.append(ts.h(s_lam),hs.h(s_turb)),label='pyBL',color=pyblcolor)
# plt.xlabel('x(m)')
# plt.ylabel(r'$H$ (m)')
# plt.xlim([0,max(s)])
# ax.legend(loc='upper right')
# plt.grid(True)
# tikzplotlib.save(
#     'figures/full_xfoil_h.tex',
#     axis_height = '\\figH',
#     axis_width = '\\figW'
#     )

# fig,ax = plt.subplots()
# plt.plot(s,c_f,label='XFOIL',color=xfoilcolor)
# # plt.plot(s,c_f*(v_inf**2)/(u_e**2),label='XFOIL (new)')
# plt.plot(s_tot,np.append(ts.c_f(s_lam),hs.c_f(s_turb)),label='pyBL',color=pyblcolor)
# plt.xlabel('x(m)')
# plt.ylabel(r'$c_f$* (m)')
# plt.xlim([0,max(s)])
# plt.ylim([-.05,.05])
# ax.legend(loc='upper right')
# plt.grid(True)
# tikzplotlib.save(
#     'figures/full_xfoil_cf.tex',
#     axis_height = '\\figH',
#     axis_width = '\\figW'
#     )


fig,ax = plt.subplots()
ax.plot(s,ts.rtheta(s),label=retheta_label,linestyle=retheta_linestyle,color='k')
plt.plot(s,2.9*pow(ts.u_e(s)*s/nu,.4),label=michel_label,linestyle=michel_linestyle,color='k')
ax.legend(loc='lower right')
plt.xlim([0,.7])
plt.ylim([0,1500])
plt.grid(True)
tikzplotlib.save(
    'figures/full_xfoil_michel.tex',
    axis_height = '\\figH',
    axis_width = '\\figW'
    )

# fig,ax = plt.subplots()
# ax.plot(s,u_e,label='XFOIL')
# ax.plot(s,ts.u_e(s),label='Spline')
# ax.legend(loc='upper right')

# fig,ax = plt.subplots()
# plt.plot(s,theta,label='XFOIL',color=xfoilcolor)
# plt.plot(s_tot,np.append(ts.theta(s_lam),hs.theta(s_turb)),label='pyBL',color=pyblcolor)
# plt.xlabel(x_label)
# plt.ylabel(r'$\Theta$ (m)')
# ax.legend(loc='upper left')
# plt.xlim([0,max(s)])
# plt.grid(True)
# tikzplotlib.save(
#     'figures/full_xfoil_theta.tex',
#     axis_height = '\\figH',
#     axis_width = '\\figW'
#     )

fig,ax = plt.subplots()
plt.plot(s_lam,ts.lam(s_lam),label='$\lambda$',color=pyblcolor)
plt.xlabel('x(m)')
plt.ylabel('$\lambda$')
plt.xlim([0,max(s_lam)])
plt.grid(True)


#relative errors
fig,ax = plt.subplots()
# plt.plot(s,theta_rel_err,label = '$\Theta$',color=thetacolor)
# plt.plot(s,h_rel_err,label='$H$',color=hcolor)
# plt.plot(s,del_star_rel_err,label='$\delta*$',color=delcolor)
# plt.plot(s,c_f_rel_err,label='$c_f$',color=cfcolor)
plt.plot(s,theta_rel_err,label = theta_label,color='k',linestyle=theta_linestyle)
plt.plot(s,del_star_rel_err,label=del_label,color='k',linestyle=del_linestyle)
plt.plot(s,c_f_rel_err,label=c_f_label,color='k',linestyle=c_f_linestyle)
plt.plot(s,h_rel_err,label=h_label,color='k',linestyle=h_linestyle)

ax.legend()
plt.xlim(0,max(s))
plt.yscale('log')
plt.grid(True)
ax.set(xlabel=x_label,ylabel=error_label)
tikzplotlib.save(
    'figures/full_xfoil_error.tex',
    axis_height = '\\figH',
    axis_width = '\\figW'
    )

fig,axs = plot_BL_params(x=s[s>=0],
                         theta=theta[s>=0],
                         h=h[s>=0],
                         delta=del_star[s>=0],
                         c_f=c_f[s>=0],
                         label=xfoil_label,
                         linestyle=xfoil_linestyle,
                         )
plt.xlim(0,max(s))
fig,axs = plot_BL_params(x=s_tot[s_tot>=0],
                         delta = np.append(ts.del_star(s_lam),hs.del_star(s_turb))[s_tot>=0],
                         h=np.append(ts.h(s_lam),hs.h(s_turb))[s_tot>=0],
                         theta=np.append(ts.theta(s_lam),hs.theta(s_turb))[s_tot>=0],
                         c_f =np.append(ts.c_f(s_lam),hs.c_f(s_turb))[s_tot>=0],
                          label=pybl_label,
                          linestyle=pybl_linestyle,
                          fig=fig,
                          axs=axs,
                          last=True,
                          file='full_xfoil',
                          )  
                 

time.sleep(2)
print('\n\n\nLaminar Separation Trigger @ x = {}'.format(t_x_sep))
print('Transition criteria @ x = {}'.format(x_tr))
print('Turbulent Separation Trigger @x = {}\n\n\n'.format(h_x_sep))

if thwaites_sep.separated==True and (x_tr is None or t_x_sep<x_tr):
    print('Laminar boundary layer has separated at x={}'.format(t_x_sep))
elif michel.transitioned==True:
    print('Boundary layer transition has occured at x={}'.format(x_tr))
    try:
        if head_sep.separated==True:
            print('Turbulent boundary layer has separated at x={}'.format(h_x_sep))
    except NameError:
        pass
    

