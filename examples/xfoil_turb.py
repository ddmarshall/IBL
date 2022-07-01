# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

from pyBL.heads_method import HeadSim, HeadSimData, HeadSeparation
import time
import tikzplotlib

from xfoil_interface import get_xfoil_data
from plot_BL_params import theta_linestyle,theta_label,del_label,del_linestyle,c_f_label,c_f_linestyle,h_label,h_linestyle,error_label,x_label
from plot_BL_params import plot_BL_params,pybl_label,pybl_linestyle,xfoil_label,xfoil_linestyle

#For consistent plotting
thetacolor = 'tab:blue'
hcolor = 'tab:orange'
delcolor = 'tab:green'
cfcolor = 'tab:red'

pyblcolor = 'tab:blue'
xfoilcolor = 'tab:red'

# Get xfoil info
v_inf = 20 #m/s 
re = 2E6
s_trans = 0.01 # transition near start of airfoil

c, theta0, h0, s, u_e, del_star, theta, c_f, h = get_xfoil_data('4412', -1, v_inf, re, s_trans)
s0 = 0
nu = v_inf*c/re

hsd = HeadSimData(s, u_e, v_inf, nu, 0.0, theta0, h0)
hs = HeadSim(hsd)
while hs.status=='running':
    hs.step()

theta_rel_err = abs(((hs.theta(s))-theta)/theta )
h_rel_err = abs(((hs.h(s))-h)/h )
del_star_rel_err = abs(((hs.del_star(s))-del_star)/del_star )
c_f_rel_err = abs(((hs.c_f(s))-c_f)/c_f )


# hs.dense_output_vec[-1](1.03)
head_sep = HeadSeparation(hs)
if head_sep.separated==True:
    print('Turbulent boundary layer has separated at x={}'.format(head_sep.x_sep))
h_x_sep =  head_sep.x_sep   

s_tot = s

    
# fig,ax = plt.subplots()
# plt.plot(s,del_star,label='XFOIL',color=xfoilcolor)
# plt.plot(s,hs.del_star(s),label='pyBL',color=pyblcolor)
# plt.xlabel('x(m)')
# plt.ylabel(r'$\delta$* (m)')
# plt.xlim([0,max(s)])
# ax.legend(loc='upper left')
# plt.grid(True)
# tikzplotlib.save(
#     'figures/xfoil_turb_del.tex',
#     axis_height = '\\figH',
#     axis_width = '\\figW'
#     )

# fig,ax = plt.subplots()
# plt.plot(s,h,label='XFOIL',color=xfoilcolor)
# plt.plot(s,hs.h(s),label='pyBL',color=pyblcolor)
# plt.xlabel('x(m)')
# plt.ylabel(r'$H$ (m)')
# plt.xlim([0,max(s)])
# ax.legend(loc='upper right')
# plt.grid(True)
# tikzplotlib.save(
#     'figures/xfoil_turb_h.tex',
#     axis_height = '\\figH',
#     axis_width = '\\figW'
#     )


# fig,ax = plt.subplots()
# plt.plot(s,c_f,label='XFOIL',color=xfoilcolor)
# # plt.plot(s,c_f*(v_inf**2)/(u_e**2),label='XFOIL (new)')
# plt.plot(s,hs.c_f(s),label='pyBL',color=pyblcolor)
# plt.xlabel('x(m)')
# plt.ylabel(r'$c_f$* (m)')
# plt.xlim([0,max(s)])
# plt.ylim([-.05,.05])
# ax.legend(loc='upper right')
# plt.grid(True)
# tikzplotlib.save(
#     'figures/xfoil_turb_cf.tex',
#     axis_height = '\\figH',
#     axis_width = '\\figW'
#     )


# # fig,ax = plt.subplots()
# # ax.plot(s,u_e,label='XFOIL')
# # ax.plot(s,ts.u_e(s),label='Spline')
# # ax.legend(loc='upper right')

# fig,ax = plt.subplots()
# plt.plot(s,theta,label='XFOIL',color=xfoilcolor)
# plt.plot(s,hs.theta(s),label='pyBL',color=pyblcolor)
# plt.xlabel('x(m)')
# plt.ylabel(r'$\Theta$ (m)')
# ax.legend(loc='upper left')
# plt.xlim([0,max(s)])
# plt.grid(True)
# tikzplotlib.save(
#     'figures/xfoil_turb_theta.tex',
#     axis_height = '\\figH',
#     axis_width = '\\figW'
#     )

#relative errors
# fig,ax = plt.subplots()
# plt.plot(s,theta_rel_err,label = '$\Theta$',color=thetacolor)
# plt.plot(s,h_rel_err,label='$H$',color=hcolor)
# plt.plot(s,del_star_rel_err,label='$\delta*$',color=delcolor)
# plt.plot(s,c_f_rel_err,label='$c_f$',color=cfcolor)
# ax.legend()
# plt.xlim(0,max(s))
# plt.yscale('log')
# plt.grid(True)
# ax.set(xlabel='$x$(m)',ylabel='Relative Error')
# tikzplotlib.save(
#     'figures/xfoil_turb_error.tex',
#     axis_height = '\\figH',
#     axis_width = '\\figW'
#     )

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
    'figures/xfoil_turb_error.tex',
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
fig,axs = plot_BL_params(x=s[s>=0.02],
                         sim=hs,
                          label=pybl_label,
                          linestyle=pybl_linestyle,
                          fig=fig,
                          axs=axs,
                          last=True,
                          file='xfoil_turb',
                          )  
                 
#explain spline at TE
fig,ax = plt.subplots()
plt.plot(s[s>=0],u_e[s>=0],color='k',linestyle=':')
# plt.plot(s[s>=0],hs.u_e(s[s>=0]))
ax.set(xlabel=x_label,ylabel='$u_e$')
tikzplotlib.save(
    'figures/xfoil_u_e_dist.tex',
    axis_height = '\\figH',
    axis_width = '\\figW'
    )
time.sleep(2)

print('Transition criteria @ x = {}'.format(0))
print('Turbulent Separation Trigger @x = {}\n\n\n'.format(h_x_sep))
if head_sep.separated==True:
            print('Turbulent boundary layer has separated at x={}'.format(h_x_sep))
 
turb_sep_vars = open('figures/turb_sep_vars.tex','w')
turb_sep_vars.write('\\newcommand\\turbsepx{'+'{0:.4g}'.format(float(head_sep.x_sep))+'}\n')
turb_sep_vars.write('\\newcommand\\h0{'+'{0:.4g}'.format(float(h[s==0]))+'}\n')
turb_sep_vars.write('\\newcommand\\headh0{'+'{0:.4g}'.format(float(hs.h([0])))+'}\n')

turb_sep_vars.close()
