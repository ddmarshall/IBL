import matplotlib.pyplot as plt
import tikzplotlib

thetacolor = 'tab:blue'
hcolor = 'tab:orange'
delcolor = 'tab:green'
cfcolor = 'tab:red'
color_list = [thetacolor,cfcolor,hcolor,delcolor]

# thetapat = ':'
# hpat = '-.'
# delpat = '--'
# cfpat = '-'
# pat_list = [thetapat,cfpat,hpat,delpat]
# xfoil_color = 
# pybl_color = 
#pybl_label
#thwaites_linear_style = 
#white_color = 
#white_label= 

x_label = '$x$(m)'
#For plotting thwaites
falkner_skan_linestyle = ':'
falkner_skan_label = 'Falkner-Skan'

thwaites_label = 'Thwaites'
thwaites_linestyle = '-.'

thwaites_lin_label = 'Thwaites (Linear)'
thwaites_lin_linestyle = '--'

thwaites_analytical_label = 'Thwaites (Analytical)'
thwaites_analytical_linestyle = (0, (3, 1, 1, 1, 1, 1))

michel_linestyle = ':'
michel_label='Michel Condition ($f(Re_x)$)'

retheta_linestyle = '-'
retheta_label = r'$Re_\theta$'

#For plotting errors
error_label = 'Relative Error'
theta_linestyle = ':'
theta_label = r'$\theta$'

del_label = '$\delta$'
del_linestyle = '-.'

c_f_label = '$c_f$'
c_f_linestyle = '--'

h_label = '$H$'
h_linestyle = (0, (3, 1, 1, 1, 1, 1))

#for plotting XFOIL
xfoil_linestyle = ':'
xfoil_label = 'XFOIL'

pybl_label = 'PyBL'
pybl_linestyle = '-.'

#for plotting fp, mapg
data_marker = '*'
data_label = 'Tabulated Values'

spline_linestyle = ':'
spline_label = '$u_e$'

smooth_label = 'Smoothed $u_e$'
smooth_linestyle = '-.'

der_label = 'Smoothed derivative'
der_linestyle = '--'

# smoothdata_marker

def abs_rel_err(truth,ans):
    return abs((truth-ans)/truth)



def plot_BL_params(theta=None,c_f=None,h=None,delta=None, x=None, theta_x=None , c_f_x=None, 
                   h_x=None,delta_x = None,fig=None,axs=None,label=None,file=None,sim=None,
                   color=None,linestyle=None,marker=None,last=False):
    good_legend_style = 'legend style={at={(-.15,-0.4)},anchor=south}'
    big_legend_style = 'legend style={at={(-.15,-0.6)},anchor=south}'
    legend_style = 'legend style'
    
    if color is None:
        color = 'k' #black
    if linestyle is None and marker is None:
        linestyle = '-'
        
    if sim is not None:
        param_list = [sim.theta,sim.c_f,sim.h,sim.del_star]
    else:
        param_list = [theta,c_f,h,delta,]
    if axs is None:
        fig,axs = plt.subplots(2,2)
        # fig.tight_layout()
        # fig.subplots_adjust(right=1,wspace=.4, hspace=.4)
        fig.subplots_adjust(right=1,wspace=.4, hspace=.4)
    
    param_title_list = [r'$\theta$ (m)','$c_f$','$H$','$\delta$ (m)',]
    param_x_list = [theta_x,c_f_x,h_x,delta_x,]
    param_loc_list = [(0,0),(1,0),(1,1),(0,1)]
    for i_param, param in enumerate(param_list):
        ax = axs[param_loc_list[i_param][0],param_loc_list[i_param][1]]
        if param_x_list[i_param] is not None:
            plotx = param_x_list[i_param]
        # elif x!=None:
        elif x is not None:
            plotx=x
        else:
            plotx=None
        if sim is not None:
            param=param(plotx)
        if marker is not None:
            # ax.plot(plotx,param,label=label,color=color,marker=marker)
            ax.plot(plotx,param,marker,label=label,color=color)

        else:
            ax.plot(plotx,param,label=label,color=color,linestyle=linestyle,marker=marker)

        ax.set(xlabel='$x$(m)',ylabel=param_title_list[i_param])
        ax.grid(True)
        if last==True and param_loc_list[i_param]==(1,1):
            handles, labels = ax.get_legend_handles_labels()
            if len(labels)>2:
                good_legend_style = big_legend_style
            ax.legend()
    # ax.plot(plotx,param,label=label,color=color,linestyle=linestyle,)
    
    # if last==True:
    #     handles, labels = ax.get_legend_handles_labels()
    #     # fig.legend(handles, labels, loc='upper center') 
    #     # fig.legend(handles, labels)
    #     ax.legend()
    if file is not None:
        tikzplotlib.save(
        'figures/'+file+'.tex',
        axis_height = '\\subplotfigH',
        axis_width = '\\subplotfigW',
        extra_groupstyle_parameters={'horizontal sep=.25\\subplotfigW','vertical sep=.25\\subplotfigW',},
        # extra_axis_parameters=['legend style={at={(-.5,0.9)},anchor= outer north east}']) #'legend style={at={(.5,0.9)},anchor=north east}'
        extra_axis_parameters=[good_legend_style])
        #legend style={at={(-.15,-0.4)},anchor=south},
        made_file = open('figures/'+file+'.tex', "r")
        lines = made_file.readlines()
        made_file.close()
        remake_file = open('figures/'+file+'.tex', "w")
        for line in lines:
            if good_legend_style in line: #doesn't copy over matplotib's legend position choice
                remake_file.write(line)
            elif line.count('{')!=line.count('}'):
                print('{/=}, keeping line')
                remake_file.write(line)
            elif legend_style not in line:
                remake_file.write(line)

                
        remake_file.close()
    
    return [fig,axs]
        
