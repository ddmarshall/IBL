from analytical_thwaites_wedge import wedgeflow
from flatplate_mild_pressure_gradient import head_fp_mpg
import matplotlib.pyplot as plt

#thwaites comparison of fits and spline
wedgeflow(alpha=0,r0=0.05,npts=50,Vinf=1,nu=1.45E-5,n_plot=100,theta0=None,name='thwaites_fs_fp')
wedgeflow(.5,0.05,150,1,1.45E-5,100,'thwaites_fs_angle')
# wedgeflow(.5,0.05,250,0.01,1.45E-5,100,'thwaites_fs_angle_bad')
# wedgeflow(1,0.05,100,10,1.45E-5,100,theta0=None,name='thwaites_fs_stag')
# wedgeflow(1,0.2,25,1,1.45E-5,50,theta0=0,name='thwaites_fs_stag')
wedgeflow(1.000,0.05,250,1,1.45E-5,50,theta0=None,name='thwaites_fs_stag')
head_fp_mpg()



plt.show()
pass
#xfoil



