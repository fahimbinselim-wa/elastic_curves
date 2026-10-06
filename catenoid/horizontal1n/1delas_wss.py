from fenics import *
import sys, math
import numpy as np
import time
from datetime import date

from ufl import cosh
from ufl import tanh
from ufl import sinh

#-------------------------------------------------     
#create mesh and define function space
#create mesh and define function space
x1=0
x2=1
mesh = IntervalMesh(1000,x1,x2)
xmesh=mesh.coordinates()
x = SpatialCoordinate(mesh)

# Define test functions
P1 = FiniteElement('CG', interval, 2)
P2 = FiniteElement('CG', interval,2)


N_cur = int(9)
element1 = MixedElement([P1]*N_cur) #MixedElement([P1,P1.....])
V = FunctionSpace(mesh, element1)  


#-------------------------------------------------
# Define test functions

#Curve
f1, f2, f3, f4, f5, f6, f7, f8, f9 = TestFunctions(V) 
# 
#---------------Initial Conditions------------------


#-------------------Reference Functions---------------


ux1 = -1
ux = str(ux1)

gbs = Constant(1.0) #sqrt(bar{g})

kb =  1/cosh(ux1)  #0.648054#2736638854

taub = Constant(0.)


#--------------------Initial Curve Functions---------------
#Inextensible 
gs = gbs

dl='+0.1e-7'

kk1 = 2

kk = str(kk1)

kg0 = '-tanh('+ux+')/cosh('+ux+')'#3475645731

kn10 ='(-pow(1/cosh('+ux+'),2))'#3416140261

kn20 = '-'+kn10

taug0 ='0.'#+dl

phi0 ='(pi/2 + acos('+str(kb)+'))'

l10 = '-0.'#+dl

l20 = '-'+kn10+'*'+l10

u0 = 'x[0]'#'pi/'+kk

v0 = str(ux1)


#--------------------Initial Curve Functions---------------

#-----------------------------------------------------

fc0 = Expression(( kg0, kn10,  kn20, taug0, phi0, l10, l20, u0, v0), 
              degree=1 )
              
              #((..vector..), degree of element, userdefined parameters)

fcn = interpolate(fc0, V) #interpolatedd from initial strings

#solved functions are Function()
#Curve
fc = Function(V)  #guess, solution


#----------------------

#vector function of the curve

kgn, kn1n, kn2n, taugn, phin, l1n, l2n, uun, vvn,   = split(fcn); #initial guess
kg, kn1, kn2, taug, phi, l1, l2, uu, vv = split(fc); #new solution



#----------------BC-------------

boundary_markers = MeshFunction('size_t', mesh, mesh.topology().dim()-1)


def boundary_L(x, on_boundary):
    tol = 1E-14
    return on_boundary and near(x[0], x1, tol)


def boundary_R(x, on_boundary):
    tol = 1E-14
    return on_boundary and near(x[0], x2, tol)

#-------------------Weak Formulation-----------------------

#gs=gbs (known)

kgb = kb*cos(phi)
kn1b = kb*sin(phi)
taugb = taub - (phi.dx(0))
#taugbn = taub - 1/gs*(thn.dx(0))
g2 = gs*gs


    
bc_u0 = DirichletBC(V.sub(7), Constant(0.), boundary_L)

bc_v0 = DirichletBC(V.sub(8), ux1, boundary_L)


bc = [ bc_u0,  bc_v0]#,bc_tg2,bc_tg1]
 #bc_kg, bc_phi, bc_kn1, bc_taug, 
#-------------------Weak Formulation-----------------------


a_uu = pow(cosh(vv),2)
a_uv = Constant(0)
a_vv = a_uu
a_uun = pow(cosh(vv),2)
a_vvn = a_uun

b_uu = Constant(-1)
b_uv = Constant(0)
b_vv = Constant(1)
b_vvn = Constant(-1)

kG = -1/pow(cosh(vv),4)
H = Constant(0.0)



#-----------------------------------------------------------------

gb = gs
gb2 = gs
gm = 10e-1
Y = 1
nu = 0

w_g = 0 #3*(1- (gb2/g2))*(gb2/(g2*g2))

w_kg = 1/3*(gm**2)*(kg - kgb)

w_kn1 = 1/3*(gm**2)*( kn1 - kn1b )

w_taug = 2/3*(gm**2)/(1+nu)*( taug - taugb )

dm = 1#(H-kn1)

m1 = - l1*kn2 - l2
m2 = 2*l1*taug

w_kn1x = w_kn1 + m1
w_taugx = w_taug + m2

b1 = 0#1/gs*( 2*g2*w_g - taug*w_taugx  - kg*w_kg - kn1*w_kn1x)
b1x = 0#1/gs*( 2*g2*w_g  - kg*w_kg - kn1*w_kn1x)
b2 = ((1/(g2))*w_kg).dx(0) - (1/gs)*(w_kn1x*taug - w_taugx*kn1)
b3 = ((1/(g2))*w_kn1x).dx(0) + (1/gs)*( w_kg*taug - w_taugx*kg)


#---------Euler-Lagrange------------

#trial function kg

#g
#Fga = (  (fg)*gs*(b1x).dx(0) - ( w_taugx/gs*(taug).dx(0) - taug*(kg*w_kn1x - kn1*w_kg) )*fg + fg*gs*( kg*b2 + kn1*b3  ) )
#F0 = Fga*dm*dm*dx

#kg
F1a = ( - f1.dx(0)*b2 - f1*( gs*kg*b1 + gs*taug*b3  ) )
F1 = F1a*dm*dm*dx

#kn1
F2a = ( - f2.dx(0)*b3 - f2*( gs*kn1*b1 - gs*taug*b2  ) )
F2 = F2a*dm*dm*dx

#kn2
F3 = (kn2 - (2*H - kn1))*f3*dx

#tg
F4 = ( (kG - kn1*kn2 + taug*taug ))*f4*dx #no kn1n

#l1
F5 = ( m2.dx(0)*f5 + m1*gb*kg *f5)*dm*dm*dx - w_taugx/gs*f5*ds(1) + w_taugx/gs*f5*ds(2)

#l2
F6 = ( l2 + kn1*l1 )*f6*dx

#th
F7a = ( - f7.dx(0)*w_taug + f7*( w_kn1*kgb - w_kg*kn1b) )
F7 = F7a*dx
#-------------------------------------------

gm_uvv = -cos(uun)*sin(uun)
gm_vuv = cos(uun)/sin(uun)
gm_vvu = gm_vuv


kb2 = 1
L = 16

g2s =  ( inner(grad(uu), grad(uu))*a_uu + 2*inner(grad(uu), grad(vv))*a_uv  + inner(grad(vv), grad(vv))*a_vv ) 



taugs = (1/sqrt(a_uu*a_vv - pow(a_uv,2)))*(1/g2)*(inner(grad(uu), grad(uu))*(b_uv*a_uu - b_uu*a_uv)  + inner(grad(uu), grad(vv))*(b_vv*a_uu - b_uu*a_vv) 
+  inner(grad(vv), grad(vv))*(b_vv*a_uv - b_uv*a_vv))

kn2s = 1/g2*((a_uu/a_vv)*inner(grad(uu), grad(uu))*b_vv  - 2*inner(grad(uu), grad(vv))*b_uv + (a_vv/a_uu)*inner(grad(vv), grad(vv))*b_uu )

kn1s =  1/g2*( inner(grad(uu), grad(uu))*b_uu + 2*inner(grad(uu), grad(vv))*b_uv + inner(grad(vv), grad(vv))*b_vv  ) 


F8 = ((gs*gs) - 1/L*g2s)*f8*dx

#F6 = (1/L*kgs-  kg)*f6*dx#+ kgb 


F9 = (kn2 - 1/L*kn2s)*f9*dx

F9 = (taug - 1/L*taugs)*f9*dx
#F9 = ((g2)*kn1 - 1/L*kn1s)*f9*dx
#-------------------------------------------






#-------------------------------------------------

Fc = (F1 +  F2 + F3 + F4+ F5 + F6 + F7 + F8 + F9 )

#-------------------------------------------------

#-----------------------

dgc1 = TrialFunction(V)

fc.assign(fcn)



Lc = Fc
ac = -derivative(Lc, fc, dgc1)

dfc = Function(V)
dkg, dkn1, dkn2, dtaug, dphi, dl1, dl2, duu, dvv = split(dfc);

#-------------------------------------------------

tol = 1.0E-6
iter = 0
maxiter = 55
eps = 1.0
nframe = 100
# u_k must have right boundary conditions here
p = Progress("Looping", nframe) #since would take a long timem, to check the progress


bc_left = [
    DirichletBC(V.sub(i), Constant(0.0), "near(x[0],0)")
    for i in [7,8] 
]


while eps > tol and iter < maxiter:
    iter += 1
    print(iter, 'iteration')
    

    A, b = assemble_system(ac, Lc, bcs=bc_left)#, [bc_dkn21, bc_dkn22]) 
    solve(A, dfc.vector(), b)    
    eps = np.linalg.norm(dfc.vector()[:], ord=np.Inf)
    print('Norm:', eps)
    fc.assign(fc + 0.25*dfc)




dgc = TrialFunction(V)
#fc.assign(fcn)
    


J = derivative(Fc, fc, dgc)  


problem = NonlinearVariationalProblem(Fc, fc, bcs=bc, J=J)
solver  = NonlinearVariationalSolver(problem)


prm = solver.parameters
prm['nonlinear_solver']='newton'
prm["newton_solver"]["absolute_tolerance"]= 1e-7#1e-9  5921 secs
prm["newton_solver"]["relative_tolerance"] = 1e-6 #1e-7
prm["newton_solver"]["maximum_iterations"] = 20
prm["newton_solver"]["error_on_nonconvergence"]=False

(Nit, conv)=solver.solve()



today = date.today()
d3 = today.strftime("%y_%m_%d")


filenamekg = 'kg_'+str(kk)+'_'+d3+'.txt'
filenamekn1 = 'kn1_'+str(kk)+'_'+d3+'.txt'
filenamekn2 = 'kn2_'+str(kk)+'_'+d3+'.txt'
filenametg = 'taug_'+str(kk)+'_'+d3+'.txt'
filenamephi = 'phi_'+str(kk)+'_'+d3+'.txt'

filenamel1 = 'l1_'+str(kk)+'_'+d3+'.txt'
filenamel2 = 'l2_'+str(kk)+'_'+d3+'.txt'

filenameuu = 'uu_'+str(kk)+'_'+d3+'.txt'
filenamevv = 'vv_'+str(kk)+'_'+d3+'.txt'


print('kg:')
dm1 = V.sub(0).dofmap()
a_nodes = fc.vector().vec()[dm1.dofs()]
print(a_nodes)
np.savetxt(filenamekg, a_nodes,  fmt='%.6e', delimiter='\t')

print('kn1:')
dm1 = V.sub(1).dofmap()
a_nodes = fc.vector().vec()[dm1.dofs()]
print(a_nodes)
np.savetxt(filenamekn1, a_nodes,  fmt='%.6e', delimiter='\t')

print('kn2:')
dm1 = V.sub(2).dofmap()
a_nodes = fc.vector().vec()[dm1.dofs()]
print(a_nodes)
np.savetxt(filenamekn2, a_nodes,  fmt='%.6e', delimiter='\t')

print('taug:')
dm1 = V.sub(3).dofmap()
a_nodes = fc.vector().vec()[dm1.dofs()]
print(a_nodes)
np.savetxt(filenametg, a_nodes,  fmt='%.6e', delimiter='\t')

print('phi:')
dm1 = V.sub(4).dofmap()
a_nodes = fc.vector().vec()[dm1.dofs()]
print(a_nodes)
np.savetxt(filenamephi, a_nodes,  fmt='%.6e', delimiter='\t')


print('u:')
dm1 = V.sub(7).dofmap()
a_nodes = fc.vector().vec()[dm1.dofs()]
print(a_nodes)
np.savetxt(filenameuu, a_nodes,  fmt='%.6e', delimiter='\t')

print('v:')
dm1 = V.sub(8).dofmap()
a_nodes = fc.vector().vec()[dm1.dofs()]
print(a_nodes)
np.savetxt(filenamevv, a_nodes,  fmt='%.6e', delimiter='\t')



print('l1:')
dm1 = V.sub(5).dofmap()
a_nodes = fc.vector().vec()[dm1.dofs()]
print(a_nodes)
np.savetxt(filenamel1, a_nodes,  fmt='%.6e', delimiter='\t')

print('l2:')
dm1 = V.sub(6).dofmap()
a_nodes = fc.vector().vec()[dm1.dofs()]
print(a_nodes)
np.savetxt(filenamel2, a_nodes,  fmt='%.6e', delimiter='\t')

