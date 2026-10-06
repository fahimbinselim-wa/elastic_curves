from fenics import *
import sys, math
import numpy as np
import time
from datetime import date


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

R = 1
RR = str (R)
c = 1
cc = str (c)
 
gbs = Constant (1.0) #sqrt (bar {g})

kb = Constant (0.)

taub = Constant (0.)


#-------------------- Initial Curve Functions-------------- - \
#Inextensible
gs = gbs

kk1 = 2

kk = str (kk1)

dl = ' + 0.1 e - 7'

kg0 = '0.' # + dl

kn10 = '0.0'

kn20 = '0.' #+ kn10

taug0 = '-1/(1+x[0]*x[0])' # + dl

phi0 = 'pi/2 + acos(sqrt(1/(1+x[0]*x[0])))'# + kk

l10 = ' 0.' # + dl

l20 = ' - ' + kn10 + '*' + l10

u0 = 'x[0]'

v0 = '1'




#-------------------- Initial Curve Functions-------------- - #---------------------------------------------------- \

fc0 = Expression(( kg0, kn10,  kn20, taug0, phi0, l10, l20, u0, v0), 
              degree=1 )
              
              #((..vector..), degree of element, userdefined parameters)

fcn = interpolate(fc0, V) #interpolatedd from initial strings

#solved functions are Function()
#Curve
fc = Function(V)  #guess, solution


#----------------------

#vector function of the curve

kgn, kn1n, kn2n, taugn, phin, l1n, l2n, uun, vvn   = split(fcn); #initial guess
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


u1= Constant(0.5)
u2=  Constant(1.)
    
bc_u0 = DirichletBC(V.sub(7),u1 , boundary_L)

bc_v0 = DirichletBC(V.sub(8),u2, boundary_L)


bc = [ bc_u0,  bc_v0]#,bc_tg2,bc_tg1]
#
#------------------ - Weak Formulation---------------------- - 
a_uu = Constant (1.0)
a_uv = Constant (0.0)
a_vv = 1 + uu*uu
a_vvn = 1 + uun*uun

b_uu = Constant (0.0)
b_uv = -1/sqrt (a_vv)
b_uvn = -1/sqrt (a_vvn)
b_vv = Constant (0.0)
b_vvn = Constant (0.0)


#---------------------------------------- - 
kG = -pow (1/a_vv, 2)
H = Constant (.0)


#-----------------------------------------------------------------

gb = gs
gb2 = gs
gm = 10e-1
Y = 1
nu = 0

w_g = 0 #3*(1- (gb2/g2))*(gb2/(g2*g2))

w_kg = (gm**2)*(kg - kgb)

w_kn1 = (gm**2)*( kn1 - kn1b )

w_taug = 2*(gm**2)/(1+nu)*( taug - taugb )

dm = 1#(H-kn1)

m1 = - l1*kn2 - l2
m2 = 2*l1*taug

w_kn1x = w_kn1 + m1
w_taugx = w_taug + m2


b1 = 0#( 2*w_g - taug*w_taugx/(g2)  - kg*w_kg/(g2) - kn1*w_kn1x/(g2))
b2 = ((gb/(g2))*w_kg).dx(0) - (gb/gs)*(w_kn1x*taug - w_taugx*kn1)
b3 = ((gb/(g2))*w_kn1x).dx(0) + (gb/gs)*( w_kg*taug - w_taugx*kg)


#kg
F1a = ( - f1.dx(0)*b2 - f1*( gb*g2*kg*b1 + gs*taug*b3  ) )
F1 = F1a*dm*dm*dx

#kn1
F2a = ( - f2.dx(0)*b3 - f2*( gb*g2*kn1*b1 - gs*taug*b2  ) )
F2 = F2a*dm*dm*dx

#kn2
F3 = (kn2 - (2*H - kn1))*f3*dx



kG = -pow (1/a_vv, 2)

#tg
F4 = ( (kG - kn1*kn2 + taug*taug ))*f4*dx #no kn1n

F4 = ( (1/a_vv +  taug ))*f4*dx


#l1
F5 = ( m2.dx(0)*f5 + m1*kg*f5)*dm*dx - w_taugx/gs*f5*ds(1) + w_taugx/gs*f5*ds(2)


#l2
F6 = ( l2 + kn1*l1 )*f6*dx

#th
F7a = ( - f7.dx(0)*w_taug + f7*( w_kn1*kgb - w_kg*kn1b) )
F7 = F7a*dx
#-------------------------------------------




lc = 4
L = lc**2

gm_uuu = Constant(0.)
gm_uvv = Constant(0.)

gm_vuu = Constant(0.)
gm_vvv = Constant(0.)

gm_uuv = Constant(0.)
gm_uvu = gm_uuv 
gm_vuv = gm_uuv 
gm_vvu = gm_uuv 




g2s =  1/L*( inner(grad(uu), grad(uu))*a_uu + 2*inner(grad(uu), grad(vv))*a_uv  + inner(grad(vv), grad(vv))*a_vv ) 


kgs = 1/L*sqrt(a_uu*a_vv - pow(a_uv,2))*(1/g2*gs)*(pow((uu.dx(0)),3)*gm_vuu  - pow((vv.dx(0)),3)*gm_uvv + inner(grad(uu),grad(uu))*vv.dx(0)*(2*gm_vuv - gm_uuu) 
- inner(grad(vv),grad(vv))*uu.dx(0)*(2*gm_uuv - gm_vvv)
+ ( uu.dx(0)*div(grad(vv)) - vv.dx(0)*div(grad(uu)) ))

taugs = 1/L*1/(g2)*1/(sqrt(a_uu*a_vv - pow(a_uv,2)))*(inner(grad(uu), grad(uu))*(b_uv*a_uu - b_uu*a_uv)  + inner(grad(uu), grad(vv))*(b_vv*a_uu - b_uu*a_vv) 
+  inner(grad(vv), grad(vv))*(b_vv*a_uv - b_uv*a_vv))

kn2s = 1/L*1/g2*1/(a_uu*a_vv - pow(a_uv,2))*( ( inner(grad(uu),grad(uu))*(a_uu)**2 + 2*inner(grad(uu),grad(vv))*a_uu*a_uv 
+ inner(grad(vv),grad(vv))*(a_uv)**2 )*b_vv + ( inner(grad(uu),grad(uu))*(a_uv)**2 + 2*inner(grad(uu),grad(vv))*a_vv*a_uv 
+ inner(grad(vv),grad(vv))*(a_vv)**2 )*b_uu - 2*( inner(grad(uu),grad(uu))*(a_uv)*(a_uu) + inner(grad(uu),grad(vv))*(a_vv*a_uu+a_uv*a_uv) 
+ inner(grad(vv),grad(vv))*(a_vv)*(a_uv) )*b_uv )

kn1s =  1/L*1/(g2)*( inner(grad(uu), grad(uu))*b_uu + 2*inner(grad(uu), grad(vv))*b_uv + inner(grad(vv), grad(vv))*b_vv  ) 




F8 = ( g2s - g2)*f8*dx

#F6 = (kgs-  kg)*f6*dx#+ kgb 


F9 = (kn2 - kn2s)*f9*dx

#F9 = (taugs  - taug)*f9*dx
F9 = (kn1 - kn1s)*f9*dx
#-------------------------------------------------

Fc = (F1 +  F2 + F3 + F4+ F5 + F6 + F7 + F8 + F9 )

#-------------------------------------------------

energy1 = np.pi
fca = Function(V)
kga, kn1a, kn2a, tauga, phia, l1a, l2a, uua, vva = split(fca); 

print("------------------------------------------")



energy1 = np.pi
fca = Function(V)
kga, kn1a, kn2a, tauga, phia, l1a, l2a, uua, vva = split(fca); 


print("------------------------------------------")


fc.assign(fcn)  

# -----------------------------------------
# Wrap your FEniCS solve in a function
# -----------------------------------------
def compute_energy(params):
    """
    Run FEniCS solver with modified boundary values and return total bending energy.
    params: array-like [uu_right, vv_right] or any boundary parameters you want to vary.
    """

    global energy1


    #-----------------------
    dgc1 = TrialFunction(V)

    Lc = Fc
    ac = -derivative(Lc, fc, dgc1)

    dfc = Function(V)
    dkg, dkn1, dkn2, dtaug, dphi, dl1, dl2, duu, dvv = split(dfc);

    #-------------------------------------------------

    tol = 1.0E-6
    iter = 0
    maxiter = 35
    eps = 1.0
    nframe = 100
    # u_k must have right boundary conditions here
    p = Progress("Looping", nframe) #since would take a long timem, to check the progress


    bc_left = [
        DirichletBC(V.sub(i), Constant(0.0), boundary_L)
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

    u0s, v0s = params



        
    bc_u0 = DirichletBC(V.sub(7), Constant(u0s), boundary_L)

    bc_v0 = DirichletBC(V.sub(8), Constant(v0s), boundary_L)


    bc = [bc_u0,  bc_v0]#,bc_tg2,bc_tg1]

    #-----------------------


    dgc = TrialFunction(V)
    #fc.assign(fcn)


    J = derivative(Fc, fc, dgc)  



    problem = NonlinearVariationalProblem(Fc, fc, bcs=bc, J=J)
    solver  = NonlinearVariationalSolver(problem)


    prm = solver.parameters
    prm['nonlinear_solver']='newton'
    prm["newton_solver"]["absolute_tolerance"]= 1e-9#1e-9  5921 secs
    prm["newton_solver"]["relative_tolerance"] = 1e-7 #1e-7
    prm["newton_solver"]["maximum_iterations"] = 20
    prm["newton_solver"]["error_on_nonconvergence"]=False

    (Nit, conv)=solver.solve()


    if conv:
        fcn.assign(fc)

    # Compute bending energy
    kg_f, kn1_f, kn2_f, taug_f, phi_f, l1_f, l2_f, uu_f, vv_f = fc.split()
    
    kgb_f = kb*cos(phi_f)
    kn1b_f = kb*sin(phi_f)
    taugb_f = taub - (phi_f.dx(0))
    
    W_bending = (  gm**2/3 *( (kg_f - kgb_f)**2 + (kn1_f - kn1b_f)**2 + 2/(1+nu)*(taug_f - taugb_f)**2 ) ) 
    
    
    energy = assemble(W_bending*dx)
    

    
    if energy < energy1:
        energy1 = energy
        fca.assign(fc)
    
    print(f"Energy for u0={u0s:.6f}, v0={v0s:.6f} -> {energy:.6e}")
        
    fc.assign(fcn)   
    
    return energy




from scipy.optimize import minimize

# Initial guess for boundary parameters
x0 = [u1, u2]  # starting uu_right, vv_right

# Run optimization
result = minimize(compute_energy, x0, method='Nelder-Mead',
                  options={'maxiter': 2, 'disp': True})

print("\n---- Optimization results ----")
print("Optimal parameters:", result.x)
print("Minimum energy:", result.fun)



fc.assign(fca)  
bf = result.x
compute_energy(bf)

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




kg_f, kn1_f, kn2_f, taug_f, phi_f, l1_f, l2_f, uu_f, vv_f = fc.split()

kgb_f = kb*cos(phi_f)
kn1b_f = kb*sin(phi_f)
taugb_f = taub - (phi_f.dx(0))/gs

W_bending = ( gm**2/12 *( (kg_f - kgb_f)**2 + (kn1_f - kn1b_f)**2 + 2/(1+nu)*(taug_f - taugb_f)**2 ) - l1_f*( kG - (kn1_f*kn2_f - taug_f**2) )  - l2_f*( 2*H - (kn1_f+kn2_f) ) )

energy = assemble(W_bending*dx)

print(f"Energy  -> {energy:.6e}")






# ----------------------------------------------------------
# Save mapping functions
# ----------------------------------------------------------

# 1. Project kgs to CG2 space
Q = FunctionSpace(mesh, "CG", 2)
kgs_proj = project(kgs, Q)

dm_kgs = Q.dofmap()
kgs_nodes = kgs_proj.vector().get_local()[dm_kgs.dofs()]

filename_kgs = "kgs_.txt"
np.savetxt(filename_kgs, kgs_nodes, fmt="%.6e", delimiter="\t")

# ----------------------------------------------------------
Q = FunctionSpace(mesh, "CG", 2)
kn1s_proj = project(kn1s, Q)

dm_kn1s = Q.dofmap()
kn1s_nodes = kn1s_proj.vector().get_local()[dm_kn1s.dofs()]

filename_kn1s = "kn1s_.txt"
np.savetxt(filename_kn1s, kn1s_nodes, fmt="%.6e", delimiter="\t")

# ----------------------------------------------------------
Q = FunctionSpace(mesh, "CG", 2)
kn2s_proj = project(kn2s, Q)

dm_kn2s = Q.dofmap()
kn2s_nodes = kn2s_proj.vector().get_local()[dm_kn2s.dofs()]

filename_kn2s = "kn2s_.txt"
np.savetxt(filename_kn2s, kn2s_nodes, fmt="%.6e", delimiter="\t")

# ----------------------------------------------------------

Q = FunctionSpace(mesh, "CG", 2)
taugs_proj = project(taugs, Q)

dm_taugs = Q.dofmap()
taugs_nodes = taugs_proj.vector().get_local()[dm_taugs.dofs()]

filename_taugs = "taugs_.txt"
np.savetxt(filename_taugs, taugs_nodes, fmt="%.6e", delimiter="\t")