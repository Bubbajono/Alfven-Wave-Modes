import math
import numpy as np
import matplotlib
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import matplotlib.pyplot as plt
import pylab

print("")
print("For new boundary")
print("")

"""For printing all of matrix"""
np.set_printoptions(threshold=np.inf)

radius = 1.0
class vec2D:        #Class for x,y points in grid
	def __init__(self,x,y):
		self.x = x
		self.y = y
	def norm(self):
		self.x = self.x
		return  math.sqrt(self.x*self.x+self.y*self.y)
	def withinDomain(self):   #Inside consition
		return self.norm() <= radius    
	def vprint(self):
		print(self.x, self.y)

		
GridWidth = 1.0
def getVec2D(i, j, GridWidth, h):  #get x and y corresponding to i,j
      x = -GridWidth + i * h
      y = -GridWidth + j * h    
      return vec2D(x, y)

k = 0.1234    #0.1234 
C = 1j * k    #ik

Nx = 110    #Number of grid points in x and y
Ny = Nx
h = (2*GridWidth) / (Nx-1)    #calculate gridpoint spacing
print("h: ",h)
g=h/2

gridArrayIJ = np.empty([Nx, Ny],dtype=int)     #Grid of i,j points
gridArrayP = []
Np=0

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~Functions for B_0 and rho~~~~~~~~~~~~~~~~~~~~~~# 

def B0x(x,y):      #x-component of B_0
        #r_2 = (x**2)+(y**2)
        #return y*np.exp(-r_2)
        return 0

        
def B0y(x,y):      #y-component of B_0
        #r_2 = (x**2)+(y**2)
        #return -x*np.exp(-r_2)
        return 0

def B0z(x,y):      #z-component of B_0
        #return 5
        return 212.5
        
def rho(x,y):       #Plasma density
        return 6.2354     #first term only
        #return 6.2354*(2.0-(x**2)-(y**2))
        #return 6.2354*(3.0-(((x**2)+(y**2))**2.0))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~# 

B_0 = []
for j in range(0, Ny):        #INITIALLY SETTING UP GRID ON UNIT CIRCLE DOMAIN
    for i in range(0, Nx):
        myVec = getVec2D(i, j, GridWidth, h)
        if(myVec.withinDomain()):
            gridArrayIJ[j, i] = Np    
            gridArrayP.append(vec2D(i,j))
            
            vecIJ = vec2D(i,j)
            x = getVec2D(vecIJ.x, vecIJ.y, GridWidth, h).x   #get x and y
            y = getVec2D(vecIJ.x, vecIJ.y, GridWidth, h).y
            
            B_0.append(B0x(x,y))     #Get 3N vector for B_0
            B_0.append(B0y(x,y))
            B_0.append(B0z(x,y))

          
            Np = Np + 1     #counting number of points
        else:
            gridArrayIJ[j, i] = -1
#print(gridArrayIJ)
print("")
print("Np = ", Np)
print("")
print("Curl Op has", 3*Np,"by", 3*Np, "elements.")
print("")
print("Number of grid points", Nx, "by", Ny)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("")
print("")
print("")

print("Now setting up rho matrix for dividing by rho")
rho_rows=[]
rho_cols=[]
rho_vals=[]
for p in range(0,Np):
        M_find = 3*p
        
        vecIJ = gridArrayP[p]
        x = getVec2D(vecIJ.x, vecIJ.y, GridWidth, h).x   #get x,y
        y = getVec2D(vecIJ.x, vecIJ.y, GridWidth, h).y

        one_over_rho = 1.0/(rho(x,y))

        rho_rows.append(M_find)     #row of matrix block
        rho_cols.append(M_find)     #column of matrix block
        rho_vals.append(one_over_rho)    #assign value to element

        rho_rows.append(M_find+1)
        rho_cols.append(M_find+1)
        rho_vals.append(one_over_rho)
        
        rho_rows.append(M_find+2)
        rho_cols.append(M_find+2)
        rho_vals.append(one_over_rho)
        
oneoverrho = sparse.csc_matrix((rho_vals, (rho_rows, rho_cols)), shape=(3*Np, 3*Np))

print("")
print("Now setting up Bcross operator matrix.....")
Bcross_rows=[]   
Bcross_cols=[]
Bcross_vals=[]       
for p in range(0, Np):
    M_find = 3*p
    
    Bcross_rows.append(M_find)     #row of matrix block    
    Bcross_cols.append(M_find+1)     #column of matrix block
    Bcross_vals.append(-B_0[(M_find+2)])    #assign value to element
    
    Bcross_rows.append(M_find)
    Bcross_cols.append(M_find+2)
    Bcross_vals.append(B_0[(M_find+1)])
    
    Bcross_rows.append(M_find+1)
    Bcross_cols.append(M_find)
    Bcross_vals.append(B_0[(M_find+2)])    
    
    Bcross_rows.append(M_find+1)
    Bcross_cols.append(M_find+2)
    Bcross_vals.append(-B_0[M_find])      
    
    Bcross_rows.append(M_find+2)
    Bcross_cols.append(M_find)
    Bcross_vals.append(-B_0[(M_find+1)])    
    
    Bcross_rows.append(M_find+2)
    Bcross_cols.append(M_find+1)
    Bcross_vals.append(B_0[M_find])        
    
Bcross = sparse.csc_matrix((Bcross_vals, (Bcross_rows, Bcross_cols)), shape=(3*Np, 3*Np))
 #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
print("")
print("Now setting up J0 via a finite difference approach")   #curl of B_0
J_0=[]
for p in range(0,Np):
    vecIJ = gridArrayP[p]
    x = getVec2D(vecIJ.x, vecIJ.y, GridWidth, h).x
    y = getVec2D(vecIJ.x, vecIJ.y, GridWidth, h).y
    xplusone_x = getVec2D(vecIJ.x+1, vecIJ.y, GridWidth, h).x   #Probably a better way
    xplusone_y = getVec2D(vecIJ.x+1, vecIJ.y, GridWidth, h).y
    xminusone_x = getVec2D(vecIJ.x-1, vecIJ.y, GridWidth, h).x
    xminusone_y = getVec2D(vecIJ.x-1, vecIJ.y, GridWidth, h).y
    yplusone_x = getVec2D(vecIJ.x, vecIJ.y+1, GridWidth, h).x
    yplusone_y = getVec2D(vecIJ.x, vecIJ.y+1, GridWidth, h).y
    yminusone_x = getVec2D(vecIJ.x, vecIJ.y-1, GridWidth, h).x
    yminusone_y = getVec2D(vecIJ.x, vecIJ.y-1, GridWidth, h).y

    J0x = B0z(yplusone_x,yplusone_y)- B0z(yminusone_x,yminusone_y)
    J0x = (1/(2*h)) * J0x

    J0y = B0z(xplusone_x,xplusone_y)- B0z(xminusone_x,xminusone_y)
    J0y = (1/(2*h)) * J0y

    J0z = B0y(xplusone_x,xplusone_y)- B0y(xminusone_x,xminusone_y)
    J0z = J0z - (B0x(yplusone_x,yplusone_y)- B0x(yminusone_x,yminusone_y))
    J0z = (1/(2*h)) * J0z

    J_0.append(J0x)     #x cpt
    J_0.append(J0y)    #y cpt
    J_0.append(J0z)    #z cpt

 #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        
print("")
print("Now setting up the Jcross matrix")
Jcross_rows=[]   
Jcross_cols=[]
Jcross_vals=[]       
for p in range(0, Np):
    M_find = 3*p
    
    Jcross_rows.append(M_find)     #row of matrix block  
    Jcross_cols.append(M_find+1)     #column of matrix block  
    Jcross_vals.append(-J_0[(M_find+2)])     #val of matrix block  
    
    Jcross_rows.append(M_find)
    Jcross_cols.append(M_find+2)
    Jcross_vals.append(J_0[(M_find+1)])
    
    Jcross_rows.append(M_find+1)
    Jcross_cols.append(M_find)
    Jcross_vals.append(J_0[(M_find+2)])    
    
    Jcross_rows.append(M_find+1)
    Jcross_cols.append(M_find+2)
    Jcross_vals.append(-J_0[M_find])      
    
    Jcross_rows.append(M_find+2)
    Jcross_cols.append(M_find)
    Jcross_vals.append(-J_0[(M_find+1)])    
    
    Jcross_rows.append(M_find+2)
    Jcross_cols.append(M_find+1)
    Jcross_vals.append(J_0[M_find])
    
Jcross = sparse.csc_matrix((Jcross_vals, (Jcross_rows, Jcross_cols)), shape=(3*Np, 3*Np))
 #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

graph_help = []    
Crows=[]   
Ccols=[]
Cvals=[]
print("")
print("Now setting up Curl (Dirichlet) operator matrix.....")
for p in range(0, Np):
    
    graph_help.append(p)
    
    M_find = 3*p    #So don't have to keep calculating 3*p
    right = 3      #For jumping to the 3x3 next to current
    left = -3
    
    vecIJ = gridArrayP[p]
    x = getVec2D(vecIJ.x, vecIJ.y, GridWidth, h).x
    y = getVec2D(vecIJ.x, vecIJ.y, GridWidth, h).y
    
    currentpoint = (gridArrayIJ[vecIJ.y, vecIJ.x])

    Crows.append(M_find)   #i,j
    Ccols.append(M_find+1)
    Cvals.append(-C)
    Crows.append(M_find+1)
    Ccols.append(M_find)
    Cvals.append(C)
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#    
    q = gridArrayIJ[vecIJ.y, vecIJ.x+1]   #i+1,j
    if(q >= 0):
            Crows.append(M_find+1)
            Ccols.append(M_find+right+2)
            Cvals.append(-1/(2*h))
            Crows.append(M_find+2)
            Ccols.append(M_find+right+1)
            Cvals.append(1/(2*h))
    
    q = gridArrayIJ[vecIJ.y, vecIJ.x-1]   #i-1,j 
    if(q >= 0):
            Crows.append(M_find+1)
            Ccols.append(M_find+left+2)
            Cvals.append(1/(2*h))
            Crows.append(M_find+2)
            Ccols.append(M_find+left+1)
            Cvals.append(-1/(2*h))

    q = gridArrayIJ[vecIJ.y+1, vecIJ.x]   #i,j+1
    if(q >= 0):
            jump = 3 * (q - currentpoint)

            Crows.append(M_find)
            Ccols.append(M_find+jump+2)
            Cvals.append(1/(2*h))
            Crows.append(M_find+2)
            Ccols.append(M_find+jump)
            Cvals.append(-1/(2*h))

    q = gridArrayIJ[vecIJ.y-1, vecIJ.x]   #i,j-1
    if(q >= 0):
            jump = (3) * (q - currentpoint)

            Crows.append(M_find)
            Ccols.append(M_find+jump+2)
            Cvals.append(-1/(2*h))
            Crows.append(M_find+2)
            Ccols.append(M_find+jump)
            Cvals.append(1/(2*h))

 #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
Curl = sparse.csc_matrix((Cvals, (Crows, Ccols)), shape=(3*Np, 3*Np)) #form sparse matrix
 #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

del(Crows)
del(Ccols)
del(Cvals)
#===========================================================================#
#===========================================================================#
#===========================================================================#


Crows=[]   
Ccols=[]
Cvals=[]
print("")
print("Now setting up Curl (Mixed Dirichlet,Neumann) operator matrix .....")
for p in range(0, Np):
    
    M_find = 3*p    #So don't have to keep calculating 3*p
    right = 3      #For jumping to the 3x3 next to current
    left = -3
    
    vecIJ = gridArrayP[p]
    x = getVec2D(vecIJ.x, vecIJ.y, GridWidth, h).x   #get x,y
    y = getVec2D(vecIJ.x, vecIJ.y, GridWidth, h).y
    
    currentpoint = (gridArrayIJ[vecIJ.y, vecIJ.x])

    Crows.append(M_find)   #i,j
    Ccols.append(M_find+1)
    Cvals.append(-C)
    Crows.append(M_find+1)
    Ccols.append(M_find)
    Cvals.append(C)
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#    
    q = gridArrayIJ[vecIJ.y, vecIJ.x+1]   #i+1,j
    if(q >= 0):
            Crows.append(M_find+1)
            Ccols.append(M_find+right+2)
            Cvals.append(-1/(2*h))
            Crows.append(M_find+2)
            Ccols.append(M_find+right+1)
            Cvals.append(1/(2*h))
    
    q = gridArrayIJ[vecIJ.y, vecIJ.x-1]   #i-1,j
    if(q >= 0):
            Crows.append(M_find+1)
            Ccols.append(M_find+left+2)
            Cvals.append(1/(2*h))
            Crows.append(M_find+2)
            Ccols.append(M_find+left+1)
            Cvals.append(-1/(2*h))

    q = gridArrayIJ[vecIJ.y+1, vecIJ.x]   #i,j+1
    if(q >= 0):
            jump = 3 * (q - currentpoint)

            Crows.append(M_find)
            Ccols.append(M_find+jump+2)
            Cvals.append(1/(2*h))
            Crows.append(M_find+2)
            Ccols.append(M_find+jump)
            Cvals.append(-1/(2*h))
    else:
            Crows.append(M_find+2)   #append central stencil point (Neumann)
            Ccols.append(M_find)
            Cvals.append(-1/(2*h))
            
    q = gridArrayIJ[vecIJ.y-1, vecIJ.x]   #i,j-1
    if(q >= 0):
            jump = (3) * (q - currentpoint)

            Crows.append(M_find)
            Ccols.append(M_find+jump+2)
            Cvals.append(-1/(2*h))
            Crows.append(M_find+2)
            Ccols.append(M_find+jump)
            Cvals.append(1/(2*h))
    else:
            Crows.append(M_find+2)   #append central stencil point (Neumann)
            Ccols.append(M_find)
            Cvals.append(1/(2*h))

 #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
CurlPin = sparse.csc_matrix((Cvals, (Crows, Ccols)), shape=(3*Np, 3*Np)) #form sparse matrix
 #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#===========================================================================#
#===========================================================================#
#===========================================================================#


del(Crows)
del(Ccols)
del(Cvals)
#===========================================================================#
#===========================================================================#
#===========================================================================#


Crows=[]   
Ccols=[]
Cvals=[]
print("")
print("Now setting up Curl (Neumann) operator matrix .....")
for p in range(0, Np):
    
    M_find = 3*p    #So don't have to keep calculating 3*p
    right = 3      #For jumping to the 3x3 next to current
    left = -3
    
    vecIJ = gridArrayP[p]
    x = getVec2D(vecIJ.x, vecIJ.y, GridWidth, h).x  #get x,y
    y = getVec2D(vecIJ.x, vecIJ.y, GridWidth, h).y
    
    currentpoint = (gridArrayIJ[vecIJ.y, vecIJ.x])

    Crows.append(M_find)   #i,j
    Ccols.append(M_find+1)
    Cvals.append(-C)
    Crows.append(M_find+1)
    Ccols.append(M_find)
    Cvals.append(C)
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#    
    q = gridArrayIJ[vecIJ.y, vecIJ.x+1]   #i+1,j
    if(q >= 0):
            Crows.append(M_find+1)
            Ccols.append(M_find+right+2)
            Cvals.append(-1/(2*h))
            Crows.append(M_find+2)
            Ccols.append(M_find+right+1)
            Cvals.append(1/(2*h))
    else:
            Crows.append(M_find+2)   #append central stencil point (Neumann)
            Ccols.append(M_find+1)
            Cvals.append(1/(2*h))
    
    q = gridArrayIJ[vecIJ.y, vecIJ.x-1]   #i-1,j
    if(q >= 0):
            Crows.append(M_find+1)
            Ccols.append(M_find+left+2)
            Cvals.append(1/(2*h))
            Crows.append(M_find+2)
            Ccols.append(M_find+left+1)
            Cvals.append(-1/(2*h))
    else:
            Crows.append(M_find+2)   #append central stencil point (Neumann)
            Ccols.append(M_find+1)
            Cvals.append(-1/(2*h))               

    q = gridArrayIJ[vecIJ.y+1, vecIJ.x]   #i,j-1
    if(q >= 0):
            jump = 3 * (q - currentpoint)

            Crows.append(M_find)
            Ccols.append(M_find+jump+2)
            Cvals.append(1/(2*h))
            Crows.append(M_find+2)
            Ccols.append(M_find+jump)
            Cvals.append(-1/(2*h))
    else:
            Crows.append(M_find+2)   #append central stencil point (Neumann)
            Ccols.append(M_find)
            Cvals.append(-1/(2*h))

    q = gridArrayIJ[vecIJ.y-1, vecIJ.x]   #i,j-1
    if(q >= 0):
            jump = (3) * (q - currentpoint)

            Crows.append(M_find)
            Ccols.append(M_find+jump+2)
            Cvals.append(-1/(2*h))
            Crows.append(M_find+2)
            Ccols.append(M_find+jump)
            Cvals.append(1/(2*h))
    else:
            Crows.append(M_find+2)   #append central stencil point (Neumann)
            Ccols.append(M_find)
            Cvals.append(1/(2*h) )

 #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
CurlPin2 = sparse.csc_matrix((Cvals, (Crows, Ccols)), shape=(3*Np, 3*Np))    #for sparse matrix
 #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#===========================================================================#
#===========================================================================#
#===========================================================================#
CCrows=[]   
CCcols=[]
CCvals=[]

print("")
print("Now setting up Curl Curl (Dirichlet) operator matrix.....")
for p in range(0, Np):
    
    
    M_find = 3*p    #So don't have to keep calculating 3*p
    right = 3      #For jumping to the 3x3 next to current
    left = -3
    
    vecIJ = gridArrayP[p]
    x = getVec2D(vecIJ.x, vecIJ.y, GridWidth, h).x
    y = getVec2D(vecIJ.x, vecIJ.y, GridWidth, h).y
    
    currentpoint = (gridArrayIJ[vecIJ.y, vecIJ.x])

    CCrows.append(M_find)
    CCcols.append(M_find)
    CCvals.append((2/h/h)-(C**2))

    CCrows.append(M_find+1)
    CCcols.append(M_find+1)
    CCvals.append((2/h/h)-(C**2))

    CCrows.append(M_find+2)
    CCcols.append(M_find+2)
    CCvals.append((2/h/h)+(2/h/h))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    q = gridArrayIJ[vecIJ.y, vecIJ.x+1]   #i+1,j
    if(q >= 0):
            CCrows.append(M_find)
            CCcols.append(M_find+right+2)
            CCvals.append(C/(2*h))

            CCrows.append(M_find+1)
            CCcols.append(M_find+right+1)
            CCvals.append(-1/h/h)

            CCrows.append(M_find+2)
            CCcols.append(M_find+right)
            CCvals.append(C/(2*h))

            CCrows.append(M_find+2)
            CCcols.append(M_find+right+2)
            CCvals.append(-1/h/h)

    q = gridArrayIJ[vecIJ.y, vecIJ.x-1]    #i-1,j
    if(q >= 0):
            CCrows.append(M_find)
            CCcols.append(M_find+left+2)
            CCvals.append(-C/(2*h))

            CCrows.append(M_find+1)
            CCcols.append(M_find+left+1)
            CCvals.append(-1/h/h)

            CCrows.append(M_find+2)
            CCcols.append(M_find+left)
            CCvals.append(-C/(2*h))

            CCrows.append(M_find+2)
            CCcols.append(M_find+left+2)
            CCvals.append(-1/h/h)
            
    q = gridArrayIJ[vecIJ.y+1, vecIJ.x]   #i,j+1
    if(q >= 0):
            jump = 3 * (q - currentpoint)

            CCrows.append(M_find)
            CCcols.append(M_find+jump)
            CCvals.append(-1/h/h)

            CCrows.append(M_find+1)
            CCcols.append(M_find+jump+2)
            CCvals.append(C/(2*h))

            CCrows.append(M_find+2)
            CCcols.append(M_find+jump+1)
            CCvals.append(C/(2*h))

            CCrows.append(M_find+2)
            CCcols.append(M_find+jump+2)
            CCvals.append(-1/h/h)
            

    q = gridArrayIJ[vecIJ.y-1, vecIJ.x]   #i,j-1
    if(q >= 0):
            jump = (3) * (q - currentpoint)

            CCrows.append(M_find)
            CCcols.append(M_find+jump)
            CCvals.append(-1/h/h)

            CCrows.append(M_find+1)
            CCcols.append(M_find+jump+2)
            CCvals.append(-C/(2*h))

            CCrows.append(M_find+2)
            CCcols.append(M_find+jump+1)
            CCvals.append(-C/(2*h))

            CCrows.append(M_find+2)
            CCcols.append(M_find+jump+2)
            CCvals.append(-1/h/h)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    q = gridArrayIJ[vecIJ.y+1, vecIJ.x+1]   #i+1,j+1
    if(q >= 0):
            jump = 3 * (q - currentpoint)
            CCrows.append(M_find)
            CCcols.append(M_find+jump+1)
            CCvals.append(1/(4*h*h))

            CCrows.append(M_find+1)
            CCcols.append(M_find+jump)
            CCvals.append(1/(4*h*h))
    q = gridArrayIJ[vecIJ.y+1, vecIJ.x-1]   #i-1,j+1
    if(q >= 0):
            jump = 3 * (q - currentpoint)
            CCrows.append(M_find)
            CCcols.append(M_find+jump+1)
            CCvals.append(-1/(4*h*h))

            CCrows.append(M_find+1)
            CCcols.append(M_find+jump)
            CCvals.append(-1/(4*h*h))
    q = gridArrayIJ[vecIJ.y-1, vecIJ.x+1]   #i+1,j-1
    if(q >= 0):
            jump = 3 * (q - currentpoint)
            CCrows.append(M_find)
            CCcols.append(M_find+jump+1)
            CCvals.append(-1/(4*h*h))

            CCrows.append(M_find+1)
            CCcols.append(M_find+jump)
            CCvals.append(-1/(4*h*h))
    q = gridArrayIJ[vecIJ.y-1, vecIJ.x-1]   #i-1,j-1
    if(q >= 0):
            jump = 3 * (q - currentpoint)
            CCrows.append(M_find)
            CCcols.append(M_find+jump+1)
            CCvals.append(1/(4*h*h))
            
            CCrows.append(M_find+1)
            CCcols.append(M_find+jump)
            CCvals.append(1/(4*h*h))


 #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
CurlCurl = sparse.csc_matrix((CCvals,(CCrows,CCcols)), shape=(3*Np, 3*Np))   #form sparse matrix
 #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
del(CCrows)
del(CCcols)
del(CCvals)

#===========================================================================#
#===========================================================================#
#===========================================================================#


CCrows=[]   
CCcols=[]
CCvals=[]
counter1=0
counter2=0
counter3=0
print("")
print("Now setting up Curl Curl (Mixed Dirichlet, Neumann) operator matrix.....")
for p in range(0, Np):
    M_find = 3*p    #So don't have to keep calculating 3*p
    right = 3      #For jumping to the 3x3 next to current
    left = -3

    counter1 = 0
    counter2 = 0
    counter3 = 0
    
    vecIJ = gridArrayP[p]
    x = getVec2D(vecIJ.x, vecIJ.y, GridWidth, h).x
    y = getVec2D(vecIJ.x, vecIJ.y, GridWidth, h).y
    
    currentpoint = (gridArrayIJ[vecIJ.y, vecIJ.x])

    counter1 = counter1 + ((2/h/h)-(C**2))       #(M_find,M_find)    #X
    counter2 = counter2                          #(M_find+2,M_find)  #X
    counter3 = counter3                          #(M_find+1,M_find)  #X

    CCrows.append(M_find+1)       #i,j
    CCcols.append(M_find+1)
    CCvals.append((2/h/h)-(C**2))

    CCrows.append(M_find+2)
    CCcols.append(M_find+2)
    CCvals.append((2/h/h)+(2/h/h))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    q = gridArrayIJ[vecIJ.y, vecIJ.x+1]       #i+1,j
    if(q >= 0):
            CCrows.append(M_find)
            CCcols.append(M_find+right+2)
            CCvals.append(C/(2*h))

            CCrows.append(M_find+1)
            CCcols.append(M_find+right+1)
            CCvals.append(-1/h/h)

            CCrows.append(M_find+2)
            CCcols.append(M_find+right)
            CCvals.append(C/(2*h))

            CCrows.append(M_find+2)
            CCcols.append(M_find+right+2)
            CCvals.append(-1/h/h)
    else:
            counter2 = counter2 + (C/(2*h))   #append central stencil point (Neumann)

    q = gridArrayIJ[vecIJ.y, vecIJ.x-1]        #i-1,j
    if(q >= 0):
            CCrows.append(M_find)
            CCcols.append(M_find+left+2)
            CCvals.append(-C/(2*h))

            CCrows.append(M_find+1)
            CCcols.append(M_find+left+1)
            CCvals.append(-1/h/h)

            CCrows.append(M_find+2)
            CCcols.append(M_find+left)
            CCvals.append(-C/(2*h))

            CCrows.append(M_find+2)
            CCcols.append(M_find+left+2)
            CCvals.append(-1/h/h)
    else:
            counter2 = counter2 + (-C/(2*h))   #append central stencil point (Neumann)
            
    q = gridArrayIJ[vecIJ.y+1, vecIJ.x]       #i,j+1
    if(q >= 0):
            jump = 3 * (q - currentpoint)

            CCrows.append(M_find)
            CCcols.append(M_find+jump)
            CCvals.append(-1/h/h)

            CCrows.append(M_find+1)
            CCcols.append(M_find+jump+2)
            CCvals.append(C/(2*h))

            CCrows.append(M_find+2)
            CCcols.append(M_find+jump+1)
            CCvals.append(C/(2*h))

            CCrows.append(M_find+2)
            CCcols.append(M_find+jump+2)
            CCvals.append(-1/h/h)
    else:
            counter1 = counter1 + (-1/h/h)     #append central stencil point (Neumann)

    q = gridArrayIJ[vecIJ.y-1, vecIJ.x]       #i,j-1
    if(q >= 0):
            jump = (3) * (q - currentpoint)

            CCrows.append(M_find)
            CCcols.append(M_find+jump)
            CCvals.append(-1/h/h)

            CCrows.append(M_find+1)
            CCcols.append(M_find+jump+2)
            CCvals.append(-C/(2*h))

            CCrows.append(M_find+2)
            CCcols.append(M_find+jump+1)
            CCvals.append(-C/(2*h))

            CCrows.append(M_find+2)
            CCcols.append(M_find+jump+2)
            CCvals.append(-1/h/h)
    else:
            counter1 = counter1 + (-1/h/h)    #append central stencil point (Neumann) 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    q = gridArrayIJ[vecIJ.y+1, vecIJ.x+1]       #i+1,j+1
    if(q >= 0):
            jump = 3 * (q - currentpoint)
            CCrows.append(M_find)
            CCcols.append(M_find+jump+1)
            CCvals.append(1/(4*h*h))

            CCrows.append(M_find+1)
            CCcols.append(M_find+jump)
            CCvals.append(1/(4*h*h))
    else:
            counter3 = counter3 + (1/(4*h*h))   #append central stencil point (Neumann)
    q = gridArrayIJ[vecIJ.y+1, vecIJ.x-1]       #i-1,j+1
    if(q >= 0):
            jump = 3 * (q - currentpoint)
            CCrows.append(M_find)
            CCcols.append(M_find+jump+1)
            CCvals.append(-1/(4*h*h))

            CCrows.append(M_find+1)
            CCcols.append(M_find+jump)
            CCvals.append(-1/(4*h*h))
    else:
            counter3 = counter3 + (-1/(4*h*h))   #append central stencil point (Neumann)
    q = gridArrayIJ[vecIJ.y-1, vecIJ.x+1]       #i+1,j-1
    if(q >= 0):
            jump = 3 * (q - currentpoint)
            CCrows.append(M_find)
            CCcols.append(M_find+jump+1)
            CCvals.append(-1/(4*h*h))

            CCrows.append(M_find+1)
            CCcols.append(M_find+jump)
            CCvals.append(-1/(4*h*h))
    else:
            counter3 = counter3 + (-1/(4*h*h))   #append central stencil point (Neumann)
    q = gridArrayIJ[vecIJ.y-1, vecIJ.x-1]       #i-1,j-1
    if(q >= 0):
            jump = 3 * (q - currentpoint)
            CCrows.append(M_find)
            CCcols.append(M_find+jump+1)
            CCvals.append(1/(4*h*h))
            
            CCrows.append(M_find+1)
            CCcols.append(M_find+jump)
            CCvals.append(1/(4*h*h))
    else:
            counter3 = counter3 + (1/(4*h*h))   #append central stencil point (Neumann)

    CCrows.append(M_find)
    CCcols.append(M_find)
    CCvals.append(counter1)   #append central stencil point (Neumann)

    CCrows.append(M_find+2)
    CCcols.append(M_find)
    CCvals.append(counter2)   #append central stencil point (Neumann)

    CCrows.append(M_find+1)
    CCcols.append(M_find)
    CCvals.append(counter3)   #append central stencil point (Neumann)

    
 #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
CurlCurlPin = sparse.csc_matrix((CCvals,(CCrows,CCcols)), shape=(3*Np, 3*Np))   #form sparse matrix
 #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
del(CCrows)
del(CCcols)
del(CCvals)

#===========================================================================#
#===========================================================================#
#===========================================================================#

CCrows=[]   
CCcols=[]
CCvals=[]
counter1=0
counter2=0
counter3=0
counter4=0
counter5=0
counter6=0
print("")
print("Now setting up Curl Curl (Neumann) operator matrix.....")
for p in range(0, Np):
    M_find = 3*p    #So don't have to keep calculating 3*p
    right = 3      #For jumping to the 3x3 next to current
    left = -3

    counter1=0
    counter2=0
    counter3=0
    counter4=0
    counter5=0
    counter6=0
    
    vecIJ = gridArrayP[p]
    x = getVec2D(vecIJ.x, vecIJ.y, GridWidth, h).x
    y = getVec2D(vecIJ.x, vecIJ.y, GridWidth, h).y
    
    currentpoint = (gridArrayIJ[vecIJ.y, vecIJ.x])

    counter1 = counter1 + ((2/h/h)-(C**2))       #(M_find,M_find)   #X
    counter2 = counter2                          #(M_find+2,M_find) #X
    counter3 = counter3                          #(M_find+1,M_find) #X
    
    counter4 = counter4 + ((2/h/h)-(C**2))       #(M_find+1,M_find+1) #Y
    counter5 = counter5                          #(M_find+2,M_find+1) #Y
    counter6 = counter6                          #(M_find,M_find+1) #Y

    CCrows.append(M_find+2)    #i,j
    CCcols.append(M_find+2)
    CCvals.append((2/h/h)+(2/h/h))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    q = gridArrayIJ[vecIJ.y, vecIJ.x+1]    #i+1,j
    if(q >= 0):
            CCrows.append(M_find)
            CCcols.append(M_find+right+2)
            CCvals.append(C/(2*h))

            CCrows.append(M_find+1)
            CCcols.append(M_find+right+1)
            CCvals.append(-1/h/h)

            CCrows.append(M_find+2)
            CCcols.append(M_find+right)
            CCvals.append(C/(2*h))

            CCrows.append(M_find+2)
            CCcols.append(M_find+right+2)
            CCvals.append(-1/h/h)
    else:
            counter2 = counter2 + (C/(2*h))   #append central stencil point (Neumann)
            counter4 = counter4 + (-1/h/h)

    q = gridArrayIJ[vecIJ.y, vecIJ.x-1]     #i-1,j
    if(q >= 0):
            CCrows.append(M_find)
            CCcols.append(M_find+left+2)
            CCvals.append(-C/(2*h))

            CCrows.append(M_find+1)
            CCcols.append(M_find+left+1)
            CCvals.append(-1/h/h)

            CCrows.append(M_find+2)
            CCcols.append(M_find+left)
            CCvals.append(-C/(2*h))

            CCrows.append(M_find+2)
            CCcols.append(M_find+left+2)
            CCvals.append(-1/h/h)

    else:
            counter2 = counter2 + (-C/(2*h))   #append central stencil point (Neumann)
            counter4 = counter4 + (-1/h/h)
            
    q = gridArrayIJ[vecIJ.y+1, vecIJ.x]     #i,j+1
    if(q >= 0):
            jump = 3 * (q - currentpoint)

            CCrows.append(M_find)
            CCcols.append(M_find+jump)
            CCvals.append(-1/h/h)

            CCrows.append(M_find+1)
            CCcols.append(M_find+jump+2)
            CCvals.append(C/(2*h))

            CCrows.append(M_find+2)
            CCcols.append(M_find+jump+1)
            CCvals.append(C/(2*h))

            CCrows.append(M_find+2)
            CCcols.append(M_find+jump+2)
            CCvals.append(-1/h/h)
    else:
            counter1 = counter1 + (-1/h/h)   #append central stencil point (Neumann)
            counter5 = counter5 + (C/(2*h))

    q = gridArrayIJ[vecIJ.y-1, vecIJ.x]     #i,j-1
    if(q >= 0):
            jump = (3) * (q - currentpoint)

            CCrows.append(M_find)
            CCcols.append(M_find+jump)
            CCvals.append(-1/h/h)

            CCrows.append(M_find+1)
            CCcols.append(M_find+jump+2)
            CCvals.append(-C/(2*h))

            CCrows.append(M_find+2)
            CCcols.append(M_find+jump+1)
            CCvals.append(-C/(2*h))

            CCrows.append(M_find+2)
            CCcols.append(M_find+jump+2)
            CCvals.append(-1/h/h)
    else:
            counter1 = counter1 + (-1/h/h)   #append central stencil point (Neumann)
            counter5 = counter5 + (-C/(2*h))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    q = gridArrayIJ[vecIJ.y+1, vecIJ.x+1]     #i+1,j+1
    if(q >= 0):
            jump = 3 * (q - currentpoint)
            CCrows.append(M_find)
            CCcols.append(M_find+jump+1)
            CCvals.append(1/(4*h*h))

            CCrows.append(M_find+1)
            CCcols.append(M_find+jump)
            CCvals.append(1/(4*h*h))
    else:
            counter3 = counter3 + (1/(4*h*h))   #append central stencil point (Neumann)
            counter6 = counter6 + (1/(4*h*h)) 
            
    q = gridArrayIJ[vecIJ.y+1, vecIJ.x-1]     #i-1,j+1
    if(q >= 0):
            jump = 3 * (q - currentpoint)
            CCrows.append(M_find)
            CCcols.append(M_find+jump+1)
            CCvals.append(-1/(4*h*h))

            CCrows.append(M_find+1)
            CCcols.append(M_find+jump)
            CCvals.append(-1/(4*h*h))
    else:
            counter3 = counter3 + (-1/(4*h*h))   #append central stencil point (Neumann)
            counter6 = counter6 + (-1/(4*h*h))
            
    q = gridArrayIJ[vecIJ.y-1, vecIJ.x+1]     #i+1,j-1
    if(q >= 0):
            jump = 3 * (q - currentpoint)
            CCrows.append(M_find)
            CCcols.append(M_find+jump+1)
            CCvals.append(-1/(4*h*h))

            CCrows.append(M_find+1)
            CCcols.append(M_find+jump)
            CCvals.append(-1/(4*h*h))
    else:
            counter3 = counter3 + (-1/(4*h*h))   #append central stencil point (Neumann)
            counter6 = counter6 + (-1/(4*h*h))
            
    q = gridArrayIJ[vecIJ.y-1, vecIJ.x-1]     #i-1,j-1
    if(q >= 0):
            jump = 3 * (q - currentpoint)
            CCrows.append(M_find)
            CCcols.append(M_find+jump+1)
            CCvals.append(1/(4*h*h))
            
            CCrows.append(M_find+1)
            CCcols.append(M_find+jump)
            CCvals.append(1/(4*h*h))
    else:
            counter3 = counter3 + (1/(4*h*h))   #append central stencil point (Neumann)
            counter6 = counter6 + (1/(4*h*h))

    CCrows.append(M_find)
    CCcols.append(M_find)
    CCvals.append(counter1)   #append central stencil point (Neumann)

    CCrows.append(M_find+2)
    CCcols.append(M_find)
    CCvals.append(counter2)   #append central stencil point (Neumann)

    CCrows.append(M_find+1)
    CCcols.append(M_find)
    CCvals.append(counter3)   #append central stencil point (Neumann)

    CCrows.append(M_find+1)
    CCcols.append(M_find+1)
    CCvals.append(counter4)   #append central stencil point (Neumann)

    CCrows.append(M_find+2)
    CCcols.append(M_find+1)
    CCvals.append(counter5)   #append central stencil point (Neumann)

    CCrows.append(M_find)
    CCcols.append(M_find+1)
    CCvals.append(counter6)   #append central stencil point (Neumann)
    
 #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
CurlCurlPin2 = sparse.csc_matrix((CCvals,(CCrows,CCcols)), shape=(3*Np, 3*Np))
 #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
del(CCrows)
del(CCcols)
del(CCvals)
#===========================================================================#
#===========================================================================#
#===========================================================================#

print("")
print("Now forming force operators for each operator combination")

firstterm1 = Bcross*Curl*Curl*Bcross     #Dirichlet
firstterm2 = Bcross*CurlPin*CurlPin*Bcross    #Mixed Dirichlet,Neumann
firstterm3 = Bcross*CurlPin2*CurlPin2*Bcross   #Neumann
firstterm4 = Bcross*CurlCurl*Bcross   #Dirichlet
firstterm5 = Bcross*CurlCurlPin*Bcross   #Mixed Dirichlet,Neumann
firstterm6 = Bcross*CurlCurlPin2*Bcross   #Neumann

secondterm1 = Jcross*Curl*Bcross
secondterm2 = Jcross*CurlPin*Bcross
secondterm3 = Jcross*CurlPin2*Bcross

result1 = oneoverrho * firstterm1      #Multiple by 1/rho
result1 = -result1
result2 = oneoverrho * firstterm2
result2 = -result2
result3 = oneoverrho * firstterm3
result3 = -result3
result4 = oneoverrho * firstterm4
result4 = -result4
result5 = oneoverrho * firstterm5
result5 = -result5
result6 = oneoverrho * firstterm6
result6 = -result6

bothterms1 = oneoverrho * (firstterm1 + secondterm1)     #Include second term for inhomogeneous case with current
bothterms1 = -bothterms1
bothterms2 = oneoverrho * (firstterm2 + secondterm2)
bothterms2 =  -bothterms2
bothterms3 = oneoverrho * (firstterm3 + secondterm3) 
bothterms3 =  -bothterms3
bothterms4 = oneoverrho * (firstterm4 + secondterm1) 
bothterms4 =  -bothterms4
bothterms5 = oneoverrho * (firstterm5 + secondterm2) 
bothterms5 =  -bothterms5
bothterms6 = oneoverrho * (firstterm6 + secondterm3)  
bothterms6 =  -bothterms6

#===========================================================================#
#===========================================================================#
#===========================================================================#

 #TESTING WITH (1,0,0) VEC AND (1,0,0) SPARSE MATRIX FOR EIGENSOLVER
onezerozero = []
for p in range(0,Np):
        onezerozero.append(1)
        onezerozero.append(0)
        onezerozero.append(0)

rrows = []
rcols = []
rvals = []
for p in range(0,Np):
        M_find = 3*p
        rrows.append(M_find)
        rcols.append(M_find)
        rvals.append(1)      
rtest = sparse.csc_matrix((rvals,(rrows,rcols)), shape=(3*Np, 3*Np))


#print(result4*onezerozero)

print("")
#===========================================================================#
#===========================================================================#
#===========================================================================#
#===========================================================================#
#===========================================================================#
#===========================================================================#
#===========================================================================#

#Making a circle of radius 0.4
x_circle = -0.39999
y_circle = 0
increment = 0.0001
N = 8000
circle = np.zeros([2,N])
ylol = []
xlol = []
for i in range(0,N):
       y_circle = ((0.16-(x_circle**2))**0.5)
       #print(y_circle)
       x_circle = x_circle + increment
       circle[0,i] = y_circle
       circle[1,i] = x_circle
       ylol.append(y_circle)
       xlol.append(x_circle)

x_circle = 0.39999
y_circle = 0
increment = -0.0001
for i in range(0,N):
       y_circle = ((0.16-(x_circle**2))**0.5)
       #print(y_circle)
       x_circle = x_circle + increment
       circle[0,i] = y_circle
       circle[1,i] = x_circle
       ylol.append(-y_circle)
       xlol.append(x_circle)

#===========================================================================#
#===========================================================================#
#===========================================================================#
#===========================================================================#
#===========================================================================#


print("")
print("Now calculating eigensystem....")
num_of_eigvals = 10         #NUMBER OF EIGVALS TO FIND
target = 100           #GIVE ARNOLDI ALGORITHM A TARGET EIGENVALUE
vals, vecs = linalg.eigsh(result6, k=num_of_eigvals, sigma=target)        #COMPUTER ARNOLDI ALGORITHM
print("")
print(vals)

####------------PLOTTING--------------#########
xlist=[]
for i in range(0, Nx):
	xlist.append(getVec2D(i, 0, radius, h).x)

ylist=[]
for j in range(0, Ny):
	ylist.append(getVec2D(0, j, radius, h).y)

plotX, plotY = np.meshgrid(xlist, ylist)
     
for mySolution in range(0,num_of_eigvals):
	resultArrayIJ = np.zeros([Nx, Ny])
	for p in range(0, Np-1):
		ijVec = gridArrayP[p]
		resultArrayIJ[ijVec.x,ijVec.y] = vecs[3*p,mySolution].real
	fig=plt.figure()
	#plt.title('Laplacian solution')
	plt.title('Eigenfunction associated with first converged eigenvalue')
	plt.xlabel('x')
	plt.ylabel('y')
	cp = plt.contourf(plotX, plotY, resultArrayIJ,10)
	plt.plot(xlol,ylol)
	plt.colorbar(cp)
	matplotlib.pyplot.show()
	plt.close(fig)


