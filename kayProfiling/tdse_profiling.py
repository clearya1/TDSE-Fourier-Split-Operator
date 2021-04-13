import matplotlib.pylab as plt
import numpy as np
from scipy.constants import pi
from mpi4py import MPI
from mpi4py_fft import PFFT, newDistArray, DistArray
from sys import stdout, argv
#from matplotlib.animation import FuncAnimation
#import time
#import h5py
import cProfile, pstats, io
from pstats import SortKey

import tiseAnalytic as an
import energy

""" Not matrix multiplication here but element wise multiplication for all matmuls. ~25% faster than tdse because multiplying across all cores instead of multiplying all on one core """
    

#----------setting up basic parameters---------------

"""Physical Parameters"""
m = 1           #mass of particle
hbar=1
t0 = 0.0        #initial time
w = 1           #frequency of potential = 1/2*m*w^2*x^2

"""Numeric Parameters"""
N = int(argv[1])        #resolution of grid
xMax = int(argv[2])
yMax = xMax
h = 0.01        #time resolution
tol = 1e-5

#----------------Normalising Functions---------------

def fullNorm(x):                #returns the full norm from list of norms from each core
    return np.linalg.norm(x)

def normalise(u):
    part_norm = np.linalg.norm(u)          #get the norm of each section of wavefuntion
    norms = comm.gather(part_norm, root=0)
    if rank==0:
        norm = fullNorm(norms)             #combine all the norms into total norm
    else:
        norm = None
    norm = comm.bcast(norm,root=0)         #send out the total norm to all cores
    return u/norm

#-------setting up the communicator for MPI----------

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def profile(filename=None, comm=comm):
    def prof_decorator(f):
        def wrap_f(*args, **kwargs):
            pr = cProfile.Profile()
            pr.enable()
            result = f(*args, **kwargs)
            pr.disable()
            s = io.StringIO()
            sortby = SortKey.CUMULATIVE
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)

            if filename is None:
                #ps.print_stats()
                print(s.getvalue())
            else:
                filename_r = filename + "{}".format(rank)
                ps.dump_stats(filename_r)

            return result
        return wrap_f
    return prof_decorator

#-----setting up the Fast Fourier Transform----------

N2D = np.array([N, N], dtype=int)
fft = PFFT(comm, N2D, axes=(1,0), dtype=np.complex, grid=(1,-1,))  #defaults to rfftn for real input array.

#-----------setting up initial conditions------------

u = newDistArray(fft, False)                 #creating the distributed array for the input of fft
k = newDistArray(fft, True)                  #creating the distributed array for the output of fft

UrI = newDistArray(fft, True)
UrR = newDistArray(fft, True).astype(np.float)
UkI = newDistArray(fft, True)
UkR = newDistArray(fft, True).astype(np.float)

if rank==0:                                  #necessary buffers
    groundState = np.ones(N2D).astype(u.dtype)*-1
    buff = np.ones(N2D).astype(u.dtype)*-1
    V = np.ones(N2D).astype(u.dtype)
    P = np.ones(N2D).astype(u.dtype)
else:
    groundState = None
    buff = None
    
#-------Functions to generate the propagators--------

pxMax = (pi/xMax)*N/2
pyMax = (pi/yMax)*N/2
dx = 2*xMax/N
dy = 2*yMax/N
dpx = pi/xMax
dpy = pi/yMax

x = np.zeros(N)
y = np.zeros(N)
xp = np.zeros(N)
yp = np.zeros(N)

for i in range(int(N/2)):
    x[i] = -xMax + (i+1)*dx
    x[i + int(N/2)] = (i+1)*dx
    
    y[i] = -yMax + (i+1)*dy
    y[i + int(N/2)] = (i+1)*dy
    
    xp[i] = (i+1)*dpx
    xp[i + int(N/2)] = -pxMax + (i+1)*dpx
    
    yp[i] = (i+1)*dpy
    yp[i + int(N/2)] = -pyMax + (i+1)*dpy
    
def radiusSq(x,y):         #simple radius function
    return x**2+y**2

#--------Ur

def harmonicPotential(x):          #creates V(r) matrix for harmonic potential
    return 0.5*m*w**2*x

@profile(filename="profiling/spaceProp")
def spaceProp(m,w):                #generates propagators and V matrix - GLOBAL

    global UrR, UrI, V
    
    if rank==0:
        X,Y = np.meshgrid(x,y)
        R = radiusSq(X,Y)

        V = harmonicPotential(R)

        UrRtemp = np.exp(-V*h/2.0/hbar)    #NxN matrix = exp(-V(r)*h/2*hbar). imaginary time evolution. t -> -it
        UrItemp = np.exp(-1j*V*h/2.0/hbar).astype(u.dtype)
    else:
        UrRtemp = None
        UrItemp = None

    comm.Scatter(UrRtemp,UrR,root=0)
    comm.Scatter(UrItemp,UrI,root=0)

#---------Uk

@profile(filename="profiling/momProp")
def momProp(m,w):                   #generates propagators and V matrix - GLOBAL

    global UkR, UkI, P

    if rank==0:
        KX, KY = np.meshgrid(xp,yp)
        K = radiusSq(KX,KY)
        P = (hbar)**2*K/2.0/m

        UkRtemp = np.exp(-P*h/hbar)   #NxN matrix = exp(P(k)*h/hbar). P(k) = (hbar*k)**2/2m. k = 2pi/x. t -> -it.
        UkItemp = np.exp(-1j*P*h/hbar).astype(u.dtype)
    else:
        UkRtemp = None
        UkItemp = None

    comm.Scatter(UkRtemp,UkR,root=0)
    comm.Scatter(UkItemp,UkI,root=0)

#---implementing the Fourier split-operator method---

def fso_step(Ur,Uk):

    global groundState, buff, u, k

    #Apply first Ur here
    
    u = u.redistribute(1)
    u = np.multiply(u,Ur)
    u = u.redistribute(0)
    
    #Forward FFT here
    
    k = fft.forward(u, k)
    
    #Apply Uk here

    k = np.multiply(k,Uk)
    
    #Backward FFT here
    
    u = fft.backward(k, u)
    
    #Second Ur here
    
    u = u.redistribute(1)
    u = np.multiply(u,Ur)
    u = u.redistribute(0)
    
    #Normalise here
    
    u = normalise(u)
    
#------------find Ground State Function--------------

@profile(filename="profiling/findGround")
def findGround(Ur,Uk):

    global u0, done, allDone
    
    fso_step(Ur,Uk)
    if np.allclose(u,u0,tol):
        done=1
    allDone = comm.allgather(done)             #so that every core knows the 'done' status of every other core

    t=t0

    while np.sum(allDone)<size:                #main body of code
        if np.allclose(u,u0,tol):
            done=1
        allDone = comm.allgather(done)
        
        u0 = u
        fso_step(Ur,Uk)
        t+=h
        
        #if rank==0:
        #    print(t)
        
        stdout.flush()
        comm.barrier()

"""-------------------------------------------------------------------"""
#-----------------------Running Main Code Here---------------------------
"""-------------------------------------------------------------------"""

start = time.time()

allDone = np.zeros(size)                    #for synchronising the while loops on all cores
done = 0                                    #done = 1 on each core when local while loop ends

"""set the initial wavefunction here"""

u[:] = np.random.rand(*u.shape).astype(u.dtype) + np.random.rand(*u.shape).astype(u.dtype)*1j

u = normalise(u)
u0 = u                                       #copying the initial state

spaceProp(m,w)
momProp(m,w)

findGround(UrR, UkR)

end = time.time()
print("Rank: ", rank, ". Time Taken: ", end-start)


if rank==0:
    print("Ground State Reached")
    
  
#----------------Analytic Solution-------------------
"""
if rank==0:
    analyticGroundState = an.main(m,hbar,w,N,xMax,yMax)
"""
#----------------Collecting Solutions----------------

u = u.redistribute(1)
comm.Gather(u,groundState,root=0)
u = u.redistribute(0)

k = fft.forward(u, k)
k = normalise(k)
comm.Gather(k,buff,root=0)

if rank==0:

    print("final norm = ", np.linalg.norm(groundState))
    print("momentum norm = ", np.linalg.norm(buff))
    
    spaceE = energy.energy(groundState,V)
    momE = energy.energy(buff,P)
    totalE = spaceE + momE
    
    print("spatial contribution: ", spaceE)
    print("momentum contribution: ", momE)
    print("Total Energy: ", totalE)
    
    
#------------------Plotting Solutions----------------
"""
if rank==0:

    angles = np.angle(groundState)
    angles = (angles%(2*pi)+2*pi)%(2*pi)
    density = np.square(np.abs(groundState))
    Kdensity = np.square(np.abs(buff))
    
    ticks = np.linspace(0,N,10,dtype=int)
    labels = np.round(np.linspace(-xMax,xMax,10),1)
    Klabels = np.round(np.linspace(-pxMax,pxMax,10),0)

#-------------Numeric Ground State

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.xticks(ticks,labels)
    plt.yticks(ticks,labels[::-1])
    ax.set_title('Phase of Ground State')
    
    plt.imshow(angles, interpolation='none')
    plt.colorbar()
    plt.show()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.xticks(ticks,labels)
    plt.yticks(ticks,labels[::-1])
    ax.set_title('Density of Ground State')
    
    plt.imshow(density, interpolation='none')
    plt.colorbar()
    plt.show()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.set_xlabel('k_x')
    ax.set_ylabel('k_y')
    plt.xticks(ticks,Klabels)
    plt.yticks(ticks,Klabels[::-1])
    ax.set_title('Density of Momentum')
    
    plt.imshow(np.fft.fftshift(Kdensity), interpolation='none')
    plt.colorbar()
    plt.show()
    
#----------Analytic Ground State Density
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.xticks(ticks,labels)
    plt.yticks(ticks,labels[::-1])
    ax.set_title('Analytic Density')
    
    plt.imshow(analyticGroundState, interpolation='none')
    plt.colorbar()
    plt.show()
    
#---------Comparison of Analytic and Numeric
    
    compare = np.divide(density,analyticGroundState)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.xticks(ticks,labels)
    plt.yticks(ticks,labels[::-1])
    ax.set_title('Numeric Density/Analytic Density')
    
    plt.imshow(compare, interpolation='none', vmin=0.5, vmax=1.5)
    plt.colorbar()
    con = ax.contour(compare, [0.9,1.1], colors='k', origin='upper')
    ax.clabel(con, inline=1, fontsize=10)
    plt.show()
    

#-------Numeric Slices in X and Y directions
    
    slice = [density[i,int(N/2)] for i in range(N)]
    anSlice = [analyticGroundState[i,int(N/2)] for i in range(N)]
    plt.plot(slice, label="Numeric")
    plt.plot(anSlice, label="Analytic")
    plt.title('Slice at x=0')
    plt.xticks(ticks,labels)
    plt.xlabel('y')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
    
    slice = density[int(N/2)]
    anSlice = analyticGroundState[int(N/2)]
    plt.plot(slice, label="Numeric")
    plt.plot(anSlice, label="Analytic")
    plt.title('Slice at y=0')
    plt.xticks(ticks,labels)
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
"""
#------Animating Solution after Ground State reached-----
"""
fps = int(1/h)
nSeconds = 5
snapshots = []
snapshotsPhase = []
snapshotsMom = []
snapshotsXSlice = []
snapshotsYSlice = []
t = 0
w = 2

spaceProp(m,w)
momProp(m,w)

if rank==0:
    energyList = np.zeros(nSeconds*fps)

for i in range(nSeconds*fps):

    u = u.redistribute(1)
    comm.Gather(u,groundState,root=0)
    u = u.redistribute(0)
    
    k = fft.forward(u, k)
    k = normalise(k)
    comm.Gather(k,buff,root=0)
    
    if rank==0:
    
        totalE = energy.energy(groundState,V) + energy.energy(buff,P)
        energyList[i] = totalE
        
        density = np.square(np.abs(groundState))
        snapshots.append(density)
        
        angles = np.angle(groundState)
        angles = (angles%(2*pi)+2*pi)%(2*pi)
        snapshotsPhase.append(angles)
        
        Kdensity = np.square(np.abs(buff))
        snapshotsMom.append(np.fft.fftshift(Kdensity))
        
        snapshotsXSlice.append([density[i,int(N/2)] for i in range(N)])
        snapshotsYSlice.append(density[int(N/2)])
        
    fso_step(UrI,UkI)
    

if rank==0:

#-------------plot how energy changes with time

    plt.plot(np.arange(nSeconds*fps)*h,energyList)
    plt.xlabel('time [s]')
    plt.ylabel('Energy')
    plt.title('Omega = %i' %w)
    plt.show()
    
#-------------2D animation of density

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ticks = np.linspace(0,N,12,dtype=int)
    labels = np.rint(np.linspace(-xMax,xMax,12))
    Klabels = np.round(np.linspace(-pxMax,pxMax,12),0)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.xticks(ticks,labels)
    plt.yticks(ticks,labels[::-1])
    ax.set_title('Density Evolution. Omega: 1->2')

    def animate_func(i,snapshots,twoD=True):
        if i % fps == 0:
            print( '.', end ='' )
        if twoD==True:
            im.set_array(snapshots[i])
        else:
            im.set_data(x,snapshots[i])
        t = int(i)*0.01
        time_text.set_text('time = %.1f' % t)
        return [im], time_text

    a = snapshots[0]
    im = plt.imshow(a, interpolation='none',vmax=np.max(snapshots)*1.1)
    plt.colorbar()
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,color='white')

    anim = FuncAnimation(
                        fig,
                        animate_func,
                        frames = nSeconds * fps,
                        fargs=(snapshots,),
                        interval = 1000 / fps, # in ms
                        )
    
    plt.show()
    anim.save('2D_anim.mp4', writer='ffmpeg', fps=fps)
    
#-------------2D animation of phase

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.xticks(ticks,labels)
    plt.yticks(ticks,labels[::-1])
    ax.set_title('Phase Evolution. Omega: 1->2')

    a = snapshotsPhase[0]
    im = plt.imshow(a, interpolation='none')
    plt.colorbar()
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,color='white')

    anim = FuncAnimation(
                        fig,
                        animate_func,
                        frames = nSeconds * fps,
                        fargs=(snapshotsPhase,),
                        interval = 1000 / fps, # in ms
                        )

    plt.show()
    anim.save('2D_phase_anim.mp4', writer='ffmpeg', fps=fps)

#-------------2D animation of momentum density

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.xticks(ticks,Klabels)
    plt.yticks(ticks,Klabels[::-1])
    ax.set_title('Momentum Density Evolution. Omega: 1->2')

    a = snapshotsMom[0]
    im = plt.imshow(a, interpolation='none',vmax=np.max(snapshotsMom)*1.1)
    plt.colorbar()
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,color='white')

    anim = FuncAnimation(
                        fig,
                        animate_func,
                        frames = nSeconds * fps,
                        fargs=(snapshotsMom,),
                        interval = 1000 / fps, # in ms
                        )

    plt.show()
    anim.save('2D_mom_anim.mp4', writer='ffmpeg', fps=fps)

#-------------X Slice of 2D animation

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Slice at x = 0')
    ax.set_xlabel('y')
    ax.set_ylabel('Density')
    plt.ylim(-0.01,np.max(snapshotsXSlice)*1.1)

    a = snapshotsXSlice[0]
    im, = ax.plot(x,a)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,color='black')

    anim = FuncAnimation(
                        fig,
                        animate_func,
                        frames = nSeconds * fps,
                        fargs=(snapshotsXSlice,False,),
                        interval = 1000 / fps, # in ms
                        )
    
    plt.show()
    anim.save('XSlice_anim.mp4', writer='ffmpeg', fps=fps)
    
#-------------Y Slice of 2D animation

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Slice at y = 0')
    ax.set_xlabel('y')
    ax.set_ylabel('Density')
    plt.ylim(-0.01,np.max(snapshotsYSlice)*1.1)

    a = snapshotsYSlice[0]
    im, = ax.plot(x,a)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,color='black')

    anim = FuncAnimation(
                        fig,
                        animate_func,
                        frames = nSeconds * fps,
                        fargs=(snapshotsYSlice,False,),
                        interval = 1000 / fps, # in ms
                        )
    
    plt.show()
    anim.save('YSlice_anim.mp4', writer='ffmpeg', fps=fps)
"""
