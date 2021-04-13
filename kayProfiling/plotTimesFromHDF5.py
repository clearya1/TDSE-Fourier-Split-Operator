import matplotlib.pylab as plt
import numpy as np

weaktimes = np.array([1.5621243000030518, 2.4564270257949827, 5.667662310600281, 15.551984357833863, 42.10252578258515, 140.8223353624344])

weakt1times = np.array([4.833590841293335, 22.160210275650023, 89.83878316879273, 381.3752159118652, 1526.2943887710571, 7829.265781974793])

strongtimes = np.array([392.28783016204835, 259.0197401285171, 122.43190951347351, 68.95477414131165, 37.94266412258148, 23.55146493911743, 16.490307021141053, 16.065140700340272, 13.837846493721008, 17.27278277873993])
strongtimes = strongtimes[0]/strongtimes

weakprocs = [16,32,64,128,256,512]
strongprocs = [1,2,4,8,16,32,64,128,256,512]

#should we consider gather/scatter/redistribute as parallel or serial operations? - All Serial, in fact they take longer as we increase the number of cores

"""idealS = 0.004302 + 0.00529 + 0.0327
idealP = 0.000328 + 0.00112 + 0.4773 + 5.7"""

#Serial parts: propagators, gather, scatter, redistribute, FullNorm, sum, barrier,
idealS = 0.127 + 0.0367 + 1.62 + 0.018 + 0.0433 + 0.0205 + 0.0125
idealP = 0.000 + 0.0000 + 1.3357

idealTotal = idealS + idealP

print("s = ", idealS/idealTotal)
print("p = ", idealP/idealTotal)

Nstronganalytic = np.linspace(strongprocs[0], strongprocs[-1], 100)
Nweakanalytic = np.linspace(weakprocs[0], weakprocs[-1], 100)

def strongAnalytic(N):
    return 1/(idealS + idealP/N)
    
def weakAnalytic(N):
    return idealS + idealP*N

plt.scatter(weakprocs, weakt1times/weaktimes)
plt.plot(Nweakanalytic, weakAnalytic(Nweakanalytic))
#plt.scatter(strongprocs[:-2], strongtimes[:-2])  #kinda got weird at really high nProcs
#plt.plot(Nstronganalytic, strongAnalytic(Nstronganalytic))
plt.xlabel('No. of Processors')
plt.ylabel('Speed-Up')
plt.title('Weak Scaling')
#plt.title('Strong Scaling')
plt.show()


