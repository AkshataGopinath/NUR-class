import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from qn1 import random, poisson

##################################################################################
#### part a ####

# Integration using extended mid point rule
def intg_extMP(func, limL, limU, n):
    """ 
    Function to integrate using extended midpoint rule
    Inputs-
    func : function to be integrated
    limL, limU : Limits of integration (limL- lower limit, limU - upper limit)
    n : number of steps/divisions in the interval
    """ 
    h = (limU- limL)/n  # size of each interval
    # initializing integral value
    I = 0
    x_mid = limL + (h/2)
    # applying the extended mid-point rule in the interval:
    for i in range(0, n):
        I += func(x_mid)
        x_mid += h
    return I*h  

# x = r/r_vir
N_avg = 100

# randomly generating a,b,c from a unifrom distribution between the given values; U(a,b)= U(0,1)*(b-a) + a
a = random(314152, 1.1, 2.5, 1)
b = random(1729, 0.5, 2.0, 1)
c = random(666, 1.5, 4, 1)

x_min=1e-4
x_max=5
# the number density profile to be integrated to find the value of normalizing constant A
def integrand(x, a=a,b=b,c=c):
    n_A = N_avg*(x/b)**(a-3)*np.exp(-(x/b)**c)*4*np.pi*(x**2)   
    return n_A

# integrating in limits (0, 5) and finding value of A
A = intg_extMP(integrand, 0, x_max, 100)/ (N_avg)

def q2a():
	print('a={} , b={}, c={}, A={}'.format(a,b,c,A))



###################################################################################
#### part b ####

#linear interpolation
def linear_interp( known_fn, known, unknown):
    """
    Linear interpolation function.
    Inputs:
    known_fn : function values thet are known
    known : x-coordinates of the points where the function values are known
    unknown : points for which interpolation is to be used
    """
    x = known
    x1 = unknown 
    fx = known_fn(10**x)
    fx1 = np.empty(len(x1))
    for j in range(len(x)-1):
        for i in range(len(x1)):
            if x1[i]>=x[j] and x1[i]<=x[j+1]:
                fx1[i] = ((fx[j+1] - fx[j])/(x[j+1]-x[j]))*(x1[i] - x[j]) + fx[j]

            elif x1[i] > max(x):
                fx1[i] = ((fx[len(x)-1]-fx[len(x)-2])/(x[len(x)-1]-x[len(x)-2]))*(x1[i] - x[len(x)-1]) + fx[len(x)-1]            
    return fx1

# defining n(x), density profile, given by equation 2 
def density_profile(x):
    return(A*N_avg*(x/b)**(a-3)*np.exp(-(x/b)**c) )


#calculating values of n at given x values:
x_values = [1e-4, 1e-2, 1, 5]
n_values = density_profile(x_values)


# to obtain log of the density profile, in order to interpolate in log space
def log_nx(x):
    return np.log10(density_profile(x))

def q2b():
	print("""
	Interpolating in log space using linear interpolation. 
	Given the 4 points, cubic spline interpolation would fit a cubic function through them which wouldnt be a good fit
	for the given density profile. The first 3 points lie almost on the same line by eye (in log space).
	Akima sub spline method would probably be the best option, but given the points from the previous plot,
	linear interpolation will work almost as well. 
	(Verified by checking how it looks using the Akima-sub-spline library routine)
	""")
	nx_interp = linear_interp(log_nx, np.log10(x_values), np.log10(np.logspace(np.log10(x_min),np.log10(x_max), 50)) )
	plt.figure()
	plt.xscale('log')
	plt.yscale('log')
	# log log plot of interpolated values
	plt.scatter(10**(np.log10(np.logspace(np.log10(x_min),np.log10(x_max), 50))), 10**(nx_interp), s=20, color='red', label='Interpolated values')
	plt.title('log-log scatter plot of interpolated values of n(x)')
	plt.xlabel('log(n(x))')
	plt.ylabel('log(x)')
	# log log plot for the given 4 single points
	plt.scatter(((x_values)), ((n_values)), s=40, edgecolors='black', label='Known points')
	plt.legend(loc='lower left')
	plt.savefig('logloginterp.png')



###################################################################################
### part c ####

#  2 (c) #

## Numerical differenciation
def diff_rich(func, x, m):
    """
    Numerical differenciation using Richardson extrapolation
    func: function to be differentiated
    x : where the function needs to be differentiated at
    """
    d = 2
    h = 0.1
    D = np.empty((m,m))
    for i in range(0, m):
        #central differences
        D[i, 0] = (func(x+h) - func(x-h))/(2*h) 
        # Richardson extrapolation formula
        for j in range(0,i):
            D[i,j+1] =((d**(2*(j+1)))*D[i,j]-D[i-1,j])/((d**(2*(j+1)))-1)
        # updte value of h
        h = h/d
    return D[m-1, m-1]
    
# numerically differentiating n(x) by Richardson extrapolation, stopping at m=3 as it diverges soon after that
dndx = diff_rich(density_profile ,b, 3)

def q2c():
	print('value of dn(x)/dx at x=b, by numerical differentiation is', dndx)
	print('value of dn(x)/dx at x=b, found analytically is', -498.672)




###################################################################################
#### part d ####

# 100 samples of satellites
# rejection sampling
def sampling_rej(func, x_min, x_max, trials, n, seed_x , seed_y ): 
    """ 
    Rejection sampling
    Inputs -
    func : function to be sampled
    x_max : upper limit for x-axis of function
    trails : number of random (x,y) coordinates to try
    n : number of samples required
    seed_x, seed_y : seeds for RNG for x and y 
    """
    samples = np.zeros(n)
    j=0
    # break loop when you have required number of samples
    while(j< n-1):
        # random x uniformly distributed in (0, x_max)
        x = random(seed_x, 0, x_max,1 )
        # ramdom y uniformly distributed in (0,gauss(x))
        #finding y within a gaussian envelope to the actual distribution, to make rejection sampling faster
        y = random(seed_y, 0, gauss(x), 1)
        seed_x = np.round(x*100000).astype(int)
        seed_y = np.round(y*100000).astype(int)
        # accept y values that lie below the curve of the function
        if y < func(x):
            samples[j]= x
            j +=1
       
    return np.array(samples)            

# gaussian envelope
def gauss(x):
    amplitude = 2400
    x0 = 0.83
    sigma = 0.5
    gauss = (1/(np.sqrt(2*np.pi)*sigma**2))*np.exp(-( (x-x0)**2 / ( 2 * sigma**2 ) ) )
    return amplitude*gauss/np.amax(gauss)

# generating 3d satellite positions
# generating the distributuion of radius
def Nx(x):
    return (density_profile(x)*4*np.pi*x**2)/ N_avg

# function to generate the positions of the satellites (r, theta, phi)
def sats3d(seed_x, seed_y, seed_thet, seed_ph):
    Nsat =100
    # sampling the above generated distribution using rejection sampling, to obtain radii
    r_x = sampling_rej(Nx, 0, x_max, 10000000, Nsat, seed_x, seed_y)

    # generating theta and phi (in radians) on a sphere, using a uniform random distribution
    thet = np.pi*random(seed_thet,0,1, Nsat)
    ph = 2*np.pi*random(seed_ph, 0, 1, Nsat)
    return r_x, thet, ph  

r_x, thet, ph = sats3d(519, 18, 228, 1024)

sat_pos = np.empty((100, 3))
sat_pos[:, 0] = r_x
sat_pos[:, 1] = thet
sat_pos[:, 2] = ph

def q2d():
	print('positions of 100 satellites in (r, theta(radians), phi(radians)) are: \n', sat_pos)



###################################################################################
#### part e ####

# 1000 halos with 100 satellites each
halos = 1000
Nsat =100
r_Nx = np.empty((halos, Nsat))
theta = np.empty((halos, Nsat))
phi = np.empty((halos, Nsat))

def Nx1(x):
    return (density_profile(x)*4*np.pi*x**2)

# generating a thousand random seeds to find the radial samples
seed_x = np.round(random(734, 0, 154552, 1000)).astype(int)
seed_y = np.round(random(12, 0, 200000, 1000)).astype(int)

def halosats(seed_x, seed_y):
    Nsat =100
    # sampling the above generated distribution using rejection sampling
    r_x = sampling_rej(Nx1, 0, x_max, 10000000, Nsat, seed_x, seed_y)
    return r_x

for i in range(0, halos):
    # sampling the above generated distribution using rejection sampling
    r_Nx[i] = halosats(seed_x[i], seed_y[i]) 

x_range = np.linspace(1e-4, 5, 1000)
bins = np.logspace(np.log10(1e-4), np.log10(5), 20)

def q2e():
	print('The histogram seems to match the profile in shape, however falls short in amplitude. This could be due to more number of satellites in the bin than expected(as the initial bins are not filled). There are also no galaxies in the (1e-4, 9e-3) range,because we are sampling in a linear range of uniformly distributed values, while the plotting is in log scale.')
	plt.figure()
	plt.hist(np.ravel(r_Nx), bins = bins, density=True, edgecolor='black', linewidth=1.5, label='Avg #of satellites in each bin')
	plt.plot(x_range, Nx(x_range), linewidth=2, linestyle ='--', label='Profile of distribution N(x)')
	plt.xscale("log")
	plt.yscale("log")
	plt.title('Satellite galaxies in 1000 halos')
	plt.ylabel('log(N(x))')
	plt.xlabel('log(x)')
	plt.legend(loc='upper left')
	plt.savefig('haloshist.png')


###################################################################################
#### part f ####

# Root finding using bisection rule
def bisection_root(a, b, func, tol):
    """
    Root finding using bisection rule
    (a, b) : interval in which we expect to find the root
    func : function for which the root is to be found
    tol : tolerance for the root, below which the algorithm can stop iterating 
    """
    # initial interval
    x_, x = a,b
    # initial guess for root
    mid = (x_+x)/2
    i=0
    # root is found when the func(root)=0, search as long as the value of func(mid)> tolerance value
    while np.abs(func(mid)) > tol:
        # update guess for root value
        mid = (x_+x)/2
        if func(x_)*func(mid) < 0:
            # update lower limit of interval
            x = mid
        elif func(mid)*func(x) < 0:
            # update upper limit of interval
            x_ = mid
        i+=1  
    # return final value of root within tolerance
    return mid 

# root finding: solutions to N(x) = y/2
y = np.max(Nx1(x_range))
#defining the function whose solutions have to be found
def func_root(x):
    return Nx1(x) - (y/2)

def q2f():
	print("""since we know how the fuction well, using bisection method to find the 2 roots in the intervals between 
	   x_min and position of the peak of the function, and between the position of the peak of the function and x_max
	""")
	root1 = bisection_root( x_min, x_range[np.where(Nx1(x_range)==y)], func_root, 1e-5)
	root2 = bisection_root( x_range[np.where(Nx1(x_range)==y)], x_max, func_root, 1e-5)
	print('Solutions to: N(x) = y/2, are', root1,'and',root2)



###################################################################################
#### part g ####

def select_sort(mylist):
    """
    Selection sorting.
    Input: list to be sorted
    Works by continually ensuring you place the least element in the beginning of the array/sub-array
    """
    n = len(mylist)
    for i in range (0, n-1):
        i_min = i
        for j in range(i+1, n):
            if mylist[j] < mylist[i_min]:
                i_min = j
        if i_min != i:
            mylist[i], mylist[i_min] = mylist[i_min], mylist[i]
    return(mylist)

# getting the index of the bin to which each element in r_Nx belongs
indices = (np.digitize(np.ravel(r_Nx), bins))

# to count number of satellite galaxies in each bin
count = np.zeros(10000)
for i in indices:
    count[i] += 1
# index of bin with maximum number of satellite galaxies
maxB = np.where(count==np.max(count))[0][0]
# number of galaxies in this bin
max_sats = np.max(count)

# counting and recording the galaxies in each halo, in the bin with the maximum number
countBin = np.zeros(halos)
r_maxBin=[]
for i in range(0, halos):
    for j in range(0, N_avg):
        if r_Nx[i,j] >= bins[maxB - 1] and r_Nx[i,j] < bins[maxB]:
            countBin[i] += 1
            r_maxBin.append(r_Nx[i,j])

# sorting the bin with selection sort
sortR = select_sort(r_maxBin.copy())

# finding the median of the sorted list
if len(sortR)%2 == 0:
    medR = (sortR[int(len(sortR)/2)] + sortR[int((len(sortR)/2)-1)])/2
else:
    medR = sortR[int((len(sortR)-1)/2)]
    
# 16th percentile and 84th percentile
per16R = sortR[round(len(sortR)*0.16)]
per84R = sortR[round(len(sortR)*0.84)]

def q2g():
	print('median of radial bin containing largest number of galaxies:', medR)
	print('16th percentile of radial bin containing largest number of galaxies:', per16R)
	print('84th percentile of radial bin containing largest number of galaxies:', per84R)

	# plotting the normalized histogram of the number of galaxies in this radial bin
	plt.figure()
	k_len = int(max(countBin)-min(countBin)) + 1
	kval = np.linspace(min(countBin), max(countBin),k_len)

	plt.figure()
	plt.hist(countBin, edgecolor='black', linewidth=1.5, bins=kval, label='Histogram of number of galaxies')#, density= True)

	# over-plotting Poisson distribution from 1(a)
	pois = np.empty(k_len)
	lamb = np.mean(countBin)
	for ind in range(0, k_len):
		pois[ind]=(poisson(lamb, kval[ind])*1000)
	plt.plot(kval,pois, linewidth=3, linestyle ='--', label='Poisson distribution with mean={}'.format(lamb))
	plt.title('Satellite galaxies in largest radial bin({}), for 1000 halos'.format(maxB))
	plt.xlabel('Number of galaxies in halos')
	plt.ylabel('Number of halos (Normalized)')
	plt.legend(loc='upper right')
	plt.savefig('binhistpois.png')

###################################################################################
#### part h ####

a1 = np.arange(1.1, 2.6, 0.1)
b1 = np.arange(0.5, 2.1, 0.1)
c1 = np.arange(1.5, 4.1, 0.1)

# integrating in limits (1e-4, 5) and finding value of A
A_ = np.zeros((len(a1), len(b1), len(c1)))
for i1 in range(len(a1)):
    for j1 in range(len(b1)):
        for k1 in range(len(c1)):
            def integrand1(x, a1 = a1[i1],b1 = b1[j1], c1 =c1[k1]):
                n_A = N_avg*(x/b1)**(a1-3)*np.exp(-(x/b1)**c1)*4*np.pi*x**2   
                return n_A
            A_[i1, j1, k1] = (intg_extMP(integrand1, x_min, x_max, 10)/ (N_avg) )

# 3D interpolation using trilinear interpolation for a point (a,b,c)
def A_interp3d(a, b, c): # a,b,c are the co-ordinates of the points you want to interpolate at 
	# to find the index of the known points in a1,b1,c1 between which the point (a, b, c) lies
    i = int((a - 1.1)/0.1) - 1
    j = int((b - 0.5)/0.1) - 1
    k = int((c - 1.5)/0.1) - 1
    
    mx = (a - a1[i])/(a1[i+1] - a1[i])
    my = (b - b1[j])/(b1[j+1] - b1[j])
    mz = (c - c1[k])/(c1[k+1] - c1[k])
    
	# 8 corners of the cube surrounding the point of interest (a, b, c)
    c000 = A_[i, j, k]
    c001 = A_[i, j, k+1]
    c010 = A_[i, j+1, k]
    c011 = A_[i, j+1, k+1]
    c100 = A_[i+1, j, k]
    c101 = A_[i+1, j, k+1]
    c110 = A_[i+1, j+1, k]
    c111 = A_[i+1, j+1, k+1]
    
    
    c00 = c000*(1-mx) + c100*mx
    c01 = c001*(1-mx) + c101*mx
    c10 = c010*(1-mx) + c110*mx
    c11 = c011*(1-mx) + c111*mx
    
    c_0 = c00*(1-mx) + c10*mx
    c_1 = c01*(1-mx) + c11*mx
    
	# final interpolated value at the point
    c_ = c_0*(1-mx) + c_1*mx
    return c_

# 3D interpolation using trilinear interpolation for a range of values in (a,b,c)    
def A_3d(a, b, c): # a,b,c are arrays containing values you want to interpolate at 
    c_ = np.empty((len(a), len(b), len(c)))
    for l in range(len(a)):
        for m in range(len(b)):
            for p in range(len(c)):
                # to find the index of the known points in a1,b1,c1 between which the point (a[l], b[m], c[p]) lies
                i = int((a[l] - 1.1)/0.1) - 1
                j = int((b[m] - 0.5)/0.1) - 1
                k = int((c[p] - 1.5)/0.1) - 1

                mx = (a[l] - a1[i])/(a1[i+1] - a1[i])
                my = (b[m] - b1[j])/(b1[j+1] - b1[j])
                mz = (c[p] - c1[k])/(c1[k+1] - c1[k])
                
                # 8 corners of the cube surrounding the point of interest (a[l], b[m], c[p])
                c000 = A_[i, j, k]
                c001 = A_[i, j, k+1]
                c010 = A_[i, j+1, k]
                c011 = A_[i, j+1, k+1]
                c100 = A_[i+1, j, k]
                c101 = A_[i+1, j, k+1]
                c110 = A_[i+1, j+1, k]
                c111 = A_[i+1, j+1, k+1]


                c00 = c000*(1-mx) + c100*mx
                c01 = c001*(1-mx) + c101*mx
                c10 = c010*(1-mx) + c110*mx
                c11 = c011*(1-mx) + c111*mx

                c_0 = c00*(1-mx) + c10*mx
                c_1 = c01*(1-mx) + c11*mx

                c_[l,m,p] = c_0*(1-mx) + c_1*mx
    return c_

def q2h():
	pass	

if __name__ == '__main__':
	print('2 (a):')
	q2a()
	print('\n\n\n2 (b):')
	q2b()
	print('\n\n\n2 (c):')
	q2c()
	print('\n\n\n2 (d):')
	q2d()
	print('\n\n\n2 (e):')
	q2e()
	print('\n\n\n2 (f):')
	q2f()
	print('\n\n\n2 (g):')
	q2g()

