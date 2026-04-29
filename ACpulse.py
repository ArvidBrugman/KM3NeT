import numpy as np
import scipy
import math
from scipy.interpolate import interp1d
from scipy.signal import windows
import scipy.interpolate as sip
from scipy.special import gamma


#import Shower

""" required physical constants """
sound_speed_water = 1500.0 #m/s
beta  = 2e-4         # 1/K bulk_expantion
Cp    = 3.8e3        # J/kg/K specific heat capacity
eV2J  = 1.6e-19      
pi    = math.pi
#gamma = eV2J * alpha * (sound_speed_water**2)/(4*pi*Cp)

sampling_rate = 1e6
delta_R = sound_speed_water/sampling_rate



def histc( X , bins ):

    # Define the bin edges with a vector, where the first element is the left edge of the first bin, 
    # and the last element is the right edge of the last bin
    clean_X = np.array(X)[ ( X >= np.array(bins)[0] ) & ( X <= np.array(bins)[-1] ) ]
    
    map_to_bins = np.digitize(clean_X, np.array(bins))
    r = np.zeros(np.array(bins).shape)
    
    for i in map_to_bins: r[i-1] += 1
        
    return [ r , map_to_bins ]



class ACpulse:
    """
    This class creates a neutrino signal from a rho-E map.
    """
    def __init__(self):
        self.hydropos = None
        self.showerenergy = 1e20
        self.Showerhisto = None
        self.t_axis = None
        self.plot_me = False
        
        self.r_min = 0. #cm
        self.r_max = 100. #cm
        self.Rbins = 50

        self.z_min = 0. #cm 
        self.z_max = 50000. #cm
        self.Zbins = 50
        #        self.z_min = 0
        self.z_max = 5000

        self.nmc = 9.9e5
        self.R = np.array([])
        self.EdepModel = 'A'
        self.model_constant = 10
        self.Fs = 1e6 # sampling frequency (default 1MHz)

        
    def __del__(self):
        if not self.hydropos == None:
            del self.hydropos
        if not self.Showerhisto == None:
            del self.Showerhisto

    def getSignal(self):
        self.Shower()
        self.MCGEn()
        return self.bipolarpulse()

    def Shower(self):
#        return self.cylindric_shower()
#        return self.Niess()
        return self.Saund()


    def showerFile(self, filename):
        self.filename = filename

    def sample_frequency(self, Fs):
        print('sample frequency   {0:e}'.format(Fs))
        self.Fs = Fs

    def shower_energy(self, E):
        print('shower energy   {0:e}'.format(E))
        self.showerenergy = E

    def radial_binning(self, Rbins):
        print('Radial bins  {0:e}'.format(Rbins))
        self.Rbins = Rbins
 
    def long_binning(self, Zbins):
        print('Longitudinal bins  {0:e}'.format(Zbins))
        self.Zbins = Zbins

    def hydrophonePosition(self, pos):
        # note: this is xyz, not rz #fixme, it is confusing..
        print('hydrophone position ', abs(pos[0]), 0, pos[1])
        self.hydropos = (abs(pos[0]), 0, pos[1])
        
    def pol2cart(self, rho, phi, z):
        # convert cylindrical coordinates to cartesian
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return np.array( [ np.array(x) , np.array(y) , np.array(z) ] )

    def attenuation(self, f, dkm):
        # ATTEN_FNA calculates attenuation in Sea water.
        # Inputs:  f (Hz) and dkm (distance in km)
        # Outputs: attenuation as a fraction of the pressure
        #          remaing at the particular frequency.
        # based on Ainslie and McColm J. Acoust. Soc. AM. 103 (3) 1671 1998
        # Last Mod SD 26/1/07; python code by EJB

        # f in kHz and z in km
        T  = 15.0
        S  = 37.0
        pH = 7.9
        z  = 2.0
        f1 = 0.78 * math.sqrt(S/35.) * math.exp(T/26.)
        f2 = 42 * math.exp(T/17.)
        f  = f*1e-3 # Convert to kHz
       
        attendb = 0.106*f1 * f**2/(f**2 + f1**2) * \
                  math.exp((pH-8.)/0.56) + \
                  0.52 * (1+T/43.) * (S/35.) * \
                  f2*f**2/(f**2 + f2**2) * math.exp(-z/6.) + \
                  0.00049 * f**2 * math.exp(-T/27. + z/17.)
#        print("attendb ", attendb, len(attendb), len(f), 10**-((attendb*dkm)/20.))
        return 10**-((attendb*dkm)/20.)

    def bipolarpulse(self):
        # UltraFast Acoustic Integral function
        # Inputs: Points [x,y,z] where z is oriented along the
        #         axis of the shower, units (m) size n x 3 where n
        #         is typically about 10^6. Note the value of n is
        #         very distance dependent. At 100m c 10^8 points are
        #         needed to give an ultra-clean pulse.
        #         At 10km c 10^4 points are sufficient.
        #
        # Do: is the position of the observer [x0,y0,z0]
        #

        # Outputs: p the pulse (sampling rate 1MHz default)
        # Note: zero time is at the mean shower
        #       transit time ignoring complex attenuation
        #       pw the FFT of the pulse (sampling rate 1MHz default)
        #       Exyz is a scaled version of the Velocity Potential
        # SD Last mod 7/7/08, python code by EJB

        nr = 25
        fs = self.Fs

        # binning of the time trace
        bin_a = math.pow(2,16)
        bin_b =  bin_a -1
        bin_a = -bin_a
        RR = int( abs(bin_a) + abs(bin_b) + 1  )
        time_bins = np.linspace( bin_a , bin_b , RR )
        self.t_axis = time_bins / fs
    
        f1 = np.linspace( 0 , abs(bin_a) , int( abs(bin_a)+1 ))
        f2 = np.linspace( -abs(bin_b) , -1 , int( abs(bin_b) ))
        frequency_bins = np.concatenate( ( f1 , f2 ) )
        f_axis = ( frequency_bins / RR)  *  fs

        # differential in Frequency Domain. The d/dt of exp(iwt) is iw exp(iwt)
        # A Blackman window is used to smooth the integral and is optional
        diff_filt = 1j * 2 * pi * f_axis * np.fft.fftshift( np.blackman( RR ) ) 

        # Create Velocity potential array
        Exyz = np.zeros(len(self.t_axis))

        # Convert the observer position to a matrix
        # of the same size as the number of points
        D = [self.hydropos]*int(self.nmc) # to calculate the distance from the observer for each point

        # If rotational symmetry is being used, loop through the angles
        phi = np.linspace( 0 , 360 , 1+nr )
        phi = phi[0:-1] # avoid double counting at zero
    
        # loop nr times over phi
        for i, ang in enumerate(phi):
            print("Progress: %d percent" % (100*i/len(phi)), end = "\r" )
            # convert to radians
            ang = (ang/180.0)*pi

            # Perform the transformation: rotate [lpoints,rpoints]
            points_rot =  self.points.T @ np.array( [ [  np.cos(ang), np.sin(ang), 0 ] ,
                                                      [ -np.sin(ang), np.cos(ang), 0 ] ,
                                                      [ 0 , 0 , 1 ] ] ) 
            
            # determine the distance to each point
            d2 = (points_rot-D)**2
            d  = np.sqrt( np.sum( d2 , axis=1 ) )
            
            # Determine the mean distance if not provided
            m = sum(d)/self.nmc

            # Determine the unscaled velocity potential. (Need to divide by distance)
            # check bin edges (are different in matlab)
            his = histc(d/sound_speed_water, self.t_axis+m/sound_speed_water) 
            Exyz = Exyz + his[0]
            
        # Normalise sum(Exyz) to one. 
        Exyz = Exyz/len(phi)/self.nmc

        # Scale by constants and divide by distance. 
        Exyzn = Exyz*beta/ Cp/ 4/ pi*self.showerenergy * 1.6e-19/(m+self.t_axis*sound_speed_water)
    
        # Do integral in the frequency domain
        pw = np.fft.fft(Exyzn) * diff_filt * self.attenuation(f_axis, m*1e-3)
        n = Exyzn.size
        timestep = 1/fs
        freq = np.fft.fftfreq(n, d=timestep)
        pw_max = np.amax(np.abs(pw))

        # Convert back to the time domain
        p = np.real(np.fft.ifft(pw)*fs)
#        print("length time trace and freq spectrum: ", len(self.t_axis), len(p))
        return self.t_axis, p


    def MCGEn(self):
        # agruments: jpri, lscale, rscale, n, imethod=None
        # MCGen Table based Monte Carlo Generator
        # Inputs
        # jpri - 2D-Histogam whose statistics we wish to mimic: size mxn
        # lscale - bin edges of the rows size: (m+1)*1
        # rscale - bin edges of the columns size: 1*(n+1)
        # n - the number of points to be produced
        # Last Mod 27/1/07 SD, python by EJB

        jpri = self.Edep
        # Add 1e-10 to ensure monotonic increase
        long = np.sum(jpri, axis=1) + 1e-11 * np.sum(jpri)

        n = self.nmc
        lscale = self.Z
        rscale = self.R
        
        # Interpolate lscale
        lpoints = np.interp( np.random.rand(int(n)),
                             np.concatenate(([0], np.cumsum(long) / np.sum(long))),
                             self.z_edges )
        rpoints = np.zeros(int(n))

        Throw = histc( lpoints , lscale[:,0] )
        nthrowa = Throw[0]
        
        spos = 0
        for i in range( len(nthrowa) - 1 ):
            nthrow = int(nthrowa[i])

            if nthrow > 0:
                radial = jpri[i, :] + 1e-8
                rpoints[spos:(spos + nthrow)]  = \
                    np.interp( np.random.rand(nthrow),
                               np.concatenate(([0], np.cumsum(radial) / np.sum(radial))),
                               self.r_edges)
                spos += nthrow
    
        pointsc = np.column_stack((lpoints, rpoints))

        # Convert to cartesian
        random_phase_array = [np.random.uniform(0, 2*pi) for _ in range(int(n))]
        self.points = self.pol2cart( pointsc[:,1], random_phase_array, pointsc[:,0]) 

        if self.plot_me:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
#            ax = fig.add_subplot(111, projection='3d')
#            ax.scatter(self.points[0], self.points[1], self.points[2])
            plt.hist2d(self.points[0], self.points[1], bins=100, cmap=plt.cm.jet)
#            ax.scatter(self.points[0], self.points[1], self.points[2])
            plt.show()
        
        self.points *= 1e-2  # Convert fom cm to m 
        return self.points


    ############# Shower histogram  #################
    def shower_2dhisto(self):
        # just an empty meshgrid

        r_bins =  int(self.Rbins)
        r_bin_size = (self.r_max - self.r_min)/self.Rbins

        self.r = np.arange(self.r_min + 0.5*r_bin_size, (r_bins-1) * r_bin_size, r_bin_size)
        self.r_edges = np.arange(self.r_min, self.r_max, r_bin_size)

        z_bins = int(self.Zbins)
        z_bin_size = (self.z_max - self.z_min)/self.Zbins
        self.z = np.arange(self.z_min + 0.5*z_bin_size, (z_bins-1)* z_bin_size, z_bin_size)
        self.z_edges = np.arange(self.z_min, self.z_max, z_bin_size)

#        print("min, max radii: ", self.r_min, self.r_max)
#        print("min, max z:     ", self.z_min, self.z_max)
#        print("bin sizes:      ", r_bin_size, z_bin_size)
#        
        self.R, self.Z = np.meshgrid(self.r, self.z)


    ############# Shower parmaterizations  #################
    def cylindric_shower(self):
        if not self.R.any():
            self.shower_2dhisto()
            
        if not self.EdepModel:
            self.EdepModel = 'A'

        self.Edep = np.ones_like(self.R)
        const = self.model_constant

        if self.EdepModel == 'A': # Gauss
            #return A * np.exp(-x**2 / (2 * sigma**2))
            self.Edep = np.exp(-self.R**2/(2*const**2))
            
        if self.EdepModel == 'B':
            self.Edep = const/((self.R))

        if self.EdepModel == 'C':
            self.Edep = const/(self.R*self.R)

        if self.EdepModel == 'D':
            self.Edep = np.exp(-self.R*const)

        if self.EdepModel == 'E': # Lorentzian
            self.Edep = const/ (self.R**2 + const**2)

        # try for a window function to get rid of the edge effects
        window = windows.tukey(len(self.z), 0.2)
        self.Edep *= window[:, np.newaxis]  # Apply window across all columns

        return self.Edep

        
    def Niess(self):
        """
        Niess & Bertin radial + longitudinal shower parameterisation.
        
        Returns
        -------
        yy : 2D array, (shower energy density distribution (transposed))
        """

        # Create grids (MATLAB: [rg, zg] = meshgrid(r, z))
        if not self.R.any():
            self.shower_2dhisto()
        rg = self.R
        zg = self.Z 
        Eo = self.showerenergy

        Ec = 0.05427
        Xo = 35.29      # radiation length (cm)
        b  = 0.56
        
        # --- Shower maximum position ---
        zpmax = 0.65 * np.log(Eo/Ec) + 3.93   # MATLAB log = ln
        zmax = Xo * zpmax
        a = b * zpmax + 1

        # Radial distribution
        x = 3.5 / rg
        n1 = 1.66 - 0.29 * (zg/zmax)
        n2 = 2.7

        # MATLAB: y = x.^n1 .* (x>1) + x.^n2 .* (x<=1)
        y = np.where(x > 1, x ** n1, x ** n2)
    
        # Multiply by r
        y = y * rg
    
        # MATLAB: y = diag(1./sqrt(sum(y.^2,2))) * y
        # → Normalize each row
        row_norm = np.sqrt(np.sum(y ** 2, axis=1, keepdims=True))
        y = y / row_norm

        # Longitudinal distribution
        zp = zg / Xo
        long = (Eo / Xo) * (b * zp) ** (a - 1) * np.exp(-b * zp) / gamma(a)
        
        y = y * long

        # MATLAB: yy = [y']
        yy = y.T
        self.Edep = yy
        return y

    

    def Saund(self):
        """
        SAUND parameterisation of acoustic energy deposition (Sloan & VanderBrugge)
        
        Parameters
        ----------
        r : array-like
        radial coordinates (cm)
        z : array-like
        longitudinal coordinates (cm)
        Eo : float
        shower energy (GeV)

        Returns
        -------
        tsmc : 2D array (len(z), len(r))
        energy deposition distribution
        """
        if not self.R.any():
            self.shower_2dhisto()
        r = self.R
        z = self.Z 
        Eo = self.showerenergy

#        r = np.array(r).reshape(1, -1)  # row vector
#        z = np.array(z).reshape(-1, 1)  # column vector
        
        Xo = 36.1         # cm
        Ec = 0.0838       # GeV
        r_m = 9.04        # cm
        s = 1.25

        zmax = 0.9 * Xo * np.log(Eo / Ec)
        lambda_ = 130 - 5 * np.log10(Eo * 1e-4)
        t = zmax / lambda_
        k = t**(t - 1) / np.exp(t) / lambda_ / gamma(t)

        # ---------- radial distribution ----------
        a = r / r_m
        
        rho = (1 / r_m**2) * (a**(s - 2)) * ((1 + a)**(s - 4.5)) * \
            gamma(4.5 - s) / (2 * np.pi * gamma(s) * gamma(4.5 - 2 * s))

        rho = rho * r * 2 * np.pi  # multiply by r to integrate in cylindrical coords
        rho = rho / np.sum(rho)    # normalize
        
        # ---------- longitudinal distribution ----------
        zdist = k * (z / zmax)**t * np.exp(t - z / lambda_)
        
        # ---------- full 2D distribution ----------
        tsmc = Eo * zdist * rho
        self.Edep = tsmc
        return tsmc



