import numpy as np
import matplotlib.pyplot as plt
import scipy.special
from numba import njit

# test 2
class reflect_tools:
    """collection of function to build thin film stacks and simulate x-ray
        reflectivity with s or p polarization"""

#     def __init__(self):
#         self.test = ()
# #         # self.arg = arg

    def layer_builder(layer_n,layer_t,interface_rough,slice_t):
        # self.layer_n = layer_n
        # self.layer_t = layer_t
        # self.interface_rough = interface_rough
        # self.slice_t = slice_t
        # initialize arrays
        total_layer = []
        interfaces_idx = np.zeros(len(layer_t)-1)

        # assign interface indexes
        for i in range(len(layer_t)-1):
            interfaces_idx[i] = interfaces_idx[i-1] + int(layer_t[i]/slice_t)

        # create index of refraction array
        i = 0
        for filmt in layer_t:
            temp_layer = np.ones(int(filmt/slice_t))
            temp_layer = temp_layer*layer_n[i]
            total_layer = np.concatenate((total_layer,temp_layer))
            i = i + 1
        z = np.zeros((len(total_layer),1))

        # create depth position array
        for j in range(len(z)):
            z[j] = layer_t[0]-j*slice_t

        # incorporate surface roughness into array
        i = 0 # reset counter
        for rough in interface_rough:
            idx = int(interfaces_idx[i])
            if rough == 0:
                pass
            else:
                x = np.arange(-100,100,2)
                CDF = 1/2*(1 + scipy.special.erf(x/(rough*np.sqrt(2))))
                y_real = (total_layer[idx+1])*CDF + (total_layer[idx-1])*(1-CDF)
                total_layer[idx-50:idx+50] = y_real
            i = i+1
            # self.z_layers = z
            # self.n_layers = total_layer
        return z, total_layer

    def wvl_calc(energy):
        c = 3e8 # m/s
        h = 6.626e-34 # J*s
        e = 1.06218e-19 # electron charge
        wvl = c*h/energy/e*1e10 # nm
        return wvl

    @njit
    def reflect_s(wavelength, z_layers,n_layers,alpha_i):
        n_layers = n_layers.reshape(len(n_layers),1)
        z_layers = z_layers.reshape(len(z_layers),1)

        k0 = 2*np.pi/wavelength
        X = np.zeros((len(n_layers),len(alpha_i)),dtype=np.complex_)
        Rf = np.zeros((len(alpha_i),1),dtype=np.complex_)

        # z-component of wavevector
        kz = k0*np.sqrt(n_layers**2-np.cos(alpha_i)**2)

        r = (kz[0:-1,:] - kz[1:len(n_layers)+1,:])/(kz[0:-1,:] + kz[1:len(n_layers)+1,:])

        # Recursion to calculate reflectivity at surface
        for i in range(len(n_layers)-2,-1,-1):
            X[i,:] = (np.exp(-2.j*kz[i,:]*z_layers[i]) *
                    (r[i,:]+X[i+1,:]*np.exp(2.j*kz[i+1,:]*z_layers[i])) /
                    (1+r[i,:]*X[i+1,:]*np.exp(2.j*kz[i+1,:]*z_layers[i])))

        Rf = X[0,:]
        return Rf


    @njit
    def reflect_p(wavelength, z_layers, n_layers, alpha_i):
        n_layers = n_layers.reshape(len(n_layers),1)
        z_layers = z_layers.reshape(len(z_layers),1)

        k0 = 2*np.pi/wavelength
        X = np.zeros((len(n_layers),len(alpha_i)),dtype=np.complex_)
        Rf = np.zeros((len(alpha_i),1),dtype=np.complex_)
                # z-component of wavevector
        kz = k0*np.sqrt(n_layers**2-np.cos(alpha_i)**2)

        # r = (kz[0:-1,:] - kz[1:len(n_layers)+1,:])/(kz[0:-1,:] + kz[1:len(n_layers)+1,:])
        n = n_layers[1:]/n_layers[0:-1]
        r = (-n**2*kz[0:-1,:] + kz[1:len(n_layers)+1,:])/(n**2*kz[0:-1,:] + kz[1:len(n_layers)+1,:])

        # Recursion to calculate reflectivity at surface
        for i in range(len(n_layers)-2,-1,-1):
            X[i,:] = np.exp(-2.j*kz[i,:]*(z_layers[i]))*((r[i,:]+X[i+1,:]*np.exp(2.j*kz[i+1,:]*(z_layers[i]))) /\
                    (1+r[i,:]*X[i+1,:]*np.exp(2.j*kz[i+1,:]*(z_layers[i]))))

        Rf = X[0,:]
        return Rf

class EFI_tools:
    """ collection of tools to build thin films and calculate electric
        field intensities as a function of film depth and incidence angle"""

    def layer_builder(layer_n,layer_t,interface_rough,slice_t):
        # self.layer_n = layer_n
        # self.layer_t = layer_t
        # self.interface_rough = interface_rough
        # self.slice_t = slice_t
        # initialize arrays
        total_layer = []
        interfaces_idx = np.zeros(len(layer_t)-1)

        # assign interface indexes
        for i in range(len(layer_t)-1):
            interfaces_idx[i] = interfaces_idx[i-1] + int(layer_t[i]/slice_t)

        # create index of refraction array
        i = 0
        for filmt in layer_t:
            temp_layer = np.ones(int(filmt/slice_t))
            temp_layer = temp_layer*layer_n[i]
            total_layer = np.concatenate((total_layer,temp_layer))
            i = i + 1
        z = np.zeros((len(total_layer),1))

        # create depth position array
        for j in range(len(z)):
            z[j] = layer_t[0]-j*slice_t

        # incorporate surface roughness into array
        i = 0 # reset counter
        for rough in interface_rough:
            idx = int(interfaces_idx[i])
            if rough == 0:
                pass
            else:
                x = np.arange(-100,100,2)
                CDF = 1/2*(1 + scipy.special.erf(x/(rough*np.sqrt(2))))
                y_real = (total_layer[idx+1])*CDF + (total_layer[idx-1])*(1-CDF)
                total_layer[idx-50:idx+50] = y_real
            i = i+1
            # self.z_layers = z
            # self.n_layers = total_layer
        return z, total_layer

    @njit
    def EFI_s(wavelength, z_layers,n_layers,alpha_i):
        n_layers = n_layers.reshape(len(n_layers),1)
        z_layers = z_layers.reshape(len(z_layers),1)

        k0 = 2*np.pi/wavelength
        R = np.zeros((len(n_layers),len(alpha_i)),dtype=np.complex_)
        T = np.zeros((len(n_layers),len(alpha_i)),dtype=np.complex_)
        X = np.zeros((len(n_layers),len(alpha_i)),dtype=np.complex_)
        EFI = np.zeros((len(n_layers),len(alpha_i)))
        Rf = np.zeros((len(alpha_i),1),dtype=np.complex_)

        # z-component of wavevector
        kz = k0*np.sqrt(n_layers**2-np.cos(alpha_i)**2)

        r = (kz[0:-1,:] - kz[1:len(n_layers)+1,:])/(kz[0:-1,:] + kz[1:len(n_layers)+1,:])

        # Recursion to calculate reflectivity at surface
        for i in range(len(n_layers)-2,-1,-1):
            X[i,:] = (np.exp(-2.j*kz[i,:]*z_layers[i]) *
                    (r[i,:]+X[i+1,:]*np.exp(2.j*kz[i+1,:]*z_layers[i])) /
                    (1+r[i,:]*X[i+1,:]*np.exp(2.j*kz[i+1,:]*z_layers[i])))

        Rf = X[0,:]
        R[0,:] = X[0,:]
        T[0,:] = 1
        # Recursion to calculate R, T in film and used to calculate EFI
        rj1j = (kz[1:len(n_layers)+1,:] - kz[0:-1,:])/(kz[0:-1,:] + kz[1:len(n_layers)+1,:])
        tj1j = (1 + rj1j)
        for i in range(0,len(n_layers)-1):
            R[i+1,:] = ((1/tj1j[i,:]) *
                      (T[i,:]*rj1j[i,:]*np.exp(-1.j*(kz[i+1,:] + kz[i,:])*z_layers[i]) +
                      R[i,:]*np.exp(-1.j*(kz[i+1,:] - kz[i,:])*z_layers[i])))
            T[i+1,:] = ((1/tj1j[i,:]) *
                      (T[i,:]*np.exp(1.j*(kz[i+1,:] - kz[i,:])*z_layers[i]) +
                      R[i,:]*rj1j[i,:]*np.exp(1.j*(kz[i+1,:] + kz[i,:])*z_layers[i])))
        R[-1,:] = 0
        ER = R*np.exp(1.j*kz*z_layers)
        ET = T*np.exp(-1.j*kz*z_layers)
    #     print(ET.shape,ER.shape)
        EFI = (np.abs(ER + ET)**2)
        return Rf, EFI

    @njit
    def EFI_p(wavelength, z_layers,n_layers,alpha_i):
        n_layers = n_layers.reshape(len(n_layers),1)
        z_layers = z_layers.reshape(len(z_layers),1)

        k0 = 2*np.pi/wavelength
        R = np.zeros((len(n_layers),len(alpha_i)),dtype=np.complex_)
        T = np.zeros((len(n_layers),len(alpha_i)),dtype=np.complex_)
        X = np.zeros((len(n_layers),len(alpha_i)),dtype=np.complex_)
        EFI = np.zeros((len(n_layers),len(alpha_i)))
        Rf = np.zeros((len(alpha_i),1),dtype=np.complex_)

        # z-component of wavevector
        kz = k0*np.sqrt(n_layers**2-np.cos(alpha_i)**2)

        n = n_layers[1:]/n_layers[0:-1]
        r = (-n**2*kz[0:-1,:] + kz[1:len(n_layers)+1,:])/(n**2*kz[0:-1,:] + kz[1:len(n_layers)+1,:])

        # Recursion to calculate reflectivity at surface
        for i in range(len(n_layers)-2,-1,-1):
            X[i,:] = np.exp(-2.j*kz[i,:]*(z_layers[i]))*((r[i,:]+X[i+1,:]*np.exp(2.j*kz[i+1,:]*(z_layers[i]))) /\
                    (1+r[i,:]*X[i+1,:]*np.exp(2.j*kz[i+1,:]*(z_layers[i]))))

        Rf = X[0,:]
        R[0,:] = X[0,:]
        T[0,:] = 1
        # Recursion to calculate R, T in film and used to calculate EFI
        # n = n_layers[0:-1]/n_layers[1:]
        rj1j = (-n**2*kz[1:len(n_layers)+1,:] + kz[0:-1,:])/(kz[0:-1,:] + n**2*kz[1:len(n_layers)+1,:])
        # tj1j = 2*n*kz[1:len(n_layers)+1,:]/(n**2*kz[1:len(n_layers)+1,:]+kz[0:-1,:])
        tj1j = 1+rj1j
        for i in range(0,len(n_layers)-1):
            R[i+1,:] = ((1/tj1j[i,:]) *
                      (T[i,:]*rj1j[i,:]*np.exp(-1.j*(kz[i+1,:] + kz[i,:])*z_layers[i]) +
                      R[i,:]*np.exp(-1.j*(kz[i+1,:] - kz[i,:])*z_layers[i])))
            T[i+1,:] = ((1/tj1j[i,:]) *
                      (T[i,:]*np.exp(1.j*(kz[i+1,:] - kz[i,:])*z_layers[i]) +
                      R[i,:]*rj1j[i,:]*np.exp(1.j*(kz[i+1,:] + kz[i,:])*z_layers[i])))
        R[-1,:] = 0
        ER = R*np.exp(1.j*kz*z_layers)
        ET = T*np.exp(-1.j*kz*z_layers)
    #     print(ET.shape,ER.shape)
        EFI = (np.abs(ER + ET)**2)
        return Rf, EFI, ER, ET, R, T

if __name__ == '__main__':

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    alpha_i = np.deg2rad(np.arange(0.1,89,.5))
    t_PS = 150 # angstroms
    t_PMMA = 450 # angstroms

    n_air = np.complex(1,0)
    # 280 eV
    delta_PMMA = 0.00146372709
    beta_PMMA = 0.000126856146
    n_PMMA = np.complex(1-delta_PMMA,beta_PMMA)

    delta_PS = 0.000624888053
    beta_PS = 6.27421832E-05
    n_PS = np.complex(1-delta_PS,beta_PS)

    delta_Si = 0.00509571517
    beta_Si = 0.00303027453
    n_Si = np.complex(1-delta_Si,beta_Si)

    wvl = reflect_tools.wvl_calc(280)
    # print(wvl)
    qz = 4*np.pi/wvl*np.sqrt(1-np.cos(alpha_i)**2)
    z_layers, n_layers = reflect_tools.layer_builder((n_air,n_PS,n_PMMA,n_Si),(100,t_PS,t_PMMA,100),(0,0,0),2)
    Rp = reflect_tools.reflect_p(wvl,z_layers,n_layers,alpha_i)
    Rs = reflect_tools.reflect_s(wvl,z_layers,n_layers,alpha_i)

    fig, ax = plt.subplots()
    plt.semilogy(qz,np.abs(Rp)**2,label='p-polarization')
    plt.semilogy(qz,np.abs(Rs)**2,label='s-polarization')
    plt.xlabel(r'q$_z$ [$\AA^{-1}$]')
    plt.ylabel('Reflectivity')
    plt.legend(loc='lower left')
    plt.title(r'$\lambda$ = 280 eV')
    axin = inset_axes(ax,width='35%',height='35%')
    axin.plot(-z_layers,1-n_layers.real)
    plt.show()

    # _, EFI_s = EFI_tools.EFI_s(wvl,z_layers,n_layers,alpha_i)
    # plt.figure()
    # plt.imshow(EFI_s,cmap='jet',aspect='auto',extent=[0.1,89,-700,100])

    # Rp, EFI_p, ER, ET, R, T = EFI_tools.EFI_p(wvl,z_layers,n_layers,alpha_i)
    # plt.figure()
    # plt.imshow(EFI_p,cmap='jet',aspect='auto',extent=[0.1,89,-700,100])

    # # wvl = 1.54 angstroms
    # # PMMA
    # delta_PMMA = 4.0918294E-06
    # beta_PMMA = 8.91067753E-09
    # n_PMMA = np.complex(1-delta_PMMA,beta_PMMA)

    # # Polystyrene
    # delta_PS = 3.45346848E-06
    # beta_PS = 4.89482366E-09
    # n_PS = np.complex(1-delta_PS, beta_PS)

    # # Silicon
    # delta_Si = 7.57536282E-06
    # beta_Si = 1.72802345E-07
    # n_Si = np.complex(1-delta_Si, beta_Si)

    # alpha_i = np.deg2rad(np.arange(0.1,.4,.0001))
    # z_layers, n_layers = reflect_tools.layer_builder((n_air,n_PS,n_PMMA,n_Si),(100,t_PS,t_PMMA,100),(0,0,0),2)
    # Rp = reflect_tools.reflect_p(1.54,z_layers,n_layers,alpha_i)
    # Rs = reflect_tools.reflect_s(1.54,z_layers,n_layers,alpha_i)

    # fig, ax = plt.subplots()
    # qz = 2*np.pi/1.54*np.sqrt(1-np.cos(alpha_i)**2)
    # plt.semilogy(qz,np.abs(Rp)**2,label='p-polarization')
    # plt.semilogy(qz,np.abs(Rs)**2,label='s-polarization')
    # plt.xlabel(r'q$_z$ [$\AA^{-1}$]')
    # plt.ylabel('Reflectivity')
    # plt.legend(loc='lower left')
    # plt.legend(loc='lower left')
    # plt.title(r'$\lambda$ = 8051 eV')
    # axin = inset_axes(ax,width='35%',height='35%')
    # axin.plot(-z_layers,1-n_layers.real)
    # plt.show()

    # _, EFI_s = EFI_tools.EFI_s(1.54,z_layers,n_layers,alpha_i)
    # plt.figure()
    # plt.imshow(EFI_s,cmap='jet',aspect='auto',extent=[0.1,89,-700,100])

    # Rp, EFI_p, ER, ET, R, T = EFI_tools.EFI_p(1.54,z_layers,n_layers,alpha_i)
    # plt.figure()
    # plt.imshow(EFI_p,cmap='jet',aspect='auto',extent=[0.1,89,-700,100])
