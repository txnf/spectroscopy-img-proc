import scipy as sp
import numpy as np
import pylab as plt
import tifffile
import itertools
import matplotlib as mpl
import scipy.ndimage as ndimg
import skimage
import skimage.feature.peak
import os.path
import glob
import functools
import json
import einops

# data processing code for order extraction and merging echelle spectrometers

# glass dispersion model and common glasses for initial conditions
def n_sell(lam, B1=1.03, B2=0.23, B3 = 1.01, C1=6e-3, C2=2.0e-2, C3=1.03e2):
    l2 = lam**2
    return np.sqrt(1 + B1*l2/(l2 - C1) + B2*l2/(l2 - C2) + B3*l2/(l2 - C3))
bk7 = (1.03961212,	0.231792344,	1.01046945,	6.00069867e-3,	2.00179144e-2,	103.560653)
sapp = (1.43134930,	0.65054713,	5.3414021,	5.2799261e-3,	1.42382647e-2,	325.017834)
fs = (0.696166300,	0.407942600,	0.897479400,	4.67914826e-3,	1.35120631e-2,	97.9340025)
mgf2 = (0.48755108,	0.39875031,	2.3120353,	0.001882178,	0.008951888,	566.13559)

# spectral lines and orders for the Hgar calibration lamp
# hgar_lams = np.array([0.25365, 0.25365,   0.296728, 0.296728,  0.30215, 0.30215,  
# 0.312567, 0.312567, 0.36502, 0.36502, 0.40466, 0.40466, 0.43583, 0.43583, 0.54607, 0.54607, 0.57907, 0.57907,
# 0.696543, 0.763511, 0.811531, 0.912279, ])
# hgar_ms = np.array([42, 43, 36, 37, 35, 36, 34, 35, 29, 
# 30, 26, 27, 24, 25, 19, 20, 18, 19, 16, 14, 13, 12 ])

hgar_lams = np.array([0.25365, 0.25365, 0.25365, 0.25365,   0.296728, 0.296728,  0.30215, 0.30215,  
0.312567, 0.312567, 0.36502, 0.36502, 0.40466, 0.40466, 0.43583, 0.43583, 0.54607, 0.54607, 0.57907, 0.57907,
0.696543, 0.763511, 0.811531, 0.912279, ])
hgar_ms = np.array([41, 42, 43, 44, 36, 37, 35, 36, 34, 35, 29, 
30, 26, 27, 24, 25, 19, 20, 18, 19, 16, 14, 13, 12 ])


# planck blackbody function in spectral radiance units
def planck_bb_wm2srum(lam, T):
    c1 = 1.191041e8 # W / m^2 / sr / um^4
    c2 = 1.4387752e4 # um K
    return c1/(sp.exp(c2/lam/T) - 1)/lam**5 # W/m^2/sr/um


# peak finding in image for refinement of fit

def find_nearest_peak(img, x, y, r=25):
    x,y = int(x), int(y)
    s = img[x-r:x+r,y-r:y+r]
    pks = skimage.feature.peak.peak_local_max(s, min_distance=1, exclude_border=True)
    print(f"pks = {pks}")
    d = np.sqrt((pks[:,0] - x)**2 + (pks[:,1] - y)**2)
    mi = np.argmin(d)
    return pks[mi,0]+x-2*r, pks[mi,1]+y-2*r

class Echelle:
    def __init__(self, img, wlcal=None, radcal=None):
    
        self.glass = sapp

        self.lams = np.arange(0.2,1,0.0001)
        print(f"setting spectral range: (0.2 um  - 1.0 um dl = 0.00005um")

        self.ms = np.arange(9,45)
        print(f"using orders m=9 - 45")

        self.img = img
        
        
        self.border_x = 5
        self.border_y = 3

        if wlcal is None:
            print(f"no wavelength calibration provided")
        else:
            self.wlcal = wlcal
        
        if radcal is None:
            print(f"no radiometric calibration provided")

    
    def plot_frames(self):
        if self.img.ndim < 3:
            print(f"not an image stack: self.img.ndim = {self.img.ndim}")
            return
        f,a = plt.subplots(1,1)    
        a.plot(self.img.mean((1,2)))


    def display_image(self, frame=None, ax=None, **kwargs):
        if ax is None:
            fig,ax = plt.subplots(1,1)
        if self.img.ndim == 2:
            ax.imshow(self.img, aspect='auto', cmap=plt.cm.gray_r,  **kwargs)
        elif self.img.ndim == 3:
            if frame:
                ax.imshow(self.img[frame,:,:], aspect='auto', cmap=plt.cm.gray_r, **kwargs)
            else:
                ax.imshow(self.img.mean(0), aspect='auto', cmap=plt.cm.gray_r, **kwargs)



    def plot_points(self, lams, orders, frame=None, ax=None, **kwargs):
        if ax is None:
            fig,ax = plt.subplots(1,1)
        xs, ys = Echelle.echelle(lams, orders, *self.wlcal)
        ax.plot(xs, ys, lw=0, **kwargs)
    
    def wlcal_initial_guess(self, pts):
        print(f"running initial guess: {pts}")
        lams, ms, xs, ys = [np.array(ai) for ai in zip(*pts)]
        #print(f"{lams} \n {ms} \n {xs} \n {ys}")
        d0 = 1000/65
        ang0 = 65
        dms, ams = Echelle.echelle(lams, ms, 1, 0, 1, 0, d0, *self.glass, ang0)
        (sx, x0), _, _, _ = np.linalg.lstsq(np.c_[dms, np.ones(len(dms))], xs)
        (sy, y0), _, _, _ = np.linalg.lstsq(np.c_[ams, np.ones(len(ams))], ys)
        
        self.wlcal = sx, x0, sy, y0, d0, *self.glass, ang0
    

    def wlcal_refine_guess_hgar(self):
        """
        refine the wavelength calibration - assuming that the current image is a hgar spectrum
        """
        print(f"refining wl calibration fit")
        init_xs, init_ys = Echelle.echelle(hgar_lams, hgar_ms, *self.wlcal)
        #print(f"{init_xs}, {init_ys}")

        refined_xys = [Echelle.find_nearest_max(self.img, ixsi, iysi, r=20) for ixsi,iysi in zip(init_xs, init_ys)]
        ref_xs, ref_ys = zip(*refined_xys)
        #print(f"{ref_xs}, {ref_ys}")

        print("refining wl calibration with full hgar spectrum")
        
        def f(p):
            sx, x0, sy, y0, d, b1, b2, b3, c1, c2, c3, ang = p
            xst, yst = Echelle.echelle(hgar_lams, hgar_ms, *p)
            # dist = np.sqrt(np.sum((ref_xs-xst)**2 + (ref_ys-yst)**2))
            dist = np.sum(np.sqrt((ref_xs-xst)**2 + (ref_ys-yst)**2))
            
            #print(f"dist = {dist}")
            return dist
        
        popt = sp.optimize.minimize(f, self.wlcal, options={'maxiter' : 2000, 'disp' : True})
        print(f"optimized wlcal = {popt.x}")
        self.wlcal = popt.x
        txs, tys = Echelle.echelle(hgar_lams, hgar_ms, *self.wlcal)
        return ref_xs, ref_ys, txs, tys
   
    @staticmethod
    def find_nearest_max(img, x, y, r=16):
        """ find the nearest maximum point within radius r - used for peak fit refinement"""
        x,y = int(x), int(y)
        s = img.T[x-r:x+r,y-r:y+r]
        xi,yi = np.unravel_index(np.argmax(s, axis=None), s.shape)
        return xi+x-r, yi+y-r

    @staticmethod
    # def echelle(lam, m, sx, x0, sy, y0, d, b1, b2, b3, c1, c2, c3):
    def echelle(lam, m, sx, x0, sy, y0, d, b1, b2, b3, c1, c2, c3, ang):
        """
        Echelle response function that maps (lam,m) to a point in pixel space
        parameters include 
            (sx,x0,sy,y0) - scale and offset mapping points to pixel space
            d - grating dispersion
            (b1,b2,b3,c1,c2,c3) - glass dispersion parameters
        """
        apex = sp.pi*ang/180
        dang = np.arcsin(m * (lam / d))
        cang = apex * (n_sell(lam, b1, b2, b3, c1, c2, c3) - 1)
        x = sx * dang +  x0
        y = sy * cang +  y0
        
        return x,y

    def extract_orders(self):
        self.pxmsks = []
        self.orders = []
        self.ord_res = []
        for m in self.ms:
            # print(f"extracting order {m}")
            xs, ys = Echelle.echelle(self.lams, m, *self.wlcal) 
            xspx, yspx = xs.astype(int), ys.astype(int)

            # print(xspx.shape, yspx.shape)

            pxmsk *= (yspx < self.img.shape[-2]-4*self.border_y)*(yspx > 4*self.border_y)
            pxmsk = (xspx < self.img.shape[-1]-4*self.border_x)*(xspx > 4*self.border_x)
            s = Echelle.project_order(self.img, self.lams[pxmsk], xs[pxmsk], ys[pxmsk], self.border_x, self.border_y)  
            self.pxmsks.append(pxmsk)
            self.orders.append(s)
            self.ord_res.append(np.ones(s.shape[0]))
          


    def order_mask(self):
        ordmsks = []
        lams = []
        for m in self.ms:
            xs, ys = Echelle.echelle(self.lams, m, *self.wlcal)
            xspx, yspx = xs.astype(int), ys.astype(int)
            pxmsk = (xspx < self.img.shape[-1]-4*self.border_x)*(xspx > 4*self.border_x)
            pxmsk *= (yspx < self.img.shape[-2]-4*self.border_y)*(yspx > 4*self.border_y)
            sz = np.sum(pxmsk)
            # print(f"order {m}, size {sz}")
            if sz <= 0:
                continue
            mmsk = np.stack([yspx[pxmsk], xspx[pxmsk]])
            ordmsks.append(mmsk)
            lams.append(self.lams[pxmsk])
        return ordmsks, lams
            
            

    def plot_orders(self, ax=None, frame=None):
        if ax is None:
            f,ax =plt.subplots(1,1)
        if self.img.ndim == 3:
            for m, pxmsk, ord, res in zip(self.ms, self.pxmsks, self.orders, self.ord_res):
                ax.plot(self.lams[pxmsk], ord[:,frame]*res, label=f'm={m}')
        else:
            for m, pxmsk, ord, res in zip(self.ms, self.pxmsks, self.orders, self.ord_res):
                ax.plot(self.lams[pxmsk], ord*res, label=f'm={m}')
        ax.set_yscale('log')


    def derive_order_response(self, temp=2500):
        lref = planck_bb_wm2srum(self.lams, temp)
        self.ord_res = []
        for m, pxmsk, ord in zip(self.ms, self.pxmsks, self.orders):
            self.ord_res.append(lref[pxmsk] / ord)
        

    def apply_order_response(self, res):
        self.ord_res = res
            

    def polish_orders(self):
        """
        ensure that orders are continuous at the boundary
        """
       
        
    

    @staticmethod
    def project_order(timg, lam, xs, ys, wx=3, wy=5):
        knx, kny = int(3*wx), int(3*wy)
        s = []
        for l,x,y in zip(lam, xs, ys):
            px, py = int(round(x)), int(round(y))
            xg, yg = np.meshgrid(np.arange(px-knx,px+knx), np.arange(py-kny, py+kny))
            kern = np.exp(-1*((x-xg)/wx)**2 - ((y-yg)/wy)**2)
            kern /= kern.sum()

            if timg.ndim == 2:
                smap = timg.T[px-knx:px+knx, py-kny:py+kny]
                s.append((smap*kern.T).sum())
            elif timg.ndim == 3:
                smap = np.transpose(timg, (0,2,1))[:, px-knx:px+knx, py-kny:py+kny]
                s.append((smap*kern.T).sum((1,2)))
        return np.array(s)

def full_prj(A, Q, k):
    P, T, H, W = A.shape

    # Create a vertical sliding window view of A
    shape = (P, T, H - k + 1, W, k)
    strides = (A.strides[0], A.strides[1], A.strides[2], A.strides[3], A.strides[2])
    A_strided = np.lib.stride_tricks.as_strided(A, shape=shape, strides=strides)

    # Get the indices of the upper bounds of the windows
    indices = Q[0,:] - k//2
    indices = np.clip(indices, 0, H - k)

    # Get the window sums
    R = np.sum(A_strided[:, :, indices, Q[1,:]], axis=-1)

    return R


# planck blackbody function in spectral radiance units
def planck_bb_wm2srum(lam, T):
    c1 = 1.191041e8 # W / m^2 / sr / um^4
    c2 = 1.4387752e4 # um K
    return c1/(np.exp(c2/lam/T) - 1)/lam**5 # W/m^2/sr/um
    
def project_image(A, Q, k):
    # print(f"A.shape = {A.shape}")
    T, H, W = A.shape

    # Create a vertical sliding window view of A
    shape = (T, H - k + 1, W, k)
    strides = (A.strides[0], A.strides[1], A.strides[2], A.strides[1])
    A_strided = np.lib.stride_tricks.as_strided(A, shape=shape, strides=strides)

    # Get the indices of the upper bounds of the windows
    indices = Q[0,:] - k//2
    indices = np.clip(indices, 0, H - k)

    # Get the window sums
    R = np.sum(A_strided[:, indices, Q[1,:]], axis=-1)

    return R


# process flat fields

def process_flat(npz_file, wlcal, bg=None, k=10):
    df = np.load(npz_file)
    # print(f"{list(df.keys())}")
    img = df['frames'].astype(np.int16)
    
    # if 'times' in df:
    #     times = df['times']
    # else:
    #     times = np.zeros(1)

    int_times = df['int_times']

    print(f"img.shape = {img.shape}")

    if bg is not None:
        bgf = np.load(bg)
        bgi = bgf['frames'].astype(np.int16)
        bgts = bgf['int_times']
        print(f"bgi.shape = {bgi.shape}")
        print(f"bg int_times = {bgts}")
        print(f"int_times = {int_times}")
        # tt = einops.rearrange(bgi, 't h w -> t () h w')
        img = img - bgi

    ee = Echelle(img[0], wlcal = wlcal)
    oms, lams = ee.order_mask()

    ords = []
    for i, om in enumerate(oms):
        r = project_image(img, om, k)
        ords.append(r)

    return int_times, lams, ords, ee.lams, ee.ms


def process_run(npz_file, wlcal, bg=None, k=10):
    df = np.load(npz_file)
    print(f"{list(df.keys())}")
    img = df['frames'].astype(np.int16)
    
    if 'times' in df:
        times = df['times']
    else:
        times = np.empty(1)

    int_times = df['int_times']
    print(f"img.shape = {img.shape}")

    
    if bg is not None:
        bgf = np.load(bg)
        bgi = bgf['frames'].astype(np.int16)
        bgts = bgf['int_times']
        print(f"bg int_times = {bgts}")
        print(f"int_times = {int_times}")
        tt = einops.rearrange(bgi, 't h w -> t () h w')
        img = img - tt

    ee = Echelle(img[0,0], wlcal = wlcal)
    oms, lams = ee.order_mask()

    ords = []
    for i, om in enumerate(oms):
        r = full_prj(img, om, k)
        ords.append(r)

    return int_times, times, lams, ords, ee.lams, ee.ms


# fig, ax = plt.subplots(1,1)
# d = np.arange(45)
def degree_interp(dmax=200, dmin=6, mt=10):
    def f(d):
        d = np.asarray(d)
        r = np.zeros_like(d)
        r[d < mt] = dmax - d[d < mt]*(dmax-dmin)/mt
        r[d >= mt] = dmin
        return r
    return f

# ax.plot(d, deg()(d))
ref_radiance = functools.partial(planck_bb_wm2srum, T=1200)
# ref_radiance = lambda lam: np.ones_like(lam)*100

degint = degree_interp(200, 6, 12)
def response_fit(lam, ord, m):
    c = np.polynomial.chebyshev.Chebyshev.fit(lam, ord, degint(m))
    return c

def compute_overlap_vector(lams, eelams, a=0, b=0):
    overlap_vector = np.zeros_like(eelams)
    for lam in lams:
        l = lam[a:len(lam)-b]
        overlap_vector += (eelams >= l.min()) & (eelams <= l.max())
    return overlap_vector


def process_echelle(df, dfbg, ff, ffbg, wlcal, **kwargs):
    # load the data
    int_times, times, lams, ords, eelams, eems = process_run(df, wlcal, dfbg)

    # load flat field
    ff_int_times, ff_lams, ff_ords, eelams, eems = process_flat(ff, wlcal, ffbg)
    # compute spectral response function for each order

    if np.any(int_times != ff_int_times):
        raise ValueError("int_times don't match")

    print("int_times", int_times)

    if [l.shape for l in lams] != [l.shape for l in ff_lams]:
        raise ValueError("lams don't match")

    print("loaded orders")

   
    ff_fits = []
    
    for i, (lam, ord, ff_lam, ff_ord) in enumerate(zip(lams, ords, ff_lams, ff_ords)):
        sig = ff_ord[3]
        # rho = ref_radiance(ff_lam)/fit(ff_lam)
        fit = response_fit(ff_lam, sig, degint(i))
        ff_fits.append(fit)

    return int_times, times, lams, ords, ff_ords, ff_fits, eelams, eems

def combine_orders(int_times, lams, ords, ff_ords, ff_fits, eelams):
    # combine orders into single spectrum
    spectrum = np.zeros((ords[0].shape[0], ords[0].shape[1], eelams.shape[0]))

    print(f"spec shape {spectrum.shape} size = {spectrum.size * spectrum.itemsize / 1e6} MB")

    ref_radiance = functools.partial(planck_bb_wm2srum, T=1900)

    a, b = 20, 160

    overlap_vector = compute_overlap_vector(lams, eelams, a, b)

    for lam, ord, ff_ord, ff_fit in zip(lams, ords, ff_ords, ff_fits):
        # find index of lam into eelams
        # print(lam.shape, ord.shape, ff_ord.shape)
        ls = lam.shape[0]
        l = lam[a:ls-b]

        idx = np.argmin(np.abs(eelams - l[0]))
        nch = l.shape[0]

        rho = ref_radiance(l)/ff_fit(l)
        sig = ord[:, :, a:ls-b] * rho

        spectrum[:, :, idx:idx+nch] += sig
        
        
    spectrum /= overlap_vector

    return int_times, eelams, spectrum, overlap_vector