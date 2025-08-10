import numpy as np
from scipy.ndimage import gaussian_filter


# NEW METHODS (diff from previous notebook)

Na = 2 # dimensionality of actions (axis, direction)

# define action book (binary -1/1)
# each action has axis and direction in the two dimensions
# at the last path location, the action is 0,0 so that it stays there (stop signal)
# def actions(path_locations):
#     Npatts = len(path_locations)
#     abook = np.zeros((Na, Npatts))

#     velocity = [(path_locations[idx + 1][0] - path_locations[idx][0], path_locations[idx+1][1] - path_locations[idx][1]) 
#           for idx in range(len(path_locations) - 1)]

#     k = 0
#     for i in velocity:
#         if i[0] == 0:
#             abook[0,k] = 1                 #vertical axis
#             abook[1,k] = i[1]    #vertical mapping is the same as in Wgg for flattened but reverse for indexed
#         elif i[1] == 0:
#             abook[0,k] = -1        #horizontal axis (-1 easier to learn than 0)
#             abook[1,k] = i[0]     #horizontal mapping is the same as in Wgg    
#         k = k+1  
        
#     return abook



# #one-hot action book in a 4d vector
# def actions(path_locations):
#     Npatts = len(path_locations)
#     abook = np.zeros((4, Npatts))

#     velocity = [(path_locations[idx + 1][0] - path_locations[idx][0], path_locations[idx+1][1] - path_locations[idx][1]) 
#           for idx in range(len(path_locations) - 1)]

#     k = 0
#     for i in velocity:
#         if i[0] == 0 and i[1] == 1:
#             abook[2,k] = 1             # Up      
#         elif i[0] == 0 and i[1] == -1: 
#             abook[3,k] = 1             # Down
#         elif i[0] == 1 and i[1] == 0: 
#             abook[0,k] = 1             # Right
#         elif i[0] == -1 and i[1] == 0: 
#             abook[1,k] = 1             # Left
#         k = k+1  
        
#     return abook


#1d action book with different labels for six actions
def actions(path_locations):
    Npatts = len(path_locations)
    abook = np.zeros((Npatts))

    velocity = [(path_locations[idx + 1][0] - path_locations[idx][0], path_locations[idx+1][1] - path_locations[idx][1]) 
          for idx in range(len(path_locations) - 1)]

    k = 0
    """
    In fact, actions five and six will never be triggered in the case of a serpentine path.
    """
    for i in velocity:
        if i[0] == -1 and i[1] == 1:
            abook[k] = 6             # Up-Left
        elif i[0] == 1 and i[1] == -1:
            abook[k] = 5             # Down-Right 
        elif i[0] == 0 and i[1] == 1:
            abook[k] = 3             # Up      
        elif i[0] == 0 and i[1] == -1: 
            abook[k] = 4             # Down
        elif i[0] == 1 and i[1] == 0: 
            abook[k] = 1             # Right
        elif i[0] == -1 and i[1] == 0: 
            abook[k] = 2             # Left
        k = k+1  
        
    return abook




def oneDaction_mapping(action):
    if action == 0:
        axis = None
        direction = None
    if action == 1:       # Right
        axis = 0
        direction = 1
    elif action == 2:    # Left
        axis = 0
        direction = -1
    elif action == 3:    # Up
        axis = 1
        direction = 1 
    elif action == 4:    # Down
        axis = 1
        direction = -1   
    elif action == 5:    # Down-Right
        axis = [1,0]
        direction = [-1,1] 
    elif action == 6:    # Up-Left
        axis = [1,0]
        direction = [1,-1]          
        
    return axis, direction



def onehot_mapping(idx):
    if idx == 0:       # Right
        axis = 0
        direction = 1
    elif idx == 1:    # Left
        axis = 0
        direction = -1
    elif idx == 2:    # Up
        axis = 1
        direction = 1 
    elif idx == 3:    # Down
        axis = 1
        direction = -1
        
    return axis, direction



def path_codes(path_locations, pbook, sbook, nruns=1):
    Npatts = len(path_locations)
    Ns = sbook.shape[0]
    Np = pbook.shape[1]
    path_pbook = np.zeros((nruns, Np, Npatts))
    path_sbook = np.zeros((Ns, Npatts))

    k = 0
    for i in path_locations:
        path_pbook[:,:,k] = pbook[:,:,i[0],i[1]]
        path_sbook[:,k] = sbook[:,i[0],i[1]]
        k = k+1
   
    return path_pbook, path_sbook


# learns Wsp using psuedoinverse learning
def actionmap(abook, path_pbook):
    pbookinv = np.linalg.pinv(path_pbook)
    Wap = abook@pbookinv
    return Wap   #nruns,Na,Np where Na = 2 (axis and direction)



def sensorymap(path_sbook, path_pbook):
    pbookinv = np.linalg.pinv(path_pbook)
    Wsp = path_sbook@pbookinv
    return Wsp


def palacemap(palace_sbook, path_sbook_recon):
    sbookinv = np.linalg.pinv(path_sbook_recon)
    Wss_r = palace_sbook@sbookinv
    return Wss_r


# global nearest neighbor
def nearest_neighbor_idx(gin, gbook):
    est = np.einsum('ijk, jl -> ikl', gin, gbook)
    maxm = np.amax(est, axis=2)  #(nruns,1)
    idx_lst = np.zeros((len(maxm)))
    for r in range(len(maxm)):
        a = np.argwhere(est[r] == maxm[r])
        idx = np.random.choice(a[:,1])
        idx_lst[r] = idx
    return idx_lst



# 2d coordinates to flattened grid code
def corrd_to_gcode(path_locations, Npos): 
    """
    path_locations: list of (x,y) tuples
    gcode: list of grid code vectors
    """
    path_locations_flattened = [x*Npos+y for x, y in path_locations]
    gcode = gbook_flattened[:,path_locations_flattened]
    return gcode



# flattened grid code to 2d coordinates
def gcode_to_coord(gin, gbook_flattened, Npos):
    """
    Calculate the similarity (generally the inner product) between all columns of gin and gbook_flattened. 
    Find the closest column index idx: that is, the most similar spatial position. 
    Then, reverse the idx into coordinates:
    """
    nrun=0
    idx = nearest_neighbor_idx(gin, gbook_flattened)[nrun]
    x = idx//Npos
    y = idx - x*Npos
    return (int(x),int(y))


def relu(x, thresh=0):
    return x * (x > thresh)

def softplus(x, thresh=0):
    return np.log(1+np.exp(x-thresh))

def nonlin(x, thresh=2.5):
    #return relu(x, 0)
    return relu(x-thresh, 0)
    #return softplus(x, thresh=thresh)
    #return np.sign(x)
    
    
def sens_nonlin(x):
    #return relu(x, 0) 
    return np.sign(x)   


# MAPPING TO A HEXAGONAL GRID
def mapRealtoHex(hexgbook_map, path_locations):
    hex_path_locations = []

    for i in range(len(path_locations)):
        hex_path_locations.append((hexgbook_map[0,path_locations[i][0],path_locations[i][1]]+path_locations[i][1]//2, hexgbook_map[1,path_locations[i][0],path_locations[i][1]]))     

    return hex_path_locations


# smoothening the grid fields by upsampling for high resolution
def smooth_tuningcurve(avg_fields, Npos, mult=2, path=False, path_locations=None):
    avg_fields_sq = avg_fields.reshape((Npos,Npos))
    if path:
        afs = np.zeros_like(avg_fields_sq)
        afs[:] = np.nan
        afs[path_locations[:,0], path_locations[:,1]] = avg_fields_sq[path_locations[:,0], path_locations[:,1]]
    else:
        afs = np.copy(avg_fields_sq)
    afs = afs.T
    afs2 = upsample(afs,mult)
    hexed_afs = np.copy(afs2)

    for i in range(Npos):
        hexed_afs[mult*i:mult*(i+1)] = np.roll(hexed_afs[mult*i:mult*(i+1)],i) 
    
    return hexed_afs
    
      
def upsample(im, mult=2):
    height, width = np.shape(im)
    im_up = np.zeros((mult * height, mult * width))

    for i in range(height):
        for j in range(width):
            im_up[mult * i: mult * (i + 1), mult * j: mult * (j +1)] = im[i, j]
            
    return im_up


# Explicit interpolation with more upsampling for round smooth grid fields
def explicit_interpolation(hexed_afs, upsample_rate=10, sigma=10):
    hexed_afs_up = upsample(hexed_afs, upsample_rate)

    # doesn't deal with nans when plotting for paths
    #hexed_afs_up_smooth = gaussian_filter(hexed_afs_up, sigma, mode='wrap')

    # deals with nans when plotting for paths but is super slow
    # from astropy.convolution import convolve_fft as asconvolve
    # from astropy.convolution import Gaussian2DKernel
    # kernel = Gaussian2DKernel(x_stddev=sigma,y_stddev=sigma)
    # hexed_afs_up_smooth = asconvolve(hexed_afs_up,kernel,boundary='wrap')


    U=hexed_afs_up.copy()
    U[np.isnan(hexed_afs_up)]=0
    UU=gaussian_filter(U,sigma=sigma)

    W=0*hexed_afs_up.copy()+1
    W[np.isnan(hexed_afs_up)]=0
    WW=gaussian_filter(W,sigma=sigma)

    hexed_afs_up_smooth=UU/WW

    return hexed_afs_up_smooth


# smoothening the grid fields along the three axis on the discrete hexagonal space
def twoD_Gaussian(xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g