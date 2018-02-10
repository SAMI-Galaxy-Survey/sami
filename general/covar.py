import numpy as np

def create_covar_matrix_original(overlap_array,variances):
    """Create the covariance matrix for a single wavelength slice. 
        As input takes the output of the drizzle class, overlap_array, 
        and the variances of the individual fibres"""
    
    covarS = 2 # Radius of sub-region to record covariance information - probably
               # shouldn't be hard coded, but scaled to drop size in some way
    
    s = np.shape(overlap_array)
    if s[2] != len(variances):
        raise Exception('Length of variance array must be equal to the number of fibre overlap maps supplied')
    
    #Set up the covariance array
    covariance_array = np.zeros((s[0],s[1],(covarS*2)+1,(covarS*2)+1))
    if len(np.where(np.isfinite(variances) == True)[0]) == 0:
        return covariance_array
    
    #Set up coordinate arrays for the covariance sub-arrays
    xB = np.zeros(((covarS*2+1)**2),dtype=np.int)
    yB = np.zeros(((covarS*2+1)**2),dtype=np.int)
    for i in range(covarS*2+1):
        for j in range(covarS*2+1):
            xB[j+i*(covarS*2+1)] = i
            yB[j+i*(covarS*2+1)] = j
    xB = xB - covarS
    yB = yB - covarS
    
    #Pad overlap_array with covarS blank space in the spatial axis
    
    overlap_array_padded = np.zeros([s[0]+2*covarS,s[1]+2*covarS,s[2]])
    overlap_array_padded[covarS:-covarS,covarS:-covarS,:] = overlap_array
    overlap_array = overlap_array_padded

    #Loop over output pixels
    for xA in range(s[0]):
        for yA in range(s[1]):
            #Loop over each fibre
            for f in range(len(variances)):
                if np.isfinite(overlap_array[xA+covarS,yA+covarS,f]):
                    xC = xA +covarS + xB
                    yC = yA + covarS + yB
                    a = overlap_array[xA+covarS,yA+covarS,f]*np.sqrt(variances[f])
                    if np.isfinite(a) == False:
                        a = 1.0
                    #except:
                    #    code.interact(local=locals())

                    b = overlap_array[xC,yC,f]*np.sqrt(variances[f])
                    b[np.where(np.isfinite(b) == False)] = 0.0
                    covariance_array[xA,yA,:,:] = covariance_array[xA,yA,:,:] + (a*b).reshape(covarS*2+1,covarS*2+1)
            covariance_array[xA,yA,:,:] = covariance_array[xA,yA,:,:]/covariance_array[xA,yA,covarS,covarS]
    
    return covariance_array



def create_covar_matrix_vectorised(overlap_array,variances):
    """Create the covariance matrix for a single wavelength slice. 
        As input takes the output of the drizzle class, overlap_array, 
        and the variances of the individual fibres

    Dis been refactored by Francesco to add 0-th order vectorization.
    Reason is:
    1) old function `create_covar_matrix` (now `create_covar_matrix_original`)
       took >1000s/1400s of the cubing time.
    2) three for loops in python = three for loops because python is not a
       smart cookie in this respect
    So I removed one loop, but we could do more and save additional time.

    """

    covarS = 2 # Radius of sub-region to record covariance information - probably
               # shouldn't be hard coded, but scaled to drop size in some way
    
    s = np.shape(overlap_array)
    if s[2] != len(variances):
        raise Exception('Length of variance array must be equal to the number of fibre overlap maps supplied')
    
    #Set up the covariance array
    covariance_array = np.zeros((s[0],s[1],(covarS*2)+1,(covarS*2)+1))
    if len(np.where(np.isfinite(variances) == True)[0]) == 0:
        return covariance_array
    
    #Set up coordinate arrays for the covariance sub-arrays
    xB = np.zeros(((covarS*2+1)**2),dtype=np.int)
    yB = np.zeros(((covarS*2+1)**2),dtype=np.int)
    for i in range(covarS*2+1):
        for j in range(covarS*2+1):
            xB[j+i*(covarS*2+1)] = i
            yB[j+i*(covarS*2+1)] = j
    xB = xB - covarS
    yB = yB - covarS
    
    #Pad overlap_array with covarS blank space in the spatial axis
    
    overlap_array_padded = np.zeros([s[0]+2*covarS,s[1]+2*covarS,s[2]])
    overlap_array_padded[covarS:-covarS,covarS:-covarS,:] = overlap_array
    overlap_array = overlap_array_padded

    #Loop over output pixels
    for xA in range(s[0]):
        for yA in range(s[1]):
            valid = np.where(np.isfinite(overlap_array[xA+covarS,yA+covarS,:]))

            if len(valid[0])>0:

                xC = xA +covarS + xB
                yC = yA + covarS + yB
                a = overlap_array[xA+covarS,yA+covarS,valid[0]]*np.sqrt(variances[valid])

                a[np.where(~np.isfinite(a))] = 1.0

                b = overlap_array[xC,yC,:][:, valid[0]]*np.sqrt(variances[valid])

                b[np.where(np.isfinite(b) == False)] = 0.0

                ab = np.nansum(a*b, axis=1).reshape(covarS*2 + 1, covarS*2 + 1)

                covariance_array[xA,yA,:,:] += ab

            else: # This slice is useless.
                pass

            covariance_array[xA,yA,:,:] = covariance_array[xA,yA,:,:]/covariance_array[xA,yA,covarS,covarS]

    return covariance_array



def create_covar_matrix_newest(overlap_array,variances):
    """Create the covariance matrix for a single wavelength slice. 
        As input takes the output of the drizzle class, overlap_array, 
        and the variances of the individual fibres

    Dis been refactored by Francesco to add 0-th order vectorization.
    Reason is:
    1) old function `create_covar_matrix` (now `create_covar_matrix_original`)
       took >1000s/1400s of the cubing time.
    2) three for loops in python = three for loops because python is not a
       smart cookie in this respect
    So I removed one loop, but we could do more and save additional time.

    """

    raise ValueError('Still under development')
    
    covarS = 2 # Radius of sub-region to record covariance information - probably
               # shouldn't be hard coded, but scaled to drop size in some way

    # Size of the grid in the X-, Y- and fibres dimensions. `sx` and `sy` are
    # the dimensions of the SAMI spatial grid. Normally this is 50 x 50 spaxels.
    # `sf` is equal to the number of fibres in the current bundle times the
    # number of RSS files. This is typically 61 x 7 = 427.
    sx, sy, sf = np.shape(overlap_array)
    
    if sf != len(variances):
        raise ValueError('Length of variance array must be equal to the number of fibre overlap maps supplied')
    
    #Set up the covariance array
    covariance_array = np.zeros((sx,sy,(covarS*2)+1,(covarS*2)+1))
    if len(np.where(np.isfinite(variances) == True)[0]) == 0:
        return covariance_array
    
    #Set up coordinate arrays for the covariance sub-arrays
    xB = np.zeros(((covarS*2+1)**2),dtype=np.int)
    yB = np.zeros(((covarS*2+1)**2),dtype=np.int)
    for i in range(covarS*2+1):
        for j in range(covarS*2+1):
            xB[j+i*(covarS*2+1)] = i
            yB[j+i*(covarS*2+1)] = j
    xB -= covarS
    yB -= covarS
    xB = xB.reshape(covarS*2+1, covarS*2+1)
    yB = yB.reshape(covarS*2+1, covarS*2+1)
    
    #Pad overlap_array with covarS blank space in the spatial axis
    
    overlap_array_padded = np.zeros([sx+2*covarS,sy+2*covarS,sf])
    overlap_array_padded[covarS:-covarS,covarS:-covarS,:] = overlap_array
    overlap_array = overlap_array_padded

    xA = np.arange(sx, dtype=np.int)
    yA = np.arange(sy, dtype=np.int)

    xC = xA[:, None, None, None] + covarS + xB[None, None, :, :]
    yC = yA[None, :, None, None] + covarS + yB[None, None, :, :]

    a = overlap_array[xA[:, None] + covarS, yA[None, :] + covarS, :] # 50 x 50 x 427
    to_be_processed = np.where(np.isfinite(a))
    import itertools
    t2 = [[x, y] for x,y in zip(to_be_processed[0], to_be_processed[1])]
    t2 = np.array(list(k for k,_ in itertools.groupby(t2))).T
    a = a[t2[0], t2[1], :] * np.sqrt(variances)

    #a[to_be_processed][np.where(~np.isfinite(a[to_be_processed]))] = 1.0
    a[np.where(~np.isfinite(a))] = 1.0 # This new.

    fake_ax = np.arange(len(variances), dtype=np.int)

    #b = overlap_array[xC[:, :, None, :, :], yC[:, :, None, :, :], fake_ax[None, None, :, None, None]] * np.sqrt(variances[None, None, :, None, None]) # 50 x 50 x 427 x 25
    b = overlap_array[xC[:, :, None, :, :], yC[:, :, None, :, :], fake_ax[None, None, :, None, None]][t2[0], t2[1], :, :, :] * np.sqrt(variances[None, None, :, None, None]) # 50 x 50 x 427 x 25
    b[np.where(~np.isfinite(b))] = 0.0
    b = b.squeeze()

    # At this point need some sums.
    b *= a[:, :, None, None]

    b = np.nansum(b, axis=1)
    
    b /= b[:, covarS, covarS][:, None, None]

    covariance_array[t2[0], t2[1], :, :] = b
    return covariance_array


create_covar_matrix = create_covar_matrix_original
