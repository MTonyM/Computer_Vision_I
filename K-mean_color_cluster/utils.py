import matplotlib.pyplot as plt
try:
    import Image
except ImportError:
    from PIL import Image
from tqdm import tqdm
import numpy as np
import cv2


def find_k_features(photo,k):
    '''
    k-mean ++, input is an array
    '''
    l = photo.shape[0]
    k_feature = photo[np.random.choice(l,1)]
    fixed_norm = np.sum(np.square(photo),axis=1,keepdims=True)  # N x 3
    for i in tqdm(range(k-1),total=k-1):
        k_square_sum = np.sum(np.square(k_feature),axis=1,keepdims=False) # M x 3
        distance = ((fixed_norm + k_square_sum).T- 2*np.dot(k_feature,np.transpose(photo))) # M x N
        P = np.cumsum(np.min(distance,axis=0)/np.sum(np.min(distance,axis=0)))
        seed = np.random.random()
        new_k = photo[P>seed][0]
        k_feature = np.vstack((k_feature,new_k))
    return k_feature




def k_mean(photo,k,plus =True):
    if (plus == True):
        # Use the function to find best k_features
        # k mean ++
        k_feature = find_k_features(photo,k)
    else: 
        # find the feature pixel vector RANDOMLY.
        k_feature = photo[np.random.choice(photo.shape[0],k,replace=False)]
        
    last_loss = 0
    fixed_norm = np.sum(np.square(photo),axis=1,keepdims=True)  # N x 3
    for i in range(1000):
        k_square_sum = np.sum(np.square(k_feature),axis=1,keepdims=False) # M x 3
        distance = ((fixed_norm + k_square_sum).T- 2*np.dot(k_feature,np.transpose(photo))) # M x N
        
        ############################################################
        #   Vectorize progress, aspired from CS231n assignment 1 KNN Part.
        #   , makes this faster! =)
        ##########################################################
        min_loss = np.sum(np.min(distance,axis=0))
        if (abs(last_loss-min_loss) /min_loss < 0.01):
            # check if we converge.
            break
            
        last_loss = min_loss
        min_index = np.argmin(distance,axis = 0)

        # update each k feature to average.
        for i in range(k):
            k_feature[i] = np.average(photo[min_index == i],axis =0)
            
    # make picture with k colors
    for i in range(k):
        photo[min_index == i] = k_feature[i]    
    return photo
