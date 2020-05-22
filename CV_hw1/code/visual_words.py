import os, multiprocessing
from os.path import join, isfile
from sklearn import cluster
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy.ndimage
import skimage.color
from random import shuffle
import multiprocessing as mp
from functools import partial


def extract_filter_responses(opts, img):
    '''
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''

    filter_scales = opts.filter_scales
    #filter_scales=[1,2,3]
    num_scales=len(filter_scales)
    num_filter=4;
    # ----- TODO -----
    height=img.shape[0]
    width=img.shape[1]
    img=skimage.color.rgb2lab(img)
    filter_responses=np.zeros((height,width,3*num_scales*num_filter))

    for j in range(num_scales):
        for i in range(img.shape[2]):
            filter_responses[:,:,i+j*3*num_filter]=scipy.ndimage.gaussian_filter(img[:,:,i],filter_scales[j])
            filter_responses[:,:,i+3+j*3*num_filter]=scipy.ndimage.gaussian_laplace(img[:,:,i],filter_scales[j])
            filter_responses[:,:,i+6+j*3*num_filter]=scipy.ndimage.gaussian_filter1d(img[:,:,i],sigma=filter_scales[j],order=1)
            filter_responses[:,:,i+9+j*3*num_filter]=scipy.ndimage.gaussian_filter1d(img[:,:,i],sigma=filter_scales[j],axis=0,order=1)

    return filter_responses


def compute_dictionary_one_image(opts,train_files):
    '''
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    '''

    # ----- TODO -----
    #print(img)
    img_path = join(opts.data_dir,train_files)
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)/255
    responses=extract_filter_responses(opts,img)
    random_responses = np.zeros((opts.alpha,3*4*len(opts.filter_scales)))
    for i in range(responses.shape[2]):
        random_responses[:,i]=np.random.choice(responses[:,:,i].reshape(-1),opts.alpha)

    return random_responses



def compute_dictionary(opts, n_worker=1):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel
    
    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    '''

    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K

    train_files =open(join(data_dir, 'train_files.txt')).read().splitlines()

    # ----- TODO -----
    img_path = [None] * len(train_files)
    train_files_tmp = np.zeros((opts.alpha*len(train_files),3*4*len(opts.filter_scales)))


    pool = mp.Pool (mp.cpu_count())
    functio = partial(compute_dictionary_one_image, opts)
    func = pool.map(functio,train_files)

    for i in range(len(func)):
        train_files_tmp[opts.alpha*i:opts.alpha*i+opts.alpha,:] = func[i]

    # for i in range(len(train_files)):
    #     img_path[i] = join(opts.data_dir,train_files[i])
    #     train_image = Image.open(img_path[i])
    #     train_image = np.array(train_image).astype(np.float32)/255
    #     train_files_tmp [opts.alpha*i:opts.alpha*(i+1),:] = compute_dictionary_one_image(opts,train_image)
    kmeans = cluster.KMeans(n_clusters=K).fit(train_files_tmp)

    #labels=kmeans.predict(filter_responses)
    dictionary = kmeans.cluster_centers_


    ## example code snippet to save the dictionary
    np.save(join(out_dir, 'dictionary.npy'), dictionary)

    # plt.scatter(dictionary[:, 0], dictionary[:, 1], c='black', cmap='rainbow')
    # plt.show()


def get_visual_words(opts, img, dictionary):
    '''
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''


    # ----- TODO -----
    wordmap = np.zeros((img.shape[0],img.shape[1]))
    wordmap_tmp = extract_filter_responses(opts,img)
    i=0
    for row in wordmap_tmp:
        dist = scipy.spatial.distance.cdist(row,dictionary,metric='euclidean')
        j=0
        for d in dist:
            word = np.argmin(d)
            wordmap[i,j] = word
            j+=1
        i +=1
    return wordmap





