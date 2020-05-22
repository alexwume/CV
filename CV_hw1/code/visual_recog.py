import os, math, multiprocessing
from os.path import join
from copy import copy
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial

import visual_words
import util


def get_feature_from_wordmap(opts, wordmap):
    '''
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W) =
    [output]
    * hist: numpy.ndarray of shape (K)
    '''
    K = opts.K
    # ----- TODO -----

    bins=np.arange(opts.K + 1)
    hist,bin_edges=np.histogram (wordmap,bins,density=True)
    return hist

def blockshaped(a, p, q):
    p=round(p)
    q=round(q)
    m = a.shape[0]  #image row size
    n = a.shape[1]  #image column size

    # pad array with NaNs so it can be divided by p row-wise and by q column-wise
    bpr = round((m-1)//p + 1) #blocks per row
    bpc = round((n-1)//q + 1) #blocks per column
    M = round(p * bpr)
    N = round(q * bpc)


    A = np.nan* np.ones([M,N])
    A[:a.shape[0],:a.shape[1]] = a
    block_array = np.zeros((p, q ,bpr * bpc))
    previous_row = 0
    count = 0
    for row_block in range(bpc):
        previous_row = round(row_block * p)
        previous_column = 0
        for column_block in range(bpr):
            previous_column = round(column_block * q)
            block = A[previous_row:previous_row+p, previous_column:previous_column+q]
            if block.size:
                block_array[:,:, count]=block
            count +=1
    return block_array

def get_feature_from_wordmap_SPM(opts, wordmap):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^L-1)/3)
    '''
        
    K = opts.K
    L = opts.L
    # ----- TODO -----
    w = wordmap.shape[1]
    h = wordmap.shape[0]

    hist_all = np.array([])
    hist_one_layer = np.array([])
    #print('hist_one_layer shape',hist_one_layer.shape)
    #print('wordmap shape',wordmap.shape)
    wordmap_tmp = blockshaped(wordmap,((h-1) // (2 ** L)+1),((w-1) // (2 ** L)+1))
    #print('wordmap_tmp shape',wordmap_tmp.shape)
    layer = L
    for j in range(wordmap_tmp.shape[2]):
            hist_one_layer = np.append(hist_one_layer, get_feature_from_wordmap(opts, wordmap_tmp[:,:,j]))
    #print(hist_one_layer.shape)
    while (layer >= 0):
        if layer == 0 or layer == 1:
            hist_one_layer = hist_one_layer * math.pow(2,-L)
        else:
            hist_one_layer = hist_one_layer * math.pow(2, (layer-L-1))

        hist_all = np.insert(hist_all,0,hist_one_layer)
        #print('hist_all shape = ',hist_all.shape)
        #reform to a matrix
        hist_one_layer = hist_one_layer.reshape((2 ** layer ),(K * 2 ** layer))
        #print('hist_one_shape = ',hist_one_layer.shape)

        #merging adjacent rows, columns
        for i in range(hist_one_layer.shape[0]//2):
            hist_one_layer[:,2*i*K:2*i*K+K] += hist_one_layer[:,2*i*K+K:2*i*K+2*K]
            hist_one_layer[2*i,:] = hist_one_layer[2*i, :] + hist_one_layer [2*i +1, :]
        #delete even rows and columns
        for i in range(hist_one_layer.shape[0]//2):
            hist_one_layer = np.delete (hist_one_layer,i+1,0)
            hist_one_layer = np.delete (hist_one_layer,np.s_[i*K+K:i*K+2*K],1)
        hist_one_layer = hist_one_layer.reshape(-1)
        layer -=1


    hist_all = np.divide(hist_all,np.linalg.norm(hist_all,1))
    return hist_all

def get_image_feature(opts, img_path, dictionary):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K)
    '''

    # ----- TODO -----
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)/255
    img_shaped = np.zeros((img.shape[0],img.shape[1],3))
    #print('img.shape = ', img.shape)
    if len(img.shape) == 2:
        for i in range(2):
            img_shaped[:,:,i] = img
    else:
        img_shaped = img
    print('get to visual words')
    wordmap = visual_words.get_visual_words(opts, img_shaped, dictionary)
    print('get to image feature SPM')
    feature = get_feature_from_wordmap_SPM(opts, wordmap)
    #print('feature length',feature.shape)
    return feature

def build_recognition_system(opts, n_worker=1):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L
    K=opts.K

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
    dictionary = np.load(join(out_dir, 'dictionary.npy')) # fixed dictionary location!!

    # ----- TODO -----

    features = np.zeros((len(train_files),round(K * (4 ** (SPM_layer_num + 1) - 1) / 3)))

    print('start')
    pool = mp.Pool ()
    function_evaluate_train = partial(test_evaluation, opts,dictionary)
    features = pool.map(function_evaluate_train,train_files)
    print('end')
    #print('train features shape',features.shape)

    ## example code snippet to save the learned system
    np.savez_compressed(join(out_dir, 'trained_system.npz'),
        features=features,
        labels=train_labels,
        dictionary=dictionary,
        SPM_layer_num=SPM_layer_num,
    )

def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K(4^(L+1) -1)/3)
    * histograms: numpy.ndarray of shape (N,K(4^(L+1) -1)/3)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''

    # ----- TODO -----
    word_hist = np.tile(word_hist,(histograms.shape[0],1))
    intersection = np.minimum (word_hist,histograms)
    sim = np.sum(intersection, axis = 1)
    return sim

def test_evaluation(opts,dictionary,test_files):
    img_path = join(opts.data_dir,test_files)
    test_features = get_image_feature(opts, img_path, dictionary)
    return test_features

def evaluate_recognition_system(opts, n_worker=1):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, 'trained_system.npz'))
    dictionary = trained_system['dictionary']
    train_labels = trained_system['labels']

    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system['SPM_layer_num']
    test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)

    # ----- TODO -----
    #img_path = [None]*len(test_files)
    #test_features = np.zeros((len(test_files),round(test_opts.K * (4 ** (test_opts.L + 1) - 1) // 3)))
    fusion_matrix = np.zeros((8,8))

    pool = mp.Pool (mp.cpu_count())
    function_evaluate_test = partial(test_evaluation, opts,dictionary)
    test_features = pool.map(function_evaluate_test,test_files)

    for i in range(len(test_features)):
        tmp=distance_to_set(test_features[i],trained_system['features'])  # dimension : T (no. of training samples)
        compare_result_min_index = np.argmax(tmp)
        correspond_train_label = train_labels[int(compare_result_min_index)]
        true_label = test_labels[i]
        fusion_matrix[true_label, correspond_train_label] +=1


    accuracy = np.trace(fusion_matrix) / np.sum(fusion_matrix)

    return fusion_matrix, accuracy
