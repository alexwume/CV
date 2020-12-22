import numpy as np
from PIL import Image
def alignChannels(red, green, blue):
    """Given 3 images corresponding to different channels of a color image,
    compute the best aligned result with minimum abberations

    Args:
      red, green, blue - each is a HxW matrix corresponding to an HxW image

    Returns:
      rgb_output - HxWx3 color image output, aligned as desired"""
    h=red.shape[0]
    w=red.shape[1]
    tmp=100000000000
    map_RG={}
    map_RB={}
    map_GB={}
    offset_max=30

    for i in range(-2*offset_max,2*offset_max+1): #for row
        for j in range(-2*offset_max,2*offset_max+1): #for column
            #Calculate SSD at each i,j
            if -offset_max<= i <=offset_max and -offset_max<=j<=offset_max:
                SSD_RG=ssd(red[max(i,0):min(h,h+i),max(j,0):min(w,w+j)],green[max(-i,0):min(h,h-i),max(-j,0):min(w,w-j)]) #red-green
                SSD_RB=ssd(red[max(i,0):min(h,h+i),max(j,0):min(w,w+j)],blue[max(-i,0):min(h,h-i),max(-j,0):min(w,w-j)]) #red-blue
                #store values
                map_RB[i,j] = SSD_RB
                map_RG[i,j] = SSD_RG
            SSD_GB = ssd(green[max(i,0):min(h,h+i), max(j, 0):min(w, w + j)], blue[max(-i, 0):min(h, h - i), max(-j, 0):min(w, w - j)])  # blue-green
            #store value and i,j in map
            map_GB[(i,j)]=SSD_GB
    #print(map_RG)
    for key1 in map_RG.keys():
        for key2 in map_RB.keys():
            #print(map_GB[-key1[0]+key2[0],-key1[1]+key2[1]])
            key3=(-key1[0]+key2[0],-key1[1]+key2[1])
            if map_RG[key1]+map_RB[key2]+map_GB[key3]<tmp:
                tmp=map_RG[key1]+map_RB[key2]+map_GB[key3]
                pos=[key1,key2]

    print(pos)

    #forming final image
    if pos[0][0]*pos[1][0]<0:
        f_h=h+abs(pos[0][0]-pos[1][0])
    else:
        f_h = h + max(abs(pos[0][0]),abs(pos[1][0]))

    if pos[0][1]*pos[1][1]<0:
        f_w=w+abs(pos[0][1]-pos[1][1])
    else:
        f_w = w + max(abs(pos[0][1]),abs(pos[1][1]))

    pos_g=pos[0]
    pos_b=pos[1]
    print(f_h,f_w)

    array=np.zeros([f_h,f_w,3],dtype=np.uint8)
    #determine x, y base:
    x_base=min(0,pos_g[0],pos_b[0])
    y_base=min(0,pos_g[1],pos_b[1])

    array[-x_base:-x_base+h,-y_base:-y_base+w,0]=red
    print(-x_base,-x_base+h)
    array[-x_base+pos_g[0]:-x_base+pos_g[0]+h,-y_base+pos_g[1]:-y_base+pos_g[1]+w,1]=green
    print(-x_base+pos_g[0],-y_base+pos_g[1])
    array[-x_base+pos_b[0]:-x_base+pos_b[0]+h,-y_base+pos_b[1]:-y_base+pos_b[1]+w,2]=blue
    print(-x_base+pos_b[0],-y_base+pos_b[1])


    return array

def ssd(A,B):
    h=A.shape[0]
    w=A.shape[1]
    squares = (A[:, :] - B[:,:]) ** 2

    return np.sum(squares)/(h*w)


