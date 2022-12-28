# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 16:33:21 2022

@author: 14434
"""
"""
#explanations:
# levels should be 1-4 (1 save the most)    
"""
from main_functions_DALLE_HEBREW import *

#%% params
p = r'120542_masked.JPG'
opt = 1
level = 1
save_center = True
if save_center:
    level_center = 1
else:
    level_center = 0
    
noncenter_edge = 200
k_edge = True
edge_size = 5
# 1 - save the most
#%% Masks
kernel = np.ones((3,3))

#m1= plt.imread(p)

sizes = [256,1024]
size = 1024
bb = Image.open(p)
bb_array = np.array(bb)
bb_last = bb_array[:,:,-1]
some_alpha_high = 255
some_alpha_low = 0
for level in np.arange(1,7):
    for level_center in np.arange(5):
        for k_edge in [False,True]:
            if level > 3:
                bb_dilate = cv2.dilate(bb_last,kernel, iterations =level)
            else:
                bb_dilate = cv2.erode(bb_last,kernel, iterations =level)
            if level_center > 0:
                bb_dilate2 = cv2.erode(bb_last,kernel, iterations =2*level_center)
                bb_dilate[noncenter_edge :1024-noncenter_edge ,noncenter_edge :1024-noncenter_edge ] = bb_dilate2[noncenter_edge :1024-noncenter_edge ,noncenter_edge :1024-noncenter_edge ]
            bb_new = bb_array.copy()
            #bb_new[:,:,-1] = 255 - bb_dilate
            bb_new[bb_new != 0]            = 255
            bb_dilate[bb_dilate > 1] = some_alpha_high
            bb_dilate[bb_dilate <= 1] = some_alpha_low
            if opt == 1:
                
                bb_new[:,:,-1] = 255-bb_dilate.astype(np.uint8)
            else:
                bb_new[:,:,-1] =  bb_dilate #(255-bb_dilate).astype(np.uint8)
            if k_edge:
                mask_edge = np.ones(bb_dilate.shape)
                mask_edge[edge_size :1024-edge_size ,edge_size :1024-edge_size ]  = 0
                
                if opt == 1:
                    bb_dilate[mask_edge == 1] = 0
                    bb_new[:,:,-1] = 255-bb_dilate.astype(np.uint8)
                else:
                    bb_dilate[mask_edge == 1] = 255
                    bb_new[:,:,-1] =  bb_dilate #(255-bb_dilate).astype(np.uint8)            
                
                
            
            m1 = bb_new
            
            
            maskA = Image.fromarray(m1.astype(np.uint8)).convert('RGBA')
            maskA.save(r'masks/mask_A_l%d_lc_%d_%s_%d.png'%(level, level_center,str(k_edge),size))

            m2 = m1[::-1, :, :]
            maskB = Image.fromarray(m2.astype(np.uint8)).convert('RGBA')
            maskB.save(r'masks/mask_B_l%d_lc_%d_%s_%d.png'%(level, level_center,str(k_edge),size))
            m3 = m1[:, ::-1, :]
            maskC = Image.fromarray(m3.astype(np.uint8)).convert('RGBA')
            maskC.save(r'masks/mask_C_l%d_lc_%d_%s_%d.png'%(level, level_center,str(k_edge),size))
            m4 = m1[::-1, ::-1, :]
            maskD = Image.fromarray(m4.astype(np.uint8)).convert('RGBA')
            maskD.save(r'masks/mask_D_l%d_lc_%d_%s_%d.png'%(level, level_center,str(k_edge),size))
        bb_dilate_o = bb_dilate.copy()
        for size in sizes:
            if size != 1024:
                each_size = int((1024-size)/2)
                bb_dilate = bb_dilate_o.copy()
                for k_edge in [False,True]:
                    if k_edge:
                        mask_edge = np.ones(bb_dilate.shape)
                        mask_edge[edge_size + each_size :1024-edge_size-each_size ,edge_size + each_size :1024-edge_size - each_size ]  = 0
                        
                        if opt == 1:
                            bb_dilate[mask_edge == 1] = 0
                            bb_new[:,:,-1] = 255-bb_dilate.astype(np.uint8)
                        else:
                            bb_dilate[mask_edge == 1] = 255
                            bb_new[:,:,-1] =  bb_dilate #(255-bb_dilate).astype(np.uint8)            
                        
                        
                    
                    m1 = bb_new
                    maskA = Image.fromarray(m1.astype(np.uint8)).convert('RGBA')


                    m2 = m1[::-1, :, :]
                    maskB = Image.fromarray(m2.astype(np.uint8)).convert('RGBA')
                    m3 = m1[:, ::-1, :]
                    maskC = Image.fromarray(m3.astype(np.uint8)).convert('RGBA')

                    m4 = m1[::-1, ::-1, :]
                    maskD = Image.fromarray(m4.astype(np.uint8)).convert('RGBA')

                    left = each_size; top = each_size; right = 1024 - each_size; bottom = 1024 - each_size
                    maskA.crop((left, top, right, bottom)).save(r'masks/mask_A_l%d_lc_%d_%s_%d.png'%(level, level_center,str(k_edge),size))
                    maskB.crop((left, top, right, bottom)).save(r'masks/mask_B_l%d_lc_%d_%s_%d.png'%(level, level_center,str(k_edge),size))
                    maskC.crop((left, top, right, bottom)).save(r'masks/mask_C_l%d_lc_%d_%s_%d.png'%(level, level_center,str(k_edge),size))
                    maskD.crop((left, top, right, bottom)).save(r'masks/mask_D_l%d_lc_%d_%s_%d.png'%(level, level_center,str(k_edge),size))
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
#%% Create edge masks

for edge_size in [1,2,5,10,20,40]:
    bb_dilate = 255*np.ones((256,256))
    bb_dilate[edge_size:256-edge_size,edge_size:256-edge_size] = 0
    bb_new = 255*np.ones((256,256,4))
    bb_new[:,:,-1] = bb_dilate
    
    m1 = bb_new
    
    maskA = Image.fromarray(m1.astype(np.uint8)).convert('RGBA')
    
    maskA.save(r'masks/edge_%d_%d.png'%(256, edge_size))
    
for edge_size in [1,2,5,10,20,40]:
    bb_dilate = 255*np.ones((1024,1024))
    bb_dilate[edge_size:1024-edge_size,edge_size:1024-edge_size] = 0
    bb_new = 255*np.ones((1024,1024,4))
    bb_new[:,:,-1] = bb_dilate

    m1 = bb_new

    maskA = Image.fromarray(m1.astype(np.uint8)).convert('RGBA')

    maskA.save(r'masks/edge_%d_%d.png'%(1024, edge_size))







#%% Create dits masks
for edge_size in  [1,2,5,10,20,40]:
    for image_s in sizes:
        for k_edge in [True, False]                    :
            if k_edge:
                st = edge_size
                en = image_s-edge_size
            else:
                st = 0
                en = image_s
            bb_dilate = 255*np.ones((image_s,image_s))
                
                #bb_dilate[edge_size:image_s-edge_size,edge_size:image_s-edge_size] = 0
            #else:
            #    bb_dilate = np.zeros((image_s,image_s))
            
            
            for row in np.arange(st,en, 4):
               
                for col in np.arange(st,en, 4):
                    bb_dilate[row,col] = 0
            mask = np.dstack([bb_dilate]*4)
            mask = Image.fromarray(mask.astype(np.uint8)).convert('RGBA')
            mask.save(r'masks/mask_dots_%s_%d_%d.png'%(str(k_edge), image_s, edge_size))
        

#%% Create dits masks
for edge_size in  [1,2,5,10,20,40]:
    for image_s in sizes:
        for k_edge in [True, False]                    :
            if k_edge:
                st = edge_size
                en = image_s-edge_size
            else:
                st = 0
                en = image_s
            bb_dilate = 255*np.ones((image_s,image_s))
                
                #bb_dilate[edge_size:image_s-edge_size,edge_size:image_s-edge_size] = 0
            #else:
            #    bb_dilate = np.zeros((image_s,image_s))
            
            
            for row in np.arange(st,en, 4):
               
                for col in np.arange(st,en, 4):
                    bb_dilate[row,col] = 0
            mask = np.dstack([bb_dilate]*4)
            mask = Image.fromarray(mask.astype(np.uint8)).convert('RGBA')
            mask.save(r'masks/mask_dots_%s_%d_%d.png'%(str(k_edge), image_s, edge_size))


#%% Create dits masks
for edge_size in  [1,2,5,10,20,40]:
    for image_s in sizes:
        for k_edge in [True, False]                    :
            if k_edge:
                st = edge_size
                en = image_s-edge_size
            else:
                st = 0
                en = image_s
            bb_dilate = 255*np.ones((image_s,image_s))

                #bb_dilate[edge_size:image_s-edge_size,edge_size:image_s-edge_size] = 0
            #else:
            #    bb_dilate = np.zeros((image_s,image_s))


            for row in np.arange(st,en-4, 10):

                for col in np.arange(st,en-4, 10):
                    bb_dilate[row:row+6,col:col+6] = 0
            mask = np.dstack([bb_dilate]*4)
            mask = Image.fromarray(mask.astype(np.uint8)).convert('RGBA')
            mask.save(r'masks/mask_dots_%s_%d_%dr.png'%(str(k_edge), image_s, edge_size))
     





















        