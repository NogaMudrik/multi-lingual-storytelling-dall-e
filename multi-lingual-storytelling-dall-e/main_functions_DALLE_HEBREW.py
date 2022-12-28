# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 05:10:15 2022

@author: noga mudrik
"""
#https://epapers2.org/ciss2023/ESR/paper_submit.php
# Imports the Google Cloud Translation library
# dalle discord https://discord.com/channels/974519864045756446/990336929142808656
"""
Imports
"""

from datetime import datetime
from google.oauth2 import service_account
import os
import openai
import io, base64
from PIL import Image, ImageDraw,ImageEnhance
from google.cloud import translate
#from images2gif import writeGif
import numpy as np
import requests
import time
from os.path import exists
global image_s, masks, k_edge, keep_center, edge_size, keep_center, reduce_sat, rough,elipses_center_x ,elipses_center_y

import colorsys
import glob

import matplotlib.pyplot as plt
import keras_ocr
import cv2


    
#%% Global Inputs
reduce_sat = True
masks = ['mask_%s'%letter for letter in ['A','B','C','D']]
image_s =  256 #int(input('image_s')) # 256, 512, 1024
rough = ''
keep_center = True # str2bool(input('keep center'))
if keep_center:
    center_size = int(image_s/4)
    take_side = int((image_s  - center_size)/2)
k_edge = True # str2bool(input('k_edge?'))
loc_defined = False

basic_path = os.getcwd()
loc_defined = True

# google translate and openai keys
path_json = input('type the location of the translation json file (empty for current directory)')
if len(path_json) == 0:
     path_json =     basic_path
json_path =  path_json + os.sep + 'translate.json'
credentials = service_account.Credentials.from_service_account_file(json_path)
openai.api_key = input('please type your openai key')


text_path =basic_path + os.sep +'text_files'
path_initial = basic_path + os.sep + 'example_image_to_start'
path_images = basic_path + os.sep + 'images'




#%% Functions

"""
Functions
"""

def str2bool(str_to_change):
    """
    Transform 'true' or 'yes' to True boolean variable 
    Example:
        str2bool('true') - > True
    """
    if isinstance(str_to_change, str):
        str_to_change = (str_to_change.lower()  == 'true') or (str_to_change.lower()  == 'yes')  or (str_to_change.lower()  == 't')
    return str_to_change 



def make_dots(path_save_res, dots_T = 4, edge_size = 20, image_s = image_s, k_edge = k_edge, to_return = False, to_save = True):
    if k_edge:
        st = edge_size
        en = image_s-edge_size
    else:
        st = 0
        en = image_s
    bb_dilate = 255*np.ones((image_s,image_s))
   
    
    for row in np.arange(st,en, dots_T):
       
        for col in np.arange(st,en, dots_T):
            bb_dilate[row,col] = 0

    mask = Image.fromarray(bb_dilate.astype(np.uint8))
    if to_save:
        mask.save(path_save_res + os.sep + 'base_mask.png')
    if to_return:
        return bb_dilate
    
    
def txt2str(file_name, text_path = text_path, replace_n = True, replace_undesired = True,
            undesired = '";<>}{@#$%^&*~-[]@#$%^&|*()-?!,‘.’', is_file = True, cut_by_line = False, cut_by_dot = False):
    if is_file:
        if not file_name.endswith('.txt') and  not file_name.endswith('.html'): file_name = file_name + '.txt'
        f = open(text_path + os.sep + file_name , "r", encoding="utf8"); 
        tt = f.read()
    else:
        tt = file_name
    if replace_n and not cut_by_line:
        tt = tt.replace('\n', ' ')
    if cut_by_line:
        tt = tt.replace('\n', 'kkkkkk')
    if cut_by_dot:
        tt = tt.replace('.', 'kkkkkk')
    if replace_undesired:
        for un in undesired:
            tt =  tt.replace(un, ' ')
        tt = tt.replace("'", " ")
    return tt
    
    
def create_mask_det_edge(img, path_save_res):
    """
    Parameters
    ----------
    img : Image or numpy array
    path_save_res : path to save. string

    Returns
    -------
    ddd : Image
        Dthe returned corrected image

    """
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    if np.max(img) > 0:
        img = (255*img/np.max(img)).astype(np.uint8)
    ddd = Image.fromarray(img.astype(np.uint8))
    ddd = check_mask(ddd)
    ddd.save(path_save_res+ os.sep +'mask5.png')
    ddd.save(path_save_res+ os.sep +'mask5_original.png')
    return ddd
    
    
def create_mask(seed, p = 0, z = 0, y = 0, x =0, level_change = 1, multi = False, n_multi = 4, diffuse = True ,
                level =1, level_center = 1,path_save_res ='', object_det_percent = 30, object_det = False, counter = -1,
                object_det_veto  = True,dots_T  = 4, disable_directions=[], control_eli = False, quarter = False, edge_size = 20, 
                elipses_center_x = [],  elipses_center_y =  [], dots = True):
    """
        Returns
    -------
    ddd : TYPE
        x - 170-370
        y - 120- 320
        z - 20-100
        p - 50-200
    [x0, y0, x1, y1].
    """
    np.random.seed(seed)
    if diffuse and level == 0 and (not dots) and (not k_edge):
        print('you did not define a level, are you sure you want diffuse?')
        
        
    if (diffuse or k_edge or dots) and not (object_det_veto and object_det):
        if diffuse and level > 0:
            cur_mask = np.random.choice(masks)
            cur_mask = r'masks/' + cur_mask + '_l%d_lc_%d_%s_%d.png'%(level, level_center, str(k_edge), image_s)
        elif level == 0 and dots:
            make_dots(path_save_res, dots_T = dots_T , edge_size = edge_size, image_s = image_s, k_edge = k_edge )
            cur_mask = path_save_res + os.sep + 'base_mask.png'
            #cur_mask =  r'masks/mask_dots_%s_%d_%d%s.png'%(str(k_edge), image_s, edge_size,str(rough))
        elif level == 0 and (not dots) and k_edge:
            cur_mask =  r'masks/edge_%d_%d.png'%(image_s, edge_size)           

        else:
            

            cur_mask =  r'masks/zeros.png'  
            
        ddd = Image.open(cur_mask)
        plt.imshow(ddd)
        plt.title('mask')
        ddd = check_mask(ddd)
        ddd.save(path_save_res+ os.sep +'mask5.png')
            
    else:
        if object_det and object_det_veto:
            ddd = np.zeros((image_s ,image_s )).astype(np.uint8)
            ddd = Image.fromarray(ddd).convert('L')
            ddd = check_mask(ddd)
            ddd.save(path_save_res+ os.sep +'mask5.png')
        else:
            ddd = (np.random.rand(image_s ,image_s )*255).astype(np.uint8)
            ddd = Image.fromarray(ddd).convert('L')
            ddd = check_mask(ddd)
            ddd.save(path_save_res+ os.sep +'mask5.png')

            
    if level_change > 0:
        if quarter:
            mean_val = np.int(image_s/2)
            loc_eli = np.mod(counter,len(elipses_center_x))
            x1 = elipses_center_x[loc_eli]
            y1 = elipses_center_y[loc_eli]

            if dots:
                ddd_ar = np.array(ddd)
            else:
                ddd_ar = np.zeros((image_s,image_s, 4))+255
            if x1 > 0.5 and y1 > 0.5:
                ddd_ar[int(0.5*image_s):,int(0.5*image_s):,: ] = 0
            if x1 > 0.5 and y1 < 0.5:
                ddd_ar[int(0.5*image_s):,:int(0.5*image_s),: ] = 0    
            if x1 < 0.5 and y1 < 0.5:
                ddd_ar[:int(0.5*image_s),:int(0.5*image_s),: ] = 0    
            if x1 < 0.5 and y1 > 0.5:
                ddd_ar[:int(0.5*image_s),int(0.5*image_s):,: ] = 0
            ddd = check_mask(ddd_ar)
            display(ddd)

            ddd.save(path_save_res+ os.sep +'mask5.png')                
        elif not multi:
            sizex = np.random.randint(90,280)*level_change
            sizey = np.random.randint(90,280)*level_change
            #x1 = np.random.randint(0, image_s) #- sizex/2
            #y1= np.random.randint(0, image_s )# - sizey
            if control_eli:
                loc_eli = np.mod(counter,len(elipses_center_x))
                x1 = int(image_s*elipses_center_x[loc_eli])
                y1 = int(image_s*elipses_center_y[loc_eli])
                
            else:
                
                if keep_center:
                    x1 = np.random.choice([np.random.randint(0, take_side),np.random.randint(image_s - take_side, image_s)]) #- sizex/2
                    y1= np.random.choice([np.random.randint(0, take_side),np.random.randint(image_s - take_side, image_s)]) # - sizey
                    #x1 = np.hstack([x1[:take_side],x1[image_s - take_side:]] ) 
                else:
                    x1 = np.random.randint(0, image_s) #- sizex/2
                    y1= np.random.randint(0, image_s )# - sizey

            
        
            draw = ImageDraw.Draw(ddd)
            draw.ellipse((x1-int(sizex/2), y1-int(sizey/2), x1+int(sizex/2), y1+int(sizey/2)), fill=255)
            #ddd.save('mask4.png')    
            ddd = ddd.convert('RGBA')    
            draw = ImageDraw.Draw(ddd)
            draw.ellipse((x1-int(sizex/2), y1-int(sizey/2), x1+int(sizex/2), y1+int(sizey/2)), fill=255)
            ddd = check_mask(ddd)
            ddd.save(path_save_res+ os.sep +'mask5.png')
            
        else: #multi

            ddd = ddd.convert('RGBA')                
            for num in range(n_multi):
                sizex = np.random.randint(90,280)*level_change/n_multi
                sizey = np.random.randint(90,280)*level_change/n_multi
                #optsx =  np.random.randint(0, image_s ) 
                

                if keep_center:
                    x1 = np.random.choice([np.random.randint(0, take_side),np.random.randint(image_s - take_side, image_s)]) #- sizex/2
                    y1= np.random.choice([np.random.randint(0, take_side),np.random.randint(image_s - take_side, image_s)]) # - sizey
                    #x1 = np.hstack([x1[:take_side],x1[image_s - take_side:]] ) 
                else:
                    x1 = np.random.randint(0, image_s) #- sizex/2
                    y1= np.random.randint(0, image_s )# - sizey

                draw = ImageDraw.Draw(ddd)
                draw.ellipse((x1-int(sizex/2), y1-int(sizey/2), x1+int(sizex/2), y1+int(sizey/2)), fill=255)
            ddd = check_mask(ddd) 
            ddd.save(path_save_res+ os.sep +'mask5.png')
    if object_det:
        if counter < 0:
            raise ValueError('if "object_det" you must provide counter!')
        if counter == 0:
            raise ValueError('You should not have arrived here!')
            
        elif counter > 0:
            #print('hereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')
            rows_cols = np.load(path_save_res+ os.sep +'rows_cols.npy', allow_pickle=True).item()
            rows_obj = rows_cols['rows_obj']
            cols_obj = rows_cols['cols_obj']
        ddd_array = np.array(ddd)
        plt.imshow(ddd_array)
        plt.title('ddd_array')

        if len(ddd_array.shape) == 3:
            mask1 = keep_object_in_mask(rows_obj, cols_obj, ddd_array[:,:,-1], percentile = object_det_percent, disable_directions=disable_directions)
            h = plt.imshow(mask1)
            plt.colorbar(h)
            #mask1 = np.repeat( mask1.reshape(( mask1.shape[0], mask1.shape[1], 1)),ddd_array.shape[2], axis = 2)
        else:
            mask1 = keep_object_in_mask(rows_obj, cols_obj, ddd_array, percentile = object_det_percent, disable_directions=disable_directions)
        mask1 = check_mask(mask1)  
        mask1.save(path_save_res+ os.sep +'mask5.png')
        if counter == 1:
            mask1.save(path_save_res+ os.sep +'mask5_first.png')
   
            
    
    
    return ddd

def check_mask(mask1)    :
    """
    check mask
    """
    if isinstance(mask1, str):
        mask1_image = Image.open(mask1)
        
        shape_or = np.array(mask1_image).shape
        print(shape_or)
        if len(shape_or ) <3 or shape_or[-1] < 4: 
            mask1_image = check_mask(mask1_image)
            mask1_image.save(mask1)
    else:
        mask_np = np.array(mask1)    
        
        if len(mask_np.shape ) <3 or mask_np.shape[-1] < 4: 
            if len(mask_np.shape ) <3 :
                mask_np = mask_np.reshape((mask_np.shape[0], mask_np.shape[1], 1))
            
    
            mask_rep = np.repeat(np.array(mask_np).reshape((mask_np.shape[0],mask_np.shape[1],1)), 4, axis = 2)
            mask1 = Image.fromarray(mask_rep.astype(np.uint8)).convert('RGBA')   
        else:
            mask1 = Image.fromarray(mask_np.astype(np.uint8)).convert('RGBA')
        return mask1
    
# Initialize Translation client
def translate_text(text="YOUR_TEXT_TO_TRANSLATE", project_id="sylvan-task-368008", credentials= credentials,
                   original_lang = "he"):
    """Translating Text.look for samples to ref"""

    client = translate.TranslationServiceClient(credentials=credentials)

    location = "global"

    parent = f"projects/{project_id}/locations/{location}"

    # Translate text from English to French
    # Detail on supported types can be found here:
    # https://cloud.google.com/translate/docs/supported-formats
    # link to language codes https://cloud.google.com/translate/docs/languages
    # (hebrew - he, german - de, russian - ru , chinese - zh)
    response = client.translate_text(
        request={
            "parent": parent,
            "contents": [text],
            "mime_type": "text/plain",  # mime types: text/plain, text/html
            "source_language_code": original_lang,
            "target_language_code": "en-US",
        }
    )

    # Display the translation for each input text provided
    #for translation in response.translations:
    translation = response.translations[0]
    #    print("Translated text: {}".format(translation.translated_text))
    return translation.translated_text


def create_dalle_response(prompt, size = "%dx%d"%(image_s,image_s), form_return = 'b64_json', cont = False, mask_path = "mask5.png", 
                          img_path = '', saving_add='', n=1, multi =False, n_multi = 4, diffuse = True, level =1, level_center = 1,
                         path_save_res='' ):
    
  """
  a white siamese cat
  """
  if cont and (len(mask_path) == 0 or len(img_path) == 0):
      #print(len(mask_path))
      #print(len(img_path))
      raise ValueError('if cont, then mask_path and img_path should be provided')
  if cont:
      print('prompt real image to dalle!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
      print(prompt)
      print(img_path)
      #print('cont')
      #print('open')
      #print(img_path)
      if not img_path.endswith('.png'): img_path = img_path +saving_add + '.png'
      #print(img_path)
      #elif not img_path.endswith(saving_add): img_path
      img_path = img_path.replace("'","")
      #img_path = img_path.replace(":","")
      if multi and not diffuse:
          name_mask = path_save_res+ os.sep + "mask5.png"
      else:
          name_mask =path_save_res+ os.sep + "mask5.png"
      im = plt.imread(name_mask)    
      plt.imshow(im)
      plt.show()
      try:
          Image.open(img_path)
      except:
          try:
              #img_path_sl  = img_path.split(os.sep)#[-1]
              #if fgf:
              #img_path_spl_spl = img_path_sl[-1]
              #addition_part = img_path_spl_spl.split('.')[0][:55] 
              img_path = perm_full_path # (os.sep).join(img_path_sl[:-1]) + os.sep + addition_part+ '.png'
              Image.open(img_path)
          except:
              print('a problem with image path')
              
              print(img_path)
              print(len(img_path))
              raise ValueError('cannot find file img_path')
      try:
          #print('img path create')
          #print(img_path)
          check_mask(name_mask) 
          response = openai.Image.create_edit(
            image=open(img_path , "rb"),
            mask=open(name_mask, "rb"),
            prompt=prompt,
            n=n,
            size=size,
            response_format = form_return
          )
          #print('succes')
      except:
            #response = ''
          check_mask(path_save_res+ os.sep +"mask5.png") 
          response = openai.Image.create_edit(
            image=open(img_path , "rb"),
            mask=open(path_save_res+ os.sep +"mask5.png", "rb"),
            prompt="I love ice-cream",
            n=n,
            size=size,
            response_format = form_return
          )
          #print('succes')    
          former = Image.open(img_path )
          plt.imshow(former)
          plt.title('former')
  else:
      try:
          #check_mask(path_save_res+ os.sep +"mask5.png") 
          response = openai.Image.create(
            prompt=prompt,
            n=n,
            size=size,
            response_format = form_return
          )
      except:
            response = ''
  return response

def response2img(response, to_save_img = True, name_img = 'reg_name', show_img = False, saving_add = '', num_img = 'A1 ',
                 path_save_res = path_images, remove_text = True, reduce_sat = reduce_sat):
  if len(response) == 0:
      nn = np.zeros((image_s ,image_s ))
      img = Image.fromarray(nn).convert('RGB')
      cont_local = False
  else: 
      image_dict = response['data'][0]
      base_64_image = image_dict['b64_json']
      # Assuming base64_str is the string value without 'data:image/jpeg;base64,'
      img = Image.open(io.BytesIO(base64.decodebytes(bytes(base_64_image, "utf-8"))))
      cont_local = True
  if remove_text:
      img = inpaint_text(img , img_path = [], pipeline = pipeline, return_PIL = True)
  if to_save_img:
      #print('to save address')
      #print(path_save_res+ os.sep + name_img +saving_add)
      
      
      name_img = name_img.replace("'","")
      name_img = name_img.replace("’","")
      name_img = name_img.replace(":","")
      
      #print('name img response')
      print(name_img)

      #global s_desired #, sat_ratio
      if reduce_sat:
          try:
              med = np.median(np.median(img, axis = 0), axis = 0)
              s_desired = np.load(path_save_res + os.sep + 's_desired.npy')[0]
              _,s_cur,_ = colorsys.rgb_to_hsv(*med)
              sat_ratio = s_desired / s_cur
              #if num_img =='A1 ':
          except:
              
              
              med = np.median(np.median(img, axis = 0), axis = 0)
              _,s_desired,_ = colorsys.rgb_to_hsv(*med)
              np.save(path_save_res + os.sep + 's_desired.npy',s_desired)
              sat_ratio = 1
    

      if reduce_sat and sat_ratio != 1:
          #img = PIL.Image.open('bus.png')
          converter = ImageEnhance.Color(img)
          img = converter.enhance(sat_ratio)       
          
      perm_full  = name_img +saving_add
      print(perm_full)
      print('len perm full')
      print(len(perm_full))
      global perm_full_path
      if len(name_img +saving_add +'.png') > 56:
          perm_cut = perm_full[:56]
          print('perm full is long')    
          np.save(perm_cut + '.npy', perm_full)
      else:
          perm_cut = perm_full
          
      
      perm_full_path = path_save_res+ os.sep + perm_cut +'.png'
      if not (perm_cut).endswith('.png'):#path_save_res+ os.sep + name_img +saving_add
          img.save(path_save_res+ os.sep + perm_cut +'.png')
      else:
          img.save(path_save_res+ os.sep + name_img +saving_add )
      #except: 
      #    img.save(path_save_res+ os.sep + 'invalid_val_%d'%np.random.randint(1000) +saving_add +'.png')
  if show_img:
    display(img)
  return img, cont_local 


def all_folders_with_png(m_path = ''):
    if len(m_path) == 0:
        m_path = r'C:\Users\14434\Documents\GitHub\DALL-E-Hebrew\FIGURES FOR PAPER\**\*'
    all_subs = glob.glob(m_path, recursive=True)
    subs_rel = []
    for sub in all_subs:
        if not sub.endswith('.png'):
            if len(glob.glob(sub + os.sep + '*.png')):
                #print(sub)
                subs_rel.append(sub)
    return subs_rel
    
    
    
def from_hebrew_prompt_to_img(prompt_original = 'שלום לכולם',  size = "%dx%d"%(image_s ,image_s ), form_return = 'b64_json', to_save_img = True,
                              name_img_addi = '', show_img = True, addition = ' ,ציור צבעוני ', undesired = '"|;<>}{@#$%^&*~-[]@#|$%^&*()-?!,.‘’', 
                              path_save = '',  cont = False, mask_path = "mask5.png",   img_path_former = '',saving_add='א',n=1,
                              original_lang = 'he', path_save_res =path_images, topic = '',  remove_text = True, num_img = 'A1 ',
                              multi = False, n_multi = 4, diffuse = True, level =1, level_center =1,color_style = 'cymk', object_det = False, counter = -1, create_new_path = True)  :
  if len(topic) > 0 and (topic not in path_save_res) and create_new_path:
      path_save_res = path_save_res + os.sep+ topic + str(datetime.today()).replace(' ','_').replace(':','_').replace('.','_').replace('’','_')
  if not os.path.exists(path_save_res):
      os.makedirs(path_save_res, exist_ok = False)
  print(prompt_original)
  
  prompt_original = prompt_original + addition
  if original_lang == 'en':
      prompt_translation = prompt_original
  else:
      prompt_translation = translate_text(prompt_original, original_lang = original_lang)
  if len(color_style) > 0:
      prompt_translation = prompt_translation + ', ' + color_style
  if  object_det and counter < 0 :
      raise ValueError('if object_det then you must provide counter')
  elif object_det and counter == 0:
      prompt_translation = prompt_translation + ', white background'
      print('prompt translated if detect object and counter = 0')
      
      
  
  #print('img_path_former_here')
  #print(img_path_former)
  #print('cont in hebrew')
  #print(cont)
  response = create_dalle_response(prompt_translation, size, form_return , img_path = img_path_former,
                                   cont = cont, saving_add = saving_add, n=n, multi = multi, n_multi = n_multi, diffuse = diffuse,
                                   level =level, level_center =level_center, path_save_res= path_save_res)
  time.sleep(13)
  for sign_spec in undesired:
      prompt_original = prompt_original.replace(sign_spec, '_')
  prompt_original = prompt_original.replace("'", '_')
  name_img = path_save + os.sep + num_img +  prompt_original.strip()+name_img_addi.strip() 
  
 
  img,  cont_local  = response2img(response, to_save_img = to_save_img, name_img = name_img, show_img = show_img,
                     saving_add=saving_add, path_save_res = path_save_res, remove_text = remove_text, num_img = num_img)
  time.sleep(8)
  img_path = path_save_res+ os.sep + name_img +saving_add +'.png'
  return img, img_path,  cont_local
 
def song2images(song_words,window_size = 6, overlap = 2, addition = ' , ציור צבעוני' , saving_add= '', div_num = 1,
                min_thres_sim = 0.8, cont = True, initial_image = [],level_change = 1, n=1,original_lang = 'he',
                path_save_res = path_images, topic = 'storytelling',  remove_text = True, cut_by_line = False, reps = 1,remove_first_word = False,
                type_web = 'science', name_article = '',  start_from_p = 0, end_in_p = 0,pair_lines = False, diffuse = True,
                multi = False, n_multi = 4, color_style = 'cymk', level = 0, level_center = 0, repeat = 1, object_det = False, full_box = False,
                thres_white = 230,object_det_percent = 40, remove_dots_oject = True, percent_remove_obj = 0.1, background_increase = False,control_eli = False,
                edges = 10, back_c = 255, resize_dim = 150,object_det_veto  = True, back_instead_white = '', side_lock = [0.5, 0.5], object_det_init = True, to_cut = False,
                remove_first_addition = True, dots_T = 4, addi_count = 0 ,  create_new_path  = True, disable_directions = [], move_obj = False, move_obj_std = 1,
                cut_by_dot = False, quarter = False, include_half = False,  edge_size = 20, dots = True):
    """
    
    Parameters
    ----------
    song_words : TYPE
        DESCRIPTION.
    window_size : TYPE, optional
        DESCRIPTION. The default is 6.
    overlap : TYPE, optional
        DESCRIPTION. The default is 2.
    addition : TYPE, optional
        DESCRIPTION. The default is ' , ציור צבעוני'.
    saving_add : TYPE, optional
        DESCRIPTION. The default is ''.
    div_num : TYPE, optional
        DESCRIPTION. The default is 1.
    min_thres_sim : TYPE, optional
        DESCRIPTION. The default is 0.8.
    cont : TYPE, optional
        DESCRIPTION. The default is False.
    initial_image : TYPE, optional
        DESCRIPTION. The default is [].
    level_change : TYPE, optional
        DESCRIPTION. The default is 1.
    n : TYPE, optional
        DESCRIPTION. The default is 1.
    original_lang : TYPE, optional
        DESCRIPTION. The default is 'he'.
    path_save_res : TYPE, optional
        DESCRIPTION. The default is path_images.
    topic : TYPE, optional
        DESCRIPTION. The default is ''.
    remove_text : TYPE, optional
        DESCRIPTION. The default is True.
    cut_by_line : TYPE, optional
        DESCRIPTION. The default is False.
    reps : TYPE, optional
        DESCRIPTION. The default is 1.
    remove_first_word : TYPE, optional
        DESCRIPTION. The default is False.
    type_web : TYPE, optional
        DESCRIPTION. The default is 'science'.
    name_article : TYPE, optional
        DESCRIPTION. The default is ''.
    start_from_p : TYPE, optional
        DESCRIPTION. The default is 0.
    end_in_p : TYPE, optional
        DESCRIPTION. The default is 0.
    pair_lines : TYPE, optional
        DESCRIPTION. The default is False.
    diffuse : TYPE, optional
        DESCRIPTION. The default is True.
    multi : TYPE, optional
        DESCRIPTION. The default is False.
    n_multi : TYPE, optional
        DESCRIPTION. The default is 4.
    color_style : TYPE, optional
        DESCRIPTION. The default is 'cymk'.
    level : TYPE, optional
        DESCRIPTION. The default is 1.
    level_center : TYPE, optional
        DESCRIPTION. The default is 1.
    repeat : TYPE, optional
        DESCRIPTION. The default is 1.
    object_det : TYPE, optional
        DESCRIPTION. The default is False.
    full_box : TYPE, optional
        DESCRIPTION. The default is False.
    thres_white : TYPE, optional
        DESCRIPTION. The default is 230.
    object_det_percent : TYPE, optional
        DESCRIPTION. The default is 40.
    remove_dots_oject : TYPE, optional
        DESCRIPTION. The default is True.
    percent_remove_obj : TYPE, optional
        DESCRIPTION. The default is 0.1.
    background_increase : TYPE, optional
        DESCRIPTION. The default is False.
    control_eli : TYPE, optional
        DESCRIPTION. The default is False.
    edges : TYPE, optional
        DESCRIPTION. The default is 10.
    back_c : TYPE, optional
        DESCRIPTION. The default is 255.
    resize_dim : TYPE, optional
        DESCRIPTION. The default is 150.
    object_det_veto : TYPE, optional
        DESCRIPTION. The default is True.
    back_instead_white : TYPE, optional
        DESCRIPTION. The default is ''.
    side_lock : TYPE, optional
        DESCRIPTION. The default is [0.5, 0.5].
    object_det_init : TYPE, optional
        DESCRIPTION. The default is True.
    to_cut : TYPE, optional
        DESCRIPTION. The default is False.
    remove_first_addition : TYPE, optional
        DESCRIPTION. The default is True.
    dots_T : TYPE, optional
        DESCRIPTION. The default is 4.
    addi_count : TYPE, optional
        DESCRIPTION. The default is 0.
    create_new_path : TYPE, optional
        DESCRIPTION. The default is True.
    disable_directions : TYPE, optional
        DESCRIPTION. The default is [].
    move_obj : TYPE, optional
        DESCRIPTION. The default is False.
    move_obj_std : TYPE, optional
        DESCRIPTION. The default is 1.
    cut_by_dot : TYPE, optional
        DESCRIPTION. The default is False.
    quarter : TYPE, optional
        DESCRIPTION. The default is False.
    include_half : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    cartoon_imgs : TYPE
        DESCRIPTION.

    """
    
    # Define variables 
    if include_half:
        elipses_center_x = [0.1,0.5, 0.9,0.9 ,0.9, 0.5, 0.1, 0.1]
        elipses_center_y = [0.9,0.9, 0.9,0.5, 0.1 ,0.5, 0.1, 0.5 ]
    else:
        elipses_center_x = [0.9, 0.1,0.1,0.9]
        elipses_center_y = [0.1, 0.1,0.9,0.9] 


    
    if len(topic) > 0 and (topic not in path_save_res) and create_new_path : #and object_det:
        path_save_res = path_save_res + os.sep+ topic + str(datetime.today()).replace(' ','_').replace(':','_').replace('.','_').replace("'",'_').replace('’','_')
    
    if not os.path.exists(path_save_res):
        os.makedirs(path_save_res, exist_ok = False)
    np.save(path_save_res +os.sep+ 'meta.npy' , locals())    
    if ('.txt' in song_words) or (os.sep in song_words):
        try:
            song_words = txt2str(song_words, cut_by_line=cut_by_line, cut_by_dot = cut_by_dot)
            
        except:
            print('Model suspect the input is a path but cannot find the path. Are you sure the first input is a path? (if not - no action is needed)')
    elif song_words == 'web':
        song_words = web2str(type_web , name_article, start_from_p , end_in_p , div_num = div_num )
    cont_local_black = True
    if cut_by_line or cut_by_dot:
        string_list = song_words.split('kkkkkk')
        if pair_lines:
            string_list1 = string_list[1:]
            string_list2 = string_list[:-1]
            string_list = [string_list2[i] + ' ' + string_list1[i] for i in range(len(string_list1))]
    else:    
        list_words = song_words.split(' ')
        word_groups = [list_words[i:np.min([i+window_size, len(list_words)])] for i in np.arange(0, len(list_words), window_size - overlap)]
        string_list = [' '.join(sublist) for sublist in word_groups]
    if reps > 1:
        string_list = np.repeat(string_list, reps)
        
    if cont:
            
        cartoon_imgs = []
        if object_det and  object_det_init:
            if len(topic) > 0:
                
                string_list = [topic ] + list(string_list)
            else:
                string_list = [ string_list[0] ] + list(string_list)
        for counter_c, cartoon_string_spec in enumerate(string_list):
            counter = counter_c + addi_count
            if repeat > 1:
                cartoon_string_spec = (cartoon_string_spec+ ' ')*repeat

            if counter_c == 0:
                if len(initial_image) > 0 and ( not object_det  or not  object_det_init ):
                    if os.sep not in initial_image: 
                        initial_image = path_initial + os.sep + initial_image
                    img_path_former = initial_image

                    

                else:
                    img_path_former = ''
            img_path_former_former = img_path_former

            cont_local = (counter != 0 or len(initial_image) > 0) and img_path_former != ''
      
            former_store = img_path_former
            if counter == 0 and object_det and remove_first_addition and  object_det_init :
                addition_cur = ""
            else:
                addition_cur = ", " + addition
            if not object_det:
                _ = create_mask(seed = counter, level_change= level_change, multi = multi, n_multi = n_multi, diffuse = diffuse, 
                             level =level, level_center =level_center, path_save_res = path_save_res,
                             object_det_percent = object_det_percent,object_det = object_det, counter = counter, edge_size = edge_size,
                             object_det_veto =object_det_veto,dots_T = dots_T , disable_directions=disable_directions,control_eli=control_eli,
                             quarter=quarter ,  elipses_center_x =  elipses_center_x,  elipses_center_y =  elipses_center_y, dots = dots)

            if remove_first_word:
                cartoon_string_spec = ' '.join(cartoon_string_spec.split()[1:])
            img, img_path_former_return,  cont_local_black = from_hebrew_prompt_to_img(prompt_original = cartoon_string_spec, 
                                                             cont = cont_local , img_path_former=img_path_former,
                                                             addition =addition_cur,saving_add=saving_add,n=n,original_lang=original_lang,
                                                             path_save_res=path_save_res, remove_text = remove_text, level =level, level_center =level_center,
                                                             num_img = 'A'+str(counter) + ' ' , multi = multi, n_multi =  n_multi , 
                                                             diffuse = diffuse, color_style=color_style, object_det = object_det , counter =counter,
                                                             create_new_path = True)
            
            if object_det and counter == 0 and  object_det_init :
                if background_increase:
                    
                    img = add_back(img, path_save_res ,cartoon_string_spec ,form_return = 'b64_json', edges = edges, back_c = back_c, to_cut = to_cut, 
                             resize_dim = resize_dim, size = "%dx%d"%(image_s,image_s), n = n, side_lock = side_lock, original_lang=original_lang)
                    img.save(path_save_res + os.sep + 'A'+str(counter) + ' .png')

                else:
                
                    img.save(path_save_res + os.sep + 'original_img_expand.png')
                    
                rows_obj, cols_obj, cur_mask = identify_object_white_back(np.array(img), thres_white = thres_white)    
                if len(back_instead_white) > 0:
                    if not original_lang.startswith('en'):
                        prompt_translation = translate_text(back_instead_white, original_lang = original_lang)
                    else:
                        prompt_translation = back_instead_white
               
                    response = openai.Image.create(
                      prompt=prompt_translation + ', '+ color_style,
                      n=n,
                      size= "%dx%d"%(image_s,image_s),
                      response_format  = 'b64_json'
                    )
                    img_back = response2img(response)[0]
                    img_back.save(path_save_res + os.sep +'soround_img.png')
                    display(img_back)
                    img  = fill_settings(img, np.array(cur_mask), np.array(img_back), change2img = True)

                    img.save(path_save_res + os.sep + 'original_img_expand.png')
                    img_path_former = path_save_res + os.sep + 'original_img_expand.png'
             

                
                if len(initial_image) > 0:
                    initial_path = initial_image = path_initial + os.sep + initial_image
                    img_back = Image.open(initial_path)
                    img  = fill_settings(img, np.array(cur_mask), np.array(img_back), change2img = True)
                    #img.save(path_save_res + os.sep + 'original_img_expand_with_back.png')
                    img.save(path_save_res + os.sep + 'original_img_expand.png')
                    #img_path_former = path_save_res + os.sep + 'original_img_expand_with_back.png'
                    img_path_former = path_save_res + os.sep + 'original_img_expand.png'
                else: 
                    img.save(path_save_res + os.sep + 'original_img_expand.png')
                    print('elsee!!!!!!!!!!!!')
                    img_path_former = path_save_res + os.sep + 'original_img_expand.png'

                np.save(path_save_res+ os.sep +'rows_cols_only_fig.npy',{'rows_obj':rows_obj, 'cols_obj':cols_obj})
                
                
                if full_box:
                    rows_obj, cols_obj = make_full_box(rows_obj, cols_obj)
                                
                if remove_dots_oject:
                    ind_remove = np.arange(len(rows_obj))
                    np.random.shuffle(ind_remove)
                    int_keep =  int(np.ceil( (1-percent_remove_obj)*len(ind_remove)))
                    take = ind_remove[:int_keep]
                    remove = ind_remove[ - (len(ind_remove) - int_keep):]

                    for ind in remove:
                        cur_mask[rows_obj[ind], cols_obj[ind]] = 0 
                    rows_obj = rows_obj[take]
                    cols_obj = cols_obj[take]

                    
                plt.imshow(img)
                plt.title('img')
                plt.imshow(cur_mask)
                plt.title('mask' + str(cur_mask.mean()) + ' ' + str(cur_mask.shape))
                
                np.save(path_save_res+ os.sep +'rows_cols.npy',{'rows_obj':rows_obj, 'cols_obj':cols_obj})
            """
            move obj
            """
            if move_obj:
                
                rows_cols = np.load(path_save_res+ os.sep +'rows_cols_only_fig.npy', allow_pickle=True).item()
                rows_obj = rows_cols['rows_obj']
                cols_obj = rows_cols['cols_obj']
                samp = int(np.random.normal(0, move_obj_std, size = 2))
                rows_obj_mod = rows_obj + samp[0]
                cols_obj_mod = cols_obj + samp[1]
                img_array = np.array(img)
                img_array_change = img_array.copy()
                for mask_num in range(len(rows_obj)):
                    cur_row_mod = rows_obj_mod[mask_num]
                    cur_col_mod = cols_obj_mod[mask_num]
                    img_array_change[cur_row_mod,cur_col_mod] = img_array[rows_obj[mask_num], cols_obj[mask_num]]
                Image.fromarray(img_array_change).save(path_save_res + os.sep + 'new_img_expand.png')
                img_path_former = path_save_res + os.sep + 'new_img_expand.png'
                np.save(path_save_res+ os.sep +'rows_cols_only_fig.npy',{'rows_obj':rows_obj_mod, 'cols_obj':cols_obj_mod})
                rows_obj_mod, cols_obj_mod = make_full_box(rows_obj_mod, cols_obj_mod)
                np.save(path_save_res+ os.sep +'rows_cols.npy',{'rows_obj':rows_obj_mod, 'cols_obj':cols_obj_mod})

    
            if object_det and counter == 0 and len(initial_image) == 0 and len(back_instead_white) == 0:
                _ = create_mask_det_edge(cur_mask, path_save_res = path_save_res)
                
            elif object_det and counter == 0 and (len(initial_image) != 0 or len(back_instead_white) != 0):
                _ = create_mask(seed = counter, level_change= level_change, multi = multi, n_multi = n_multi, diffuse = diffuse, 
                                level =level, level_center =level_center, path_save_res = path_save_res,
                                object_det_percent = object_det_percent,object_det = object_det, counter = 1,quarter=quarter, edge_size = edge_size,
                                object_det_veto =object_det_veto,dots_T = dots_T, disable_directions=disable_directions,control_eli=control_eli ,
                                elipses_center_x =  elipses_center_x,  elipses_center_y =  elipses_center_y, dots = dots)               
            else:
                _ = create_mask(seed = counter, level_change= level_change, multi = multi, n_multi = n_multi, diffuse = diffuse, 
                                level =level, level_center =level_center, path_save_res = path_save_res,
                                object_det_percent = object_det_percent,object_det = object_det, counter = counter,quarter=quarter, edge_size = edge_size,
                                object_det_veto =object_det_veto,dots_T = dots_T, disable_directions=disable_directions,control_eli=control_eli,
                                elipses_center_x =  elipses_center_x,  elipses_center_y =  elipses_center_y, dots = dots)
            

                
            if object_det and counter == 0:
                pass
            else:
                if  cont_local_black:
                    img_path_former_former =  img_path_former    
                    img_path_former = img_path_former_return
                else:
                    img_path_former =img_path_former_former
            if object_det and counter == 0:
                img.save(path_save_res+ os.sep +'initial_image_white.png')
            else:
                cartoon_imgs.append(img)
            
            
    else:
        cartoon_imgs = [from_hebrew_prompt_to_img(prompt_original = cartoon_string_spec, addition =", " + addition, saving_add = saving_add,n=n,
                                                  original_lang=original_lang, path_save_res=path_save_res, remove_text = remove_text,color_style=color_style,
                                                  num_img = 'A'+ str(counter) + ' ',  multi = multi, n_multi =  n_multi, diffuse = diffuse, level =level,
                                                  level_center = level_center,
                                                  create_new_path =  create_new_path 
                                                  )[0] for counter, cartoon_string_spec in enumerate(string_list)]
    return cartoon_imgs

def flow_out(image_1, image_2, interval = 1, size_dec = 100):
    if not isinstance(image_1, np.ndarray):
        image_1 = np.array(image_1)
    if not isinstance(image_2, np.ndarray):
        image_2 = np.array(image_2)
    take_each = int((image_s - size_dec)/2)

    image_1[take_each  : -take_each ,take_each: -take_each,:]=  np.array(Image.fromarray(image_2).resize((size_dec, size_dec)))
    old_s = size_dec
    images_list = [Image.fromarray(image_1)]

    
    for extra in range(1,take_each, interval):
        new_s = old_s + 2*extra
        resize_im = np.array(Image.fromarray(image_2).resize((new_s,new_s) ))
        image_1[np.max([take_each - extra,0]) : np.min([-take_each + extra, image_s]),np.max([take_each - extra,0]) : np.min([-take_each + extra, image_s]),:]  = resize_im
    
        images_list.append(Image.fromarray(image_1))
    return images_list
    


def create_recu_animation(path_main,main_n1 = 'A',main_n2  = 'B' , size_dec = 150, interval = 1, duration = 1, name_save = 'gif_mi_yodea')    :
    
    main_files = glob.glob(path_main + os.sep+ '*.png')
    """
    order files
    """
    main_files = [f for f in main_files if f.split(os.sep)[-1].startswith('A')]
    start_A_uniuqe = np.unique([f.split(os.sep)[-1].split()[0] for f in main_files if f.split(os.sep)[-1].startswith('A')])
    list_order = []
    for count_A in np.arange(len(start_A_uniuqe)):
        cur_num = 'A%d'%count_A
        list_order.append(cur_num)
    list_grid = []
    images_list = []
    rep_size = int((image_s - size_dec)/(2*interval))
    former_files = []
    images_order_files = []
    for A_val_count, A_val in enumerate(list_order):
        files_opts =[file for file in main_files if file.split(os.sep)[-1].startswith(A_val)        ]
        file_first = [file for file in files_opts if file.split(os.sep)[-1].split()[1].startswith('B0')  ][0]
        image_first = Image.open(file_first)
        file_second = [file for file in files_opts if file.split(os.sep)[-1].split()[1].startswith('B2')  ][0]
        
        image_sec = Image.open(file_second)
        images_list.extend([image_first]*rep_size*10)

        images_list.extend([image_sec]*rep_size*10)
        if A_val_count != 0:
            back = image_sec
            for former_file_num in np.arange(len(former_files)-1, -1,-1):
                new = former_files[former_file_num]
                images_list.extend(flow_out(back, new, interval, size_dec))
                back = new

        former_files.append(image_sec    )
            
        images_order_files.extend([file_first, file_second])

    images_list[0].save(path_main + os.sep +name_save +'.gif', save_all=True, append_images=images_list[1:], duration = duration)
    return list_order, images_list
    
        
    
  
    
    

def make_full_box(rows_obj, cols_obj):
    leftest = np.min(cols_obj)
    rightest = np.max(cols_obj)
    uppest = np.min(rows_obj)
    lowest = np.max(rows_obj)    
    mesh = np.meshgrid(np.arange(uppest, lowest), np.arange(leftest, rightest))
    rows_obj = mesh[0].flatten()
    cols_obj = mesh[1].flatten()
    return rows_obj, cols_obj
    

def fill_settings(img, rep_mask, setting_img_or, change2img = True):
    setting_img = setting_img_or.copy()
    if not isinstance(setting_img,np.ndarray):
        setting_img = np.array(setting_img)
    if not isinstance(img,np.ndarray):
        img = np.array(img)
    if len(setting_img.shape) >= 3 and len(rep_mask.shape) < 3:
        rep_mask = np.repeat(rep_mask.reshape((rep_mask.shape[0], rep_mask.shape[1], 1)), setting_img.shape[2], axis = 2)
    elif    len(setting_img.shape) < 3 and len(rep_mask.shape) >= 3:
        rep_mask = rep_mask[:,:,-1]
    if img.shape[-1] == 3 and setting_img.shape[-1] == 4:
        img = np.dstack([img, (255*np.ones((img.shape[0], img.shape[1], 1))).astype(np.uint8) ])
    setting_img[rep_mask == 1] = img[rep_mask == 1]
    if change2img:
        setting_img = Image.fromarray(setting_img)
    return setting_img
    
    
    
def cut_white_from_sides(img_or, thres_white  = 230)  :
    img = img_or.copy()
    rows_obj, cols_obj, cur_mask = identify_object_white_back(img, thres_white)    
    leftest = np.min(cols_obj)
    rightest = np.max(cols_obj)
    uppest = np.min(rows_obj)
    lowest = np.max(rows_obj)
    img = img[uppest:lowest, leftest:rightest]
    return img
    
    
    
def add_back(img, path_save_res ,prompt ,form_return = 'b64_json', edges = 10, back_c = 255, resize_dim = 150, to_cut = False,
             size = "%dx%d"%(image_s,image_s), n = 1, change2image = True, side_lock = [0.5,0.5], original_lang = 'he', thres_white = 230):
    """
    side_lock: edge up; (1-p = edge down); edge left (complete edge right)
    """
 
    img_array = np.array(img)
    if to_cut:
        img_array = cut_white_from_sides(img_array, thres_white  = thres_white) 
    Image.fromarray(img_array).save(path_save_res + os.sep + 'cutted_image.png')
    if original_lang != 'en':
        #back_instead_white
        prompt = translate_text(prompt, original_lang = original_lang)
        #back_instead_white
    if 'white background' not in prompt:
        prompt = prompt + ', white background'
    

    resize_dim_practice_hor = int(resize_dim / image_s*img_array.shape[1])
    resize_dim_practice_ver = int(resize_dim / image_s*img_array.shape[0])
    # LEFT UPPER CORNER
    left = int(side_lock[1]*(image_s -  resize_dim_practice_hor  ) )
    right = left + resize_dim_practice_hor 
    upper = int(side_lock[0]*(image_s -  resize_dim_practice_ver)) 
    lower = upper + resize_dim_practice_ver       
 

    
    """
    creating mask
    """    
    mask_back = 255*np.ones((image_s, image_s))
    mask_back[edges:-edges,edges:-edges] = 0
    
    mask_back[upper:lower, left:right]  = 255
    mask_back = Image.fromarray(mask_back.astype(np.uint8))
    mask_back.save(path_save_res + os.sep + 'mask_back.png')
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img.save(path_save_res + os.sep + 'original_img.png')
    
    """
    create back
    """
    img_reduced = np.array(img.resize(( resize_dim_practice_hor, resize_dim_practice_ver  )))
    back = np.zeros((image_s, image_s, np.array(img).shape[2])) + back_c    

    back[upper:lower, left:right] = img_reduced
    back = back.astype(np.uint8)
    
    Image.fromarray(back).save(path_save_res + os.sep + 'back_img.png')
    time.sleep(5)
    
    """
    fill response
    """
    print('prompt refill')
    response = openai.Image.create_edit(
      image=open(path_save_res + os.sep + 'back_img.png' , "rb"),
      mask=open(path_save_res + os.sep + 'mask_back.png', "rb"),
      prompt=prompt,
      n=n,
      size=size,
      response_format = form_return
    )
    if 'white_bacground' not in prompt:
        prompt = prompt + ', white background'
    img_ex, _ = response2img(response, to_save_img = False, name_img = 'increase_back', show_img = False, saving_add = '', num_img = 'A000 ', 
                 path_save_res = path_save_res, remove_text = True, reduce_sat = True)
    img_ex.save(path_save_res + os.sep + 'original_img_expand.png')
    
    if change2image:
        back = Image.fromarray(back)
    return  back 

    
    
    

    
def identify_object_white_back(img, thres_white = 220, full_box = False):
    img = img.copy()
    if np.max(img) <= 1:
        img = img*255
    ar_img = np.array(img)    
    ar_img_sum3 = (ar_img > thres_white).sum(2)
    cur_mask = 1*(ar_img_sum3 < img.shape[2])
    rows_obj,cols_obj = np.where(ar_img_sum3 < img.shape[2])
    if full_box:
        leftest = np.min(cols_obj)
        rightest = np.max(cols_obj)
        uppest = np.min(rows_obj)
        lowest = np.max(rows_obj)
        mesh = np.meshgrid(np.arange(uppest, lowest), np.arange(leftest, rightest))
        rows_obj = mesh[0].flatten()
        cols_obj = mesh[1].flatten()
        cur_mask[uppest:lowest, leftest:rightest] = 1
    return rows_obj, cols_obj, cur_mask
    
def keep_object_in_mask(rows_obj, cols_obj, mask, percentile = 40, change_to_Image = True, disable_directions = []):
    mask = np.array(mask)
    mask = mask.copy()
    if np.max(mask) > 1:
        mask = 1*(mask > 0) 
    options_vert = ['up','down']
    options_horz = ['right','left']
    
    if ('up' in disable_directions) and ('down' in disable_directions): 
        raise ValueError('you cannot disable both up and down!')
    if ('right' in disable_directions) and ('left' in disable_directions): 
        raise ValueError('you cannot disable both right and left!')
        
    options_vert = [ve for ve in options_vert if ve not in disable_directions]
    options_horz = [ve for ve in options_horz if ve not in disable_directions]

    vert = np.random.choice(options_vert) # which side to replace?  
    horz = np.random.choice(options_horz) # which side to replace?
    

    if vert == 'up':
        perc_rows = np.percentile(rows_obj, percentile)
    else:
        perc_rows = np.percentile(rows_obj,100 -  percentile)
    if horz == 'left':
        perc_cols = np.percentile(cols_obj, percentile)
    else:
        perc_cols = np.percentile(cols_obj, 100 - percentile)


    for dot_num in np.arange(len(rows_obj)):
        cur_row = rows_obj[dot_num]
        cur_col = cols_obj[dot_num]
        left_to_thres =  cur_col < perc_cols
        up_to_thres = cur_row < perc_rows
        
        if vert == 'up' and horz == 'left':
            mask[cur_row, cur_col] =1 - 1*(up_to_thres  and left_to_thres)# 1*( (not up_to_thres) and (not left_to_thres) ) #
        elif vert == 'down' and horz == 'left':
            mask[cur_row, cur_col] =1 - 1*((not up_to_thres)  and left_to_thres)# 1*( up_to_thres and (not up_to_thres) ) 
           
        elif vert == 'down' and horz == 'right':
            mask[cur_row, cur_col] = 1-1*( (not up_to_thres)  and (not left_to_thres) )            #1*(  up_to_thres and  up_to_thres) #
        elif vert == 'up' and horz == 'right':
            mask[cur_row, cur_col] = 1 - 1*(up_to_thres  and (not left_to_thres)            )#1*( (not up_to_thres) and  up_to_thres ) #
    plt.imshow(mask)
    plt.title('dfdsfdsf')
    plt.show()
    if change_to_Image:
        mask = mask * 255
        mask = mask.astype(np.uint8)
        mask = Image.fromarray(mask)
    return mask
    

    
    
    
    
"""
Treat Website Calls
"""

def remove_brack_html(txt, remove_span = True, start_from_p = 0, end_in_p = 0):
    # starting from later paragraph
    if start_from_p != 0 or end_in_p != 0:
        txt_cut = txt.split('</p>')

        if start_from_p != 0:
            txt_cut = txt_cut[start_from_p:]
        else:
            txt_cut = txt_cut[:end_in_p]
        txt = ' '.join(txt_cut)
    split_small = txt.split('<')

    split_large = [spl_small.split('>')[1]  if ('>' in spl_small and len(spl_small.split('>')) > 1) else spl_small
                   for spl_small in   split_small]

    new_text = ' '.join(split_large)

    new_text = new_text.replace('>', '')
    new_text = txt2str(new_text,is_file=False)
    new_text = new_text.replace('  ', ' ')
    new_text = new_text.replace(os.sep, ' ')
    new_text = new_text.replace("\t", ' ')
    new_text = new_text.replace("\r", ' ')
    new_text = new_text.replace("'", ' ')
    new_text = new_text.replace('  ', ' ')
    #print('new t')
    #print(new_text)
    return new_text





def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    return (x_mid, y_mid)
global pipeline
pipeline = keras_ocr.pipeline.Pipeline()

def inpaint_text(img =[] , img_path = [], pipeline = pipeline, return_PIL = True):
    # from https://towardsdatascience.com/remove-text-from-images-using-cv2-and-keras-ocr-24e7612ae4f4
    # read image
    img = np.array(img)
    if len(img) == 0:
        img = keras_ocr.tools.read(img_path)
    # generate (word, box) tuples 
    pipeline = keras_ocr.pipeline.Pipeline()
    prediction_groups = pipeline.recognize([img])
    mask = np.zeros(img.shape[:2], dtype="uint8")
    for box in prediction_groups[0]:
        x0, y0 = box[1][0]
        x1, y1 = box[1][1] 
        x2, y2 = box[1][2]
        x3, y3 = box[1][3] 
        
        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)
        
        thickness = int( ( (x2 - x1)**2 + (y2 - y1)**2 )**0.5 )
        
        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255,    
        thickness)
        img = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)
    if return_PIL:
        img = Image.fromarray(img)
    return(img)

def call_web(type_web = 'science', name_article = '', main_split = 'mainArticle', second_split = '</div>', div_num = 1):
    # shironet or bible or science
    
    url = input('Webpage to grab source from: ')
    html_output_name = type_web + name_article # input('Name for html file: ')
    req = requests.get(url, 'html.parser')
    with open(html_output_name, 'w', encoding="utf8") as f:
        f.write(req.text)
        f.close()
        
    #print(req.text)
    spl1 = req.text.split(main_split)
    #print(spl1)
    spl2 = spl1[1].split(second_split)
    #print(spl2)
    if div_num != 1:
        txt_from_web = spl2[:div_num-1]
        txt_from_web = ' '.join(txt_from_web)
    else:
        txt_from_web = spl2[0]
    return txt_from_web

def web2str(type_web = 'science', name_article = '',  start_from_p = 0, end_in_p = 0, div_num = 1)    :
   if type_web =='science' :
       print('for brain: https://davidson.weizmann.ac.il/online/sciencepanorama/%D7%94%D7%9B%D7%99-%D7%97%D7%9B%D7%9E%D7%99%D7%9D-%D7%A9%D7%99%D7%A9?dicbo=v2-b83c675ab7b9185c3c5d85b4f52b21d3')
       print('for Galaxy:https://davidson.weizmann.ac.il/online/askexpert/astrophysics/%D7%9E%D7%94%D7%95-%D7%91%D7%A2%D7%A6%D7%9D-%D7%94%D7%99%D7%A7%D7%95%D7%9D-%D7%94%D7%A0%D7%A8%D7%90%D7%94')
       print('blood cells: https://davidson.weizmann.ac.il/online/maagarmada/med_and_physiol/%D7%9E%D6%B7%D7%A2%D6%B2%D7%A8%D6%B6%D7%9B%D6%B6%D7%AA-%D7%94%D6%B7%D7%93%D6%B8%D7%9D')
       print('for Neuton: https://davidson.weizmann.ac.il/online/sciencehistory/%D7%94%D7%90%D7%99%D7%A9-%D7%A9%D7%97%D7%A9%D7%91-%D7%A2%D7%9C-%D7%94%D7%9B%D7%95%D7%9C')
       print('for light: https://davidson.weizmann.ac.il/online/sciencepanorama/%D7%A7%D7%99%D7%A6%D7%95%D7%A8-%D7%AA%D7%95%D7%9C%D7%93%D7%95%D7%AA-%D7%94%D7%90%D7%95%D7%A8')
       print('for ionicL https://davidson.weizmann.ac.il/online/maagarmada/chemistry/%D7%9E%D7%94%D7%95-%D7%A7%D7%A9%D7%A8-%D7%99%D7%95%D7%A0%D7%99?dicbo=v2-54a4f052b999bcbd35de6e58f4cc75dc')
       print('for dna: https://davidson.weizmann.ac.il/online/askexpert/%D7%9E%D7%97%D7%A1%D7%9F-%D7%94%D7%9E%D7%99%D7%93%D7%A2-%D7%94%D7%AA%D7%95%D7%A8%D7%A9%D7%AA%D7%99')
       print('for moon: https://davidson.weizmann.ac.il/online/askexpert/%D7%90%D7%99%D7%9A-%D7%A0%D7%95%D7%A6%D7%A8-%D7%94%D7%99%D7%A8%D7%97')
       print('for atom: https://davidson.weizmann.ac.il/online/maagarmada/chemistry/%D7%9E%D7%94%D7%95-%D7%A7%D7%A9%D7%A8-%D7%99%D7%95%D7%A0%D7%99?dicbo=v2-54a4f052b999bcbd35de6e58f4cc75dc')
       print('atp: https://davidson.weizmann.ac.il/online/maagarmada/life_sci/atp-%d7%a1%d7%99%d7%a0%d7%aa%d7%90%d7%96-%e2%80%93-%d7%94%d7%9e%d7%a0%d7%95%d7%a2-%d7%94%d7%9e%d7%95%d7%9c%d7%a7%d7%95%d7%9c%d7%a8%d7%99-%d7%94%d7%9e%d7%95%d7%a9%d7%9c%d7%9d')
       print('rain: https://davidson.weizmann.ac.il/online/askexpert/%D7%9E%D7%93%D7%95%D7%A2-%D7%99%D7%95%D7%A8%D7%93-%D7%92%D7%A9%D7%9D-%D7%91%D7%97%D7%95%D7%A8%D7%A3')
       print('earth: https://davidson.weizmann.ac.il/online/askexpert/astrophysics/%D7%90%D7%99%D7%9A-%D7%A0%D7%95%D7%A6%D7%A8-%D7%9B%D7%93%D7%95%D7%A8-%D7%94%D7%90%D7%A8%D7%A5-%D7%A0%D7%92%D7%94')
       print('water: https://davidson.weizmann.ac.il/online/maagarmada/earth_sci/%D7%9E%D7%97%D7%96%D7%95%D7%A8-%D7%94%D7%9E%D7%99%D7%9D-%D7%91%D7%9B%D7%93%D7%95%D7%A8-%D7%94%D7%90%D7%A8%D7%A5')
       print('dna 1: https://davidson.weizmann.ac.il/online/maagarmada/life_sci/%D7%AA%D7%94%D7%9C%D7%99%D7%9A-%D7%A9%D7%9B%D7%A4%D7%95%D7%9C-%D7%94-dna , div 3')
       print('dna 2: https://davidson.weizmann.ac.il/online/maagarmada/life_sci/%D7%90%D7%99%D7%9A-%D7%AA%D7%90%D7%99%D7%9D-%D7%9E%D7%AA%D7%97%D7%9C%D7%A7%D7%99%D7%9D-%D7%9E%D7%99%D7%98%D7%95%D7%96%D7%94 , div3')
       print('dna3: https://davidson.weizmann.ac.il/online/askexpert/%D7%9E%D7%97%D7%A1%D7%9F-%D7%94%D7%9E%D7%99%D7%93%D7%A2-%D7%94%D7%AA%D7%95%D7%A8%D7%A9%D7%AA%D7%99 , div2')
       print('mendel: https://davidson.weizmann.ac.il/online/maagarmada/life_sci/%d7%99%d7%a1%d7%95%d7%93%d7%95%d7%aa%20%d7%94%d7%92%d7%a0%d7%98%d7%99%d7%a7%d7%94%20%e2%80%93%20%d7%97%d7%95%d7%a7%d7%99%20%d7%9e%d7%a0%d7%93%d7%9c , div2')

       text_from_web =  call_web(type_web, name_article, div_num = div_num)
   elif type_web == 'song' :
       
       print('for Danni gibor: https://shironet.mako.co.il/artist?type=lyrics&lang=1&prfid=778&wrkid=2653')
       main_split = 'artist_lyrics_text">'
       second_split = '</span>'
       text_from_web = call_web(type_web = 'song', name_article = name_article, main_split = main_split,
                                second_split =  second_split, div_num = div_num )
       
   text_from_web_clean = remove_brack_html(text_from_web, True, start_from_p , end_in_p )
   
   return text_from_web_clean


def name2num(name):
    nn = name.split(os.sep)[-1].split(' ')[0]
    try:
        n_num = int(nn[1:])
    except:
        n_num = -1
    return n_num
def path2nums(path):
    opts = glob.glob(path+ os.sep + '*.png')
    #opts = [im for im in opts if im.startswith('A')]
    nums = [name2num(opt) for opt in opts if name2num(opt) >= 0]
    sort_args = np.argsort(nums)
    opts_org = np.array(opts)[sort_args]
    return opts_org

    
def create_gif(images = [], path = '.', duration = 300, reps = 15, path_save = r'.', name_save = 'gif',
               max_images = 90,
               base_name = 'A', reorder = True, reorder2 = True):
    if checkEmptyList(images):
        if path == '':
            #path = r'C:\Users\14434\Documents\GitHub\DALL-E-Hebrew\FIGURES FOR PAPER\good sections\YEKUM_w_numbers'
            raise ValueError('path should not be empty')
        images = glob.glob(path + os.sep + '*.png')

        images = np.sort(images)
        if reorder:

            max_img = len(images)
            images = [path + os.sep + '%s ('%base_name + str(num) + ').png' for num in np.arange(1, max_img + 1)]
        if reorder2:
            images = path2nums(path)

        if len(images) > max_images:
            images = images[:max_images]
        images_read = [plt.imread(path)*255 for path in images]#lists2list([[plt.imread(path)]*reps for path in images])

    else:
        images_read = images    
    
    weighted = 1- np.linspace(0,1, reps)
    #print(weighted)
    w_inv = 1- weighted
    new_imgs_pil = []
    # if isinstance(images[0],np.ndarray):
    #     new_imgs_pil.append( Image.fromarray((images_read[0]).astype(np.uint8)))
    # else:
    #     new_imgs_pil = images
    #images_read = [im for im in images_read if im.startswith('A')]
    #print(images_read)
    new_imgs_pil.append(im2image(images_read[0]))
    
    for img_num, img in enumerate(images_read[:-1]):
        
        for i_rep, rep in enumerate(np.arange(reps)):
            img1 = im2array(img)[:,:,:3]#np.dstack([img , 255*np.ones(img.shape[:2])])#=img#
            img2 = im2array(images_read[img_num+1])[:,:,:3] #np.dstack([images_read[img_num+1] , 255*np.ones(img.shape[:2])])# images_read[img_num+1] #np.dstack([images_read[img_num+1] , 255*np.ones(img.shape[:2])])
            #img2 = images_read[img_num+1] #np.dstack([images_read[img_num+1] , 255*np.ones(img.shape[:2])])
            image_comb = weighted[i_rep] * img1 + w_inv[i_rep]*img2
            image_comb = np.round(image_comb).astype(np.uint8)
            #print(image_comb)
            #input('cont')
            #print(image_comb.shape)
            img_pil = im2image(image_comb).convert('RGBA')#Image.fromarray(image_comb)
            #display(img_pil)
            new_imgs_pil.append(img_pil)
        
        new_imgs_pil.append(im2image(images_read[img_num+1])) #Image.fromarray( (images_read[img_num+1]).astype(np.uint8))      )
    print(path_save + os.sep +name_save +'.gif')
    new_imgs_pil[0].save(path_save + os.sep +name_save +'.gif', save_all=True, append_images=new_imgs_pil[1:], duration=duration)

def lists2list(xss)    :
    return [x for xs in xss for x in xs] 

def im2image(im):
    if isinstance(im, np.ndarray):
        return Image.fromarray(im.astype(np.uint8))
    elif isinstance(im,Image.Image):
            return im
    else:
        raise ValueError('Unknown input type')
        
def im2array(im):
    if isinstance(im, np.ndarray):
        return im
    elif isinstance(im,Image.Image):
            return np.array(im)
    else:
        raise ValueError('Unknown input type')        
    

def  create_gif_old(images, smooth = True, path_save = r'.', name_save = 'gif', dur_between = 0.1, dur_main = 2.5, width = 8):
    if isinstance(images[0], str):
        images = [Image.open(image).convert('RGBA') for image in images]
    #http://elliothallmark.com/2014/06/08/making-gif-animations-with-transitions/
    #img1, img2 = (Image.open('%s' % x) for x in ['1.png','2.png'])
    #fade from 1 to 2
    if smooth:
        #not checkEmptyList(v_fade):
        images_store = []
        for j in range(len(images)-1)    :
            img1 = images[j]
            img2 = images[j+1]
            images_n = [Image.composite(img2,img1,mask) for mask in v_fade(images, width )]
            #fast on transition frams but hold end images longer
            durations = [dur_between]*len(images_n)
            durations[0] = durations[-1] = dur_main
            #fade from 2 to 1
            images_n.extend([Image.composite(img1,img2,mask) for mask in v_fade(images, width)])
            durations.extend(durations)
        images_store.extend(images_n)
    
    
        #writeGif(path_save + os.sep +name_save +'.gif',images, duration=durations)
        images_store[0].save(path_save + os.sep +name_save +'.gif', save_all=True, append_images=images_store[1:], duration=durations)#fs = 0.05)
    images[0].save(path_save + os.sep +name_save +'.gif', save_all=True, append_images=images[1:], duration=300)#fs = 0.05)
    
    
    
def v_fade(images, width = 3):
    '''make masks for fading from one image to the next through a vertical sweep.  Does this through numpy slices'''
    #global img1                #from img1 = Image.open('one.png')
    img1 = images[0]
    n = np.array(img1)        #make a numpy.array the same size as the frame
    n.dtype=np.uint8        #correct the data type
    n[:][:] = 0                #the first mask lets everything through
    #copy the array, else all masks will be the last one
    result=[Image.fromarray(n.copy())]
    for i in range(int(n.shape[0]/width)+1):
        #add vertical strips of width one at a time.
        y=i*(width)
        n[y:y+width][:] = 255
        result.append(Image.fromarray(n.copy()))
    return result    

def remove_edges(ax, include_ticks = False, top = False, right = False, bottom = False, left = False):
    ax.spines['top'].set_visible(top)    
    ax.spines['right'].set_visible(right)
    ax.spines['bottom'].set_visible(bottom)
    ax.spines['left'].set_visible(left)  
    if not include_ticks:
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])    



def checkEmptyList(obj):
    return isinstance(obj, list) and len(obj) == 0

def show_image_based_on_path(path, with_labels = False, ax = []):
    if not (path.endswith('.png') or path.endswith('jpg') or path.endswith('.PNG') or path.endswith('JPG') or  path.lower().endswith('jpeg')):
            raise ValueError('Invalid path. Path should be an image.')
    if checkEmptyList(ax):
        fig, ax = plt.subplots()
    img = plt.imread(path)
    ax.imshow(img)
    
    
    
def create_subplots_fig(path = '', num_cols = 6, max_images = 50, reorder = False, dur = 0.2,interval = 5,
                        fig = [], axs = [],with_labels = False, title = 'subplots', base_name = 'A', reorder2 = True,size_dec=100,
                        path_save =basic_path , flow = False):
    if path == '':
        path =basic_path
    images = glob.glob(path + os.sep + '*.png')
    images = np.sort(images)
    #print(images)
    if reorder:
        #images = np.sort(images)
        max_img = len(images)
        images = [path + os.sep + '%s ('%base_name + str(num) + ').png' for num in np.arange(1, max_img + 1)]
    if reorder2:
        images = path2nums(path)        
    #print(images)
    if len(images) > max_images:
        images = images[:max_images]
    num_rows = int(np.ceil(len(images)/num_cols))
    if  checkEmptyList(fig) and not  checkEmptyList(axs):
        raise ValueError('PLEASE PROVIDE FIG AS INPUT')
    elif checkEmptyList(axs):
        fig, axs = plt.subplots(num_rows, num_cols, figsize = (num_cols *5, num_rows*5))
    #fig, axs = plt.subplots(5,6, figsize = (20,20))
    axs = axs.flatten()
    #string_list = [' '.join(sublist) for sublist in word_groups]
    [show_image_based_on_path(image_path, with_labels = False, ax = axs[i]) for i,image_path in enumerate(images)]
    topic = path.split(os.sep)[-1]
    if flow:
        durs = []
        images_all = []
        images_former = [Image.open(images[-1])]
        
        for i,image_path in enumerate(images[len(images)-1:0:-1]):
            image_1 = Image.open(images[-i-1-1])
            
            
            durs.append(dur*10)
            image_2 = Image.open(images[-i-1])
            images_all.append(image_2)
            images_between = flow_out(image_2, image_1, interval = interval, size_dec = size_dec)
            durs.extend([dur]*len(images_between))
            images_all.extend(images_between)
            images_all.append(image_1)
            durs.append(dur*10)
            images_former.append(image_1)
        images_all[0].save(path_save + os.sep + topic + title +'.gif', save_all=True, append_images=images_all[1:], duration = durs)
    #[Image.open(image_path)]
    
    [ax.set_xticks([]) for ax in axs]
    [ax.set_yticks([]) for ax in axs]
    [remove_edges(ax) for ax in axs]
    if len(title) > 0: fig.suptitle(title)
    
    fig.tight_layout()
    plt.savefig(path_save + os.sep + topic + title + '.png', bbox_inches='tight')
    return images
    #if to_plot:
        
# window_size =5
# overlap =2
# list_words = jerusalem.split(' ')
#word_groups = [list_words[i:np.min([i+window_size, len(list_words)])] for i in np.arange(0, len(list_words), window_size - overlap)]

#[ax.set_title(string_list[i][::-1]) for i, ax in enumerate(axs)]
    
    
    

#%% OBject detecion (taken from https://towardsdatascience.com/yolo-object-detection-with-opencv-and-python-21e50ac599e9)

#https://towardsdatascience.com/object-detection-with-10-lines-of-code-d6cb4d86f606

# there keep most of the image


# from imageai.Detection import ObjectDetection
# import os

# execution_path = os.getcwd()

# detector = ObjectDetection()
# detector.setModelTypeAsRetinaNet()
# detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.1.0.h5"))
# detector.loadModel()
# detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "image.jpg"), output_image_path=os.path.join(execution_path , "imagenew.jpg"))

# for eachObject in detections:
#     print(eachObject["name"] , " : " , eachObject["percentage_probability"] )







# from imageai.Detection import ObjectDetection
# import os

# execution_path = os.getcwd()





# detector = ObjectDetection()
# detector.setModelTypeAsRetinaNet()
# detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
# detector.loadModel()
# detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "image.jpg"), output_image_path=os.path.join(execution_path , "imagenew.jpg"))



# for eachObject in detections:
#     print(eachObject["name"] , " : " , eachObject["percentage_probability"] )


# detections, extracted_images = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "image.jpg"), output_image_path=os.path.join(execution_path , "imagenew.jpg"), extract_detected_objects=True)








# # # read input image
# # def find_locs_image(image,scale = 0.00392, )

# # Width = image.shape[1]
# # Height = image.shape[0]


# # # read class names from text file
# # classes = None
# # with open(args.classes, 'r') as f:
# #     classes = [line.strip() for line in f.readlines()]

# # # generate different colors for different classes 
# # COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# # # read pre-trained model and config file
# # net = cv2.dnn.readNet(args.weights, args.config)

# # # create input blob 
# # blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

# # # set input blob for the network
# # net.setInput(blob)

# # # function to get the output layer names 
# # # in the architecture
# # def get_output_layers(net):
    
# #     layer_names = net.getLayerNames()
    
# #     output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# #     return output_layers

# # # function to draw bounding box on the detected object with class name
# # def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

# #     label = str(classes[class_id])

# #     color = COLORS[class_id]

# #     cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

# #     cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


# # # run inference through the network
# # # and gather predictions from output layers
# # outs = net.forward(get_output_layers(net))

# # # initialization
# # class_ids = []
# # confidences = []
# # boxes = []
# # conf_threshold = 0.5
# # nms_threshold = 0.4

# # # for each detetion from each output layer 
# # # get the confidence, class id, bounding box params
# # # and ignore weak detections (confidence < 0.5)
# # for out in outs:
# #     for detection in out:
# #         scores = detection[5:]
# #         class_id = np.argmax(scores)
# #         confidence = scores[class_id]
# #         if confidence > 0.5:
# #             center_x = int(detection[0] * Width)
# #             center_y = int(detection[1] * Height)
# #             w = int(detection[2] * Width)
# #             h = int(detection[3] * Height)
# #             x = center_x - w / 2
# #             y = center_y - h / 2
# #             class_ids.append(class_id)
# #             confidences.append(float(confidence))
# #             boxes.append([x, y, w, h])
    
# # # apply non-max suppression
# # indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

# # # go through the detections remaining
# # # after nms and draw bounding box
# # for i in indices:
# #     i = i[0]
# #     box = boxes[i]
# #     x = box[0]
# #     y = box[1]
# #     w = box[2]
# #     h = box[3]
    
# #     draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

# # # display output image    
# # cv2.imshow("object detection", image)

# # # wait until any key is pressed
# # cv2.waitKey()
    
# #  # save output image to disk
# # cv2.imwrite("object-detection.jpg", image)

# # # release resources
# # cv2.destroyAllWindows()    