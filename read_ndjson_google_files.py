# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 16:55:05 2018

@author: jsaavedr
"""

import json, cv2
import numpy as np
import argparse as argp
import os.path
import sys

#--------create image from a set of points ------------------------------------------------------------
def createImage(points):        
    x_points = [] 
    y_points = [] 
    target_size = 256
    object_size = 200
    # reading all points
    for stroke in points:
        x_points = x_points + stroke[0]
        y_points = y_points + stroke[1]        
    # min max for each axis
    min_x = min(x_points)
    max_x = max(x_points)
    min_y = min(y_points)
    max_y = max(y_points)        

    im_width = np.int ( max_x - min_x  + 1 )
    im_height= np.int (max_y - min_y  + 1  )        
    
    if im_width > im_height :
        resize_factor = np.true_divide(object_size, im_width)
    else:
        resize_factor = np.true_divide(object_size, im_height)
        
    t_width = np.int(im_width * resize_factor )
    t_height = np.int(im_height * resize_factor )
    
    center_x = np.int( sum(x_points) / len(x_points) )
    center_y = np.int( sum(y_points) / len(y_points) )
    
    center_x = np.int(t_width * 0.5 ) 
    center_y = np.int(t_height * 0.5 ) 
    
    t_center_x = np.int(target_size * 0.5)    
    t_center_y = np.int(target_size * 0.5)    
    
    offset_x = t_center_x - center_x    
    offset_y = t_center_y - center_y
   
    blank_image = np.zeros((target_size, target_size), np.uint8) 
    blank_image[:,:]=255;    
    #cv2.circle(blank_image, (), 1, 1, 8)
    for stroke in points:
        xa = -1
        ya = -1    
        for p in zip(stroke[0], stroke[1]):
            x = np.int( np.true_divide(p[0]  - min_x, im_width) * t_width) + offset_x
            y = np.int( np.true_divide(p[1]  - min_y, im_height) * t_height)  + offset_y
            #if x in range(0,1024) and y in range(0,1024):
            if xa>=0 and ya>=0 :                                
                    cv2.line(blank_image, (xa,ya), (x,y), 0, 3)
            xa=x
            ya=y            
            blank_image[y,x] = 0
    return blank_image
    
#------------------main----------------------------------------------------------    
if __name__ == "__main__":    
    parser  = argp.ArgumentParser(description = "QuickDraw to images")
    parser.add_argument("-list_of_classes",  help="name of the category", required = True)
    parser.add_argument("-source_path",  help="path of the ndjson files", required = True )
    parser.add_argument("-destine_path",  help="path where images will be saved", required = True)
    args = parser.parse_args();       
    
    class_file = args.list_of_classes
    if not os.path.isfile(class_file) :
        sys.exit(1)
    source_path = args.source_path
    destine_path =  args.destine_path
    f_log = open("quick_draw_log_images.log", "w")    
    f_log.flush()
    n_class = 0    
    with open(class_file) as f_class:
        for str_class in f_class:
            str_class = str_class.strip()
            drawing_file = source_path + os.path.sep +  str_class + ".ndjson"       
            destine_dir = destine_path + os.path.sep + str_class
            if not os.path.exists(destine_dir):
                os.makedirs(destine_dir)
            code_class = "{:03d}".format(n_class)
            f_log.write("CLASS \t" + str_class + "\t" + code_class + "\n")
            i = 0        
            with open(drawing_file) as f:    
                for str_line in f:
                    code_image = "{:08d}".format(i)
                    str_line = str_line.strip()            
                    data = json.loads(str_line)
                    coords = data["drawing"]            
                    image = createImage(coords)
                    code_image = code_class + "_" + code_image + ".jpg"
                    filename = destine_dir + os.path.sep + code_image
                    cv2.imwrite(filename, image)
                    #cv2.imshow("image", image)
                    #cv2.waitKey()
                    i = i + 1
                    if i % 1000 == 0 :
                        f_log.write("----- {} processed \n".format(i))
                        f_log.flush()
            n_class = n_class +1        
    f_log.close()
