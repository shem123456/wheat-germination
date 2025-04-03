#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# 2023.9.20 同时分析茎，种子和根的性状，并整合到一张表里
# 需要依次改485和486行的文件名称
# 2023.10.17 加一个循环，同时分析所位于文件

import cv2
import numpy as np
from skimage import measure
import pandas as pd
import math
import logging
from skimage.morphology import convex_hull_image, skeletonize
import os

from skimage import morphology

import skimage.feature as feature
import skimage.feature as feature
from skimage import measure

class rootAnalysis():
    """
    根系分析类
    input:image path
    二值化图像
    """
    def __init__(self, path, name):
        self.path = path
        self.name = name
    
    def get_image(self):
        name = os.path.join(self.path,self.name)
        # print("image_name:",name)
        img = cv2.imread(name,0)
        # cv2.imshow("img",img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        ret, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        return binary
    
    def draw_connect(self):
        image = self.get_image()

        gray = image.copy()
        ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        imgxx = image.copy()
        lable = measure.label(binary, connectivity=2)
        props = measure.regionprops(lable, intensity_image=None, cache=True)
        parr = [[i, prop.coords] for i, prop in enumerate(props)]
        return parr, imgxx
    
    def img_thin(self):
        img = self.get_image()
        binary = img
        binary[binary==255] = 1
        skeleton0 = skeletonize(binary)   # 骨架提取
        skeleton = skeleton0.astype(np.uint8)*255
        return skeleton 
    
    def img_thin_and_line(self):
        """
        计算根条数
        原理：在骨架化的图像上，画一条横细线，计算横线与骨架图的交点
        取多根横线，去所有交点数中的最大值

        return:根条数
        """
        img_thin = self.img_thin()
        length,width = img_thin.shape

        # 1300-1700,间隔设置10
        root_numbers = [0]
        root_number = 0
        point1x = []
        point1y = []
        point2x = []

        first = np.where(img_thin == 255)[0][0] * 1
        last = np.where(img_thin == 255)[0][-1] * 1
        # print("first:",first)
        # print("last:",last)
        for i in range(first,last,20):
            
            arr = img_thin[i,:]

            # 得到255的所有索引
            arr_index = np.where(arr==255)
            arr_index = arr_index[0]
            # print(i)
            # print("arr_index:",arr_index)

            root_number = len(arr_index)
            if root_number >= 2:
                point1x.append(arr_index[0])
                point1y.append(i)
                point2x.append(arr_index[-1])
             
            # 判断是否有连续的所有，有连续的，去掉连续的值
            l1 = []
            for x in sorted(set(arr_index)):
                l1.append(x)
                if x+1 not in arr_index:
                    if len(l1) != 1:
                        # print(l1)
                        root_number = root_number - len(l1) + 1
                    l1 = []
                
                
            # print("root_number:",root_number)
            root_numbers.append(root_number)
        x2 = sum(point1x) / len(point1x)
        y2 = sum(point1y) / len(point1y)
        x3 = sum(point2x) / len(point2x)
        y3 = y2
        # print("x1,y1,x2,y2",x1,y1,x2,y2)
        return max(root_numbers),x2,y2,x3,y3
    
    def count_length(self):
	# 总根长
        thinimg = self.img_thin()
        pixelnum = (thinimg[:, :] == 255).sum()
        length = pixelnum * 1
        return length, int(pixelnum)

    def count_area(self):
        parr,_ = self.draw_connect()
        pixelnum = 0
        for p in range(0, len(parr)):
            pixelnum = pixelnum + len(parr[p][1])
        area = pixelnum * 1 * 1
        return area, pixelnum

    def count_depth(self):
	# 最大根长
        img = self.get_image()
        first = np.where(img == 255)[0][0] * 1
        last = np.where(img == 255)[0][-1] * 1
        depth = last - first
        # print("first:",first)
        # print("last:",last)
        return depth
    
    def count_surface_area(self):
        img = self.get_image()
        can = cv2.Canny(img, 1, 255)
        area = np.where(img == 255)[0].shape[0]
        edge = np.where(can == 255)[0].shape[0]
        surface_area = (area - (edge / 2 + 1)) * 3.1415927 * 1 * 1
        return surface_area

    def count_volume(self):
        all_length,_ = self.count_length()
        surface_area = self.count_surface_area()
        volume = 3.1415927 * surface_area * surface_area / 4 / all_length * 0.1
        return volume
    
    def count_convex_hull_area(self):
        img = self.get_image()
        points = convex_hull_image(img)
        convexhull_area = np.where(points == True)[0].shape[0] * 1 * 1
        return convexhull_area


class seedAnalysis():
    """
    种子分析类
    input:image path
    彩色图像
    """

    def __init__(self, path1, path2):
        """初始化属性"""
        self.path1 = path1 #rgb原图
        self.path2 = path2 #mask图

    # 原图
    def image1(self):
        image = cv2.imread(self.path1)
        return image
    
    # mask图
    def image2(self):
        image = cv2.imread(self.path2,0)
        return image


    def no_background_image(self):
        image = self.image1()
        mask = self.image2()
        # gray,thresh_img = self.binary_image()
        # # 去噪
        kernel = np.ones((5,5),np.uint8)
        mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
        # 反向
        mask = cv2.bitwise_not(mask)
        # 变成3通道
        mask1 = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
        # 相加
        result = cv2.add(image,mask1)
        return result
    
    def binary_image(self):
        # 转灰度
        image_gray = cv2.cvtColor(self.no_background_image(),cv2.COLOR_BGR2GRAY)
        thresh_img = self.image2()

        return image_gray,thresh_img

    # 颜色特征
    def RGB_mean(self):
        img = self.no_background_image()
        R_mean = np.mean(img[:,:,2])
        G_mean = np.mean(img[:,:,1])
        B_mean = np.mean(img[:,:,0])
        return R_mean, G_mean, B_mean
    def HSV_mean(self):
        img = self.no_background_image()
        img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        H_mean = np.mean(img[:,:,0])
        S_mean = np.mean(img[:,:,1])
        V_mean = np.mean(img[:,:,2])
        return H_mean,S_mean,V_mean

    # 形状特征
    def morphology_trait(self):
        image_gray,thresh_img = self.binary_image()
        contours, hierarchy = cv2.findContours(thresh_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        #高级函数的用法，key直接调用函数，取面积最大的轮廓
        cnt = sorted(contours,key=cv2.contourArea)[-1]
        area = cv2.contourArea(cnt)#面积
        length = cv2.arcLength(cnt,True)#周长
        minrectangle = cv2.minAreaRect(cnt)#最小外接矩形
        (x1,y1),radius = cv2.minEnclosingCircle(cnt)#最小外接圆
        radius = int(radius)*2 #最小外接圆直径
        equi_diameter = int(np.sqrt(4*area/np.pi))#等效圆直径
        # print("cnt point:",len(cnt))
        if len(cnt) > 4:
            (x, y) , (a, b), angle = cv2.fitEllipse(cnt)#椭圆拟合
            if (a > b):
                eccentric = np.sqrt(1.0 - (b / a) ** 2)  # a 为长轴
            else:
                eccentric = np.sqrt(1.0 - (a / b) ** 2)#偏心率,范围为 [0,1]，圆的偏心率为 0 最小，直线的偏心率为 1最大
        else:
            eccentric = 0
        compact = length ** 2 / area  # 轮廓的紧致度 (compactness),紧致度是一个无量纲的测度，圆的紧致度最小，为 4 π 4\pi4π，正方形的紧致度 是 16
        rectangle_degree = area / (minrectangle[1][0]*minrectangle[1][1])#矩形度
        roundness = (4 * math.pi * area) / (length * length)#圆形度,圆的圆度为 1 最大，正方形的圆度为 π / 4 \pi / 4π/4
        
        # print("面积：",area)
        # print("周长：",length)
        # print("最小外接圆直径：",radius)
        # print("等效圆直径：",equi_diameter)
        # print("偏心率：",eccentric)
        # print("紧致度：",compact)
        # print("矩形度：",rectangle_degree)
        # print("圆形度：",roundness)
        return area,length,radius,equi_diameter,eccentric,compact,rectangle_degree,roundness,x1,y1

    # 纹理特征
    def texture_trait(self):
        image_gray,thresh_img = self.binary_image()
        graycom = feature.greycomatrix(image_gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256)
        # Find the GLCM properties
        contrast = feature.greycoprops(graycom, 'contrast')
        dissimilarity = feature.greycoprops(graycom, 'dissimilarity')
        homogeneity = feature.greycoprops(graycom, 'homogeneity')
        energy = feature.greycoprops(graycom, 'energy')
        correlation = feature.greycoprops(graycom, 'correlation')
        ASM = feature.greycoprops(graycom, 'ASM')

        contrast_mean = np.mean(contrast)#对比度
        dissimilarity_mean = np.mean(dissimilarity)#相异性
        homogeneity_mean = np.mean(homogeneity)#同质性
        energy_mean = np.mean(energy)#能量
        correlation_mean = np.mean(correlation)#相关性
        ASM_mean = np.mean(ASM)#ASM
        entropy = measure.shannon_entropy(image_gray)#熵

        return correlation_mean,dissimilarity_mean,homogeneity_mean,energy_mean,correlation_mean,ASM_mean,entropy


class seedlingAnalysis():
    """
    芽分析类
    input:image path
    二值化图像
    """
    def __init__(self, path,name):
        self.path = path
        self.name = name

    def get_image(self):
        name = os.path.join(self.path,self.name)
        # print("image_name:",name)
        img = cv2.imread(name,0)
        # cv2.imshow("img",img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        ret, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        return binary
    
    def draw_connect(self):
        image = self.get_image()

        gray = image.copy()
        ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        imgxx = image.copy()
        lable = measure.label(binary, connectivity=2)
        props = measure.regionprops(lable, intensity_image=None, cache=True)
        parr = [[i, prop.coords] for i, prop in enumerate(props)]
        return parr, imgxx
    
    def img_thin(self):
        img = self.get_image()
        binary = img
        binary[binary==255] = 1
        skeleton0 = skeletonize(binary)   # 骨架提取
        skeleton = skeleton0.astype(np.uint8)*255
        return skeleton 
    
    def img_thin_and_line(self):
        """
        计算根条数
        原理：在骨架化的图像上，画一条横细线，计算横线与骨架图的交点
        取多根横线，去所有交点数中的最大值

        return:根条数
        """
        img_thin = self.img_thin()
        length,width = img_thin.shape

        # 1300-1700,间隔设置10
        root_numbers = [0]
        root_number = 0
        point1x = []
        point1y = []
        point2x = []

        first = np.where(img_thin == 255)[0][0] * 1
        last = np.where(img_thin == 255)[0][-1] * 1
        # print("first:",first)
        # print("last:",last)
        for i in range(first,last,20):
            
            arr = img_thin[i,:]

            # 得到255的所有索引
            arr_index = np.where(arr==255)
            arr_index = arr_index[0]
            # print(i)
            # print("arr_index:",arr_index)

            root_number = len(arr_index)
            # if root_number >= 2:
            #     point1x.append(arr_index[0])
            #     point1y.append(i)
            #     point2x.append(arr_index[-1])
            
            # 判断是否有连续的所有，有连续的，去掉连续的值
            l1 = []
            for x in sorted(set(arr_index)):
                l1.append(x)
                if x+1 not in arr_index:
                    if len(l1) != 1:
                        # print(l1)
                        root_number = root_number - len(l1) + 1
                    l1 = []
                
                
            # print("root_number:",root_number)
            root_numbers.append(root_number)
        # x1 = sum(point1x) / len(point1x)
        # y1 = sum(point1y) / len(point1y)
        # x2 = sum(point2x) / len(point2x)
        # y2 = y1
        # print("x1,y1,x2,y2",x1,y1,x2,y2)
        return max(root_numbers)
    
    def count_length(self):
        thinimg = self.img_thin()
        pixelnum = (thinimg[:, :] == 255).sum()
        length = pixelnum * 1
        return length, int(pixelnum)

    def count_area(self):
        parr,_ = self.draw_connect()
        pixelnum = 0
        for p in range(0, len(parr)):
            pixelnum = pixelnum + len(parr[p][1])
        area = pixelnum * 1 * 1
        return area, pixelnum

    def count_depth(self):
        img = self.get_image()
        first = np.where(img == 255)[0][0] * 1
        last = np.where(img == 255)[0][-1] * 1
        depth = last - first
        return depth
    
    def count_convex_hull_area(self):
        img = self.get_image()
        points = convex_hull_image(img)
        convexhull_area = np.where(points == True)[0].shape[0] * 1 * 1
        return convexhull_area

# 1.籽粒的性状
def seed_save_result(path1,path2):
    
    # path1 = path1.encode('gbk')
    # path2 = path2.encode('gbk')
    seed = seedAnalysis(path1,path2)
    # image1 = seed.image1()
    # image2 = seed.image2()
    # cv2.imwrite("image1.jpg", image1)
    # cv2.imwrite("image2.jpg", image2)
    img = seed.no_background_image()
    cv2.imwrite("nbj.jpg", img)


    # 颜色特征
    R,G,B = seed.RGB_mean()
    H,S,V = seed.HSV_mean()
    # 形状特征
    area,length,radius,equi_diameter,eccentric,compact,rectangle_degree,roundness,x1,y1 = seed.morphology_trait()
    # 纹理特征
    correlation_mean,dissimilarity_mean,homogeneity_mean,energy_mean,correlation_mean,ASM_mean,entropy = seed.texture_trait()

    # 字典
    result_dict = {
            "R":R,"G":G,"B":B,"H":H,"S":S,"V":V,
            "area":area,"length":length,"radius":radius,"equi_diameter":equi_diameter,"eccentric":eccentric,
            "compact":compact,"rectangle_degree":rectangle_degree,"roundness":roundness,
            "x1":x1,"y1":y1,
            "correlation":correlation_mean,"dissimilarity":dissimilarity_mean,
            "homogeneity":homogeneity_mean,"energy":energy_mean,"correlation":correlation_mean,"ASM":ASM_mean,"entropy":entropy
    }
    return result_dict

# 2.茎的性状
def seedling_save_result(path,img_name):
    root1 = seedlingAnalysis(path,img_name)
    img = root1.get_image()
    thin_img = root1.img_thin()
    area = root1.count_area()
    root_number = root1.img_thin_and_line()
    convex_hull_area = root1.count_convex_hull_area()
    depth = root1.count_depth()
    length = root1.count_length()
    results = {
        'seedling_area':area[0],
        'seedling_number':root_number,
        'seedling_convex_hull_area':convex_hull_area,
        'seedling_height':depth,
        'seedling_length':length[0]
    }
    return results

# 3.根的性状
def root_save_result(path,img_name):
    root1 = rootAnalysis(path,img_name)
    img = root1.get_image()
    thin_img = root1.img_thin()
    area = root1.count_area()
    root_number,x2,y2,x3,y3 = root1.img_thin_and_line()
    convex_hull_area = root1.count_convex_hull_area()
    depth = root1.count_depth()
    length = root1.count_length()
    surface_area = root1.count_surface_area()
    volume = root1.count_volume()
    results = {
        'root_area':area[0],
        'root_number':root_number,
        'root_convex_hull_area':convex_hull_area,
        'root_depth':depth,
        'root_length':length[0],
        'root_surface_area':surface_area,
        'root_volume':volume,
        'x2':x2,'y2':y2,'x3':x3,'y3':y3
    }
    return results


if __name__ == "__main__":

    # 原始路径
    root_path = "./root_requirements/picture_after_resize/"
    root1_path = "./root_requirements/picture_result/"
    
    days = os.listdir(root_path)

    for day in days[6:]:
        dates = os.listdir(os.path.join(root_path,day))
        for date in dates:
            print(day)
            print(date)
            
            # 要改的路径
            day_path = day
            date_path = date
   
            all_path = os.path.join(root_path,day_path,date_path)
            list_seed = os.listdir(all_path)
            # print(list_seed)
            # 给三个部分分别建一个空的列表容器
            result_seed = []
            result_seedling = []
            result_root = []

            result_all = []

            # 按图像名循环
            for name in list_seed                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         :
                if name.endswith(".jpg"):
                    print(name)

                    # 1.种子的性状
                    path1 = os.path.join(root_path,day_path,date_path,name)
                    path2 = os.path.join(root1_path,"seed",day_path,date_path,name)
                    print("path1:",path1)
                    print("path2:",path2)
                    # 去掉.jpg
                    name1 = name.replace(".jpg","")
                    name_info = name1.split("-")
                    result_dict1 = {"Day":day_path,"Date":date_path,"Stress":name_info[0],"Variety":name_info[1],"repeat1":name_info[2],"repeat2":name_info[3]}
                    result_dict2 = seed_save_result(path1,path2)
                    result_dict1.update(result_dict2)
                    print(result_dict1)
                    # result_seed.append(result_dict1)
            

                    # 茎的性状
                    path3 = os.path.join(root1_path,"seedling", day_path, date_path)
                    # print("path3:"+path3+"/"+name)
                    try:
                        result_dict3 = seedling_save_result(path3,name)
                    except:
                        print("No pass:",name)
                        result_dict3 = {
                                    'seedling_area':0,
                                    'seedling_number':0,
                                    'seedling_convex_hull_area':0,
                                    'seedling_height':0,
                                    'seedling_length':0,
                                }
                    print(result_dict3)
                    # result_seedling.append(result_dict3)

                    # 根的性状
                    path4 = os.path.join(root1_path,"root", day_path, date_path)
                    # print("path4:",path4)
                    try:
                        result_dict4 = root_save_result(path4,name)
                    except:
                        result_dict4 = {
                                    'root_area':0,
                                    'root_number':0,
                                    'root_convex_hull_area':0,
                                    'root_depth':0,
                                    'root_length':0,
                                    'root_surface_area':0,
                                    'root_volume':0,
                                    'x2':0,'y2':0,'x3':0,'y3':0
                                }
                    print(result_dict4)
                    # result_root.append(result_dict4)

                    result_dict1.update(result_dict3)
                    result_dict1.update(result_dict4)
                    result_all.append(result_dict1)

            df = pd.DataFrame(result_all)
            csv_path = os.path.join('./root_requirements/csv_result3',"{}-{}.csv".format(day_path,date_path))#改了
            columns = ['Day','Date','Stress','Variety','repeat1','repeat2','R','G','B','H','S','V',
                    'area','length','radius','equi_diameter','eccentric','compact','rectangle_degree',
                    'roundness','correlation','dissimilarity','homogeneity','energy','entropy','ASM',
                    'seedling_number','seedling_height','seedling_length','seedling_area','seedling_convex_hull_area',
                    'root_number','root_depth','root_length','root_volume','root_area','root_convex_hull_area',
                    'root_surface_area','x1','y1','x2','y2','x3','y3']
            df.to_csv(csv_path,index=False,columns=columns)
            print("{}保存!".format(csv_path))  
