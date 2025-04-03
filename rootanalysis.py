#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import cv2 as cv
import numpy as np
from skimage import measure
import pandas as pd
import math
import logging
from skimage.morphology import convex_hull_image, skeletonize
import os


class rootAnalysis():
    def __init__(self, conf, original_img_name):
        self.conf = conf
        self.original_img_name = original_img_name
    
    def start_analysis(self):
        self.df = pd.DataFrame(
            columns=["Image_Name", "Total_Root_Length(cm)", "Total_Root_Projected_Area(cm2)",
                     'Total_Surface_Area(cm2)',
                     'Total_Root_Volume(cm3)',
                     'Convex_Hull_Area(cm2)', 'Max_Root_Depth(cm)', 'Angle_Left', 'Angle_Center',
                     'Angle_Right'])
        self.original_img_name = self.original_img_name
        # img_list = os.listdir( os.path.join(
        #     self.conf["save_path"], 'workspace', 'predict'))
        img_list = os.listdir("test_rootanalysis/workspace/predict/")
        temp_list = []
        for name in img_list:
            # if (name.split("_")[-1] == "W.jpg"):
            temp_list.append(name)
        for img in temp_list:
            try:
                self.img_analysis_b_m(img)
            except:
                pass
        os.makedirs(os.path.join(
            self.conf["save_path"], 'workspace', "output"), exist_ok=True)
        self.df.to_csv(os.path.join(
            self.conf["save_path"], 'workspace', 'output', "result.csv"), index=False)
    
    def get_dataframe(self):
        return self.df
    
    def img_analysis_b_m(self, img):
        src = cv.imread(os.path.join(
            self.conf["save_path"], 'workspace', 'predict', img))
        r, new = cv.threshold(src, 127, 255, cv.THRESH_BINARY)
        gray_img = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

        # Check if there are any root pixels in the image
        if self.check_pxiel(gray_img) == 0:
            msg = "We did not detect any root pixels. Please check the uploaded images and resubmit the task."
            print(img)
            raise ValueError("None root pixels")
            
        
        # Analysis of the connectivity domain
        parr, imgxx = self.draw_connect(src, 0)
        # Image Skeleton Extraction
        thinimg = self.img_thin(new)
        # root length (primary root + branched root)
        all_length, all_pnum_l = self.count_length(thinimg)
        # root projected area (primary root + branched root)
        all_area, all_pnum_a = self.count_area(parr)
        r, new2 = cv.threshold(gray_img, 100, 255, cv.THRESH_BINARY)
        # root surface area (primary root + branched root)
        all_surface_area = self.count_surface_area(new2)
        convex_area = self.count_convex_hull_area(gray_img)
        max_root_depth = self.count_depth(new2)
        volume = self.count_volume(all_surface_area, all_length)
        # angle_left, angle_center, angle_right = self.count_angle(new)
        angle_left, angle_center, angle_right = 0, 0, 0

        # Add data to dataframe
        data = {
            "Image_Name": img,
            "Total_Root_Length(cm)": format(all_length, '.2f'),
            "Total_Root_Projected_Area(cm2)": format(all_area, '.2f'),
            'Total_Surface_Area(cm2)': format(all_surface_area, '.2f'),
            'Total_Root_Volume(cm3)': format(volume, '.3f'),
            'Convex_Hull_Area(cm2)': format(convex_area, '.2f'),
            'Max_Root_Depth(cm)': format(max_root_depth, '.2f'),
            'Angle_Left': format(angle_left, '.2f'),
            'Angle_Center': format(angle_center, '.2f'),
            'Angle_Right': format(angle_right, '.2f')
        }
        self.df = pd.concat([self.df, pd.DataFrame(data, index=[0])])

    def check_pxiel(self, src):
        check_res = np.where(src > 10)
        if int(check_res[0].shape[0]) == 0:
            return 0
        else:
            return 1
        
    def draw_connect(self, image, type):
        if type == 1:
            gray = image.copy()
        else:
            gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

        ret, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
        imgxx = image.copy()
        lable = measure.label(binary, connectivity=2)
        props = measure.regionprops(lable, intensity_image=None, cache=True)
        parr = [[i, prop.coords] for i, prop in enumerate(props)]
        return parr, imgxx


    def img_thin(self, img):
        skeleton = skeletonize(img)
        # gray = cv.cvtColor(skeleton, cv.COLOR_RGB2GRAY)
        # gray = (gray > 1) * 255
        gray = cv.cvtColor(skeleton, cv.COLOR_RGB2GRAY)
        _, thresh = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
        return thresh


    def count_length(self, thinimg):
        pixelnum = (thinimg[:, :] == 255).sum()
        length = pixelnum * float(self.conf["cf_value"])
        return length, int(pixelnum)

    def count_area(self, parr):
        pixelnum = 0
        for p in range(0, len(parr)):
            pixelnum = pixelnum + len(parr[p][1])
        area = pixelnum * float(self.conf["cf_value"]) * float(self.conf["cf_value"])
        return area, pixelnum

    def count_depth(self, img):
        first = np.where(img == 255)[0][0] * float(self.conf["cf_value"])
        last = np.where(img == 255)[0][-1] * float(self.conf["cf_value"])
        depth = last - first
        return depth

    def count_volume(self, surface_area, all_length):
        volume = 3.1415927 * surface_area * surface_area / 4 / all_length * 0.1
        return volume

    def count_surface_area(self, img):
        can = cv.Canny(img, 1, 255)
        area = np.where(img == 255)[0].shape[0]
        edge = np.where(can == 255)[0].shape[0]
        surface_area = (area - (edge / 2 + 1)) * 3.1415927 * float(self.conf["cf_value"]) * float(self.conf["cf_value"])
        return surface_area

    def count_convex_hull_area(self, img):
        points = convex_hull_image(img)
        convexhull_area = np.where(points == True)[0].shape[0] * float(self.conf["cf_value"]) * float(self.conf["cf_value"])
        return convexhull_area

    def count_angle(self, initial):
        t = cv.cvtColor(initial * 255, cv.COLOR_BGR2GRAY) * 255
        edge = cv.Canny(t[0:200, :], 50, 150, apertureSize=3)
        line_points_plus = []
        line_points_minus = []
        lines = cv.HoughLines(edge, 1, np.pi / 180, 40)
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            if y1 > 0:
                line_points_plus.append([x1, y1, x2, y2])
            else:
                line_points_minus.append([x1, y1, x2, y2])
        cross_point_res = []
        for plus in line_points_plus:
            for minus in line_points_minus:
                cross_point_res.append(self.cross_point(plus, minus))
        for i in range(0, len(cross_point_res)):
            for j in range(0, 2):
                cross_point_res[i][j] = int(cross_point_res[i][j])
        need_to_fit = np.array(cross_point_res)
        first_line = 0
        t_line_index = 0
        for t_line in t[0:200, :]:
            res = [i for i in np.where(t_line > 100)[0]]
            if len(res) > 0:
                first_line = t_line_index
                break
            t_line_index += 1
        temp = [list(i) for i in need_to_fit if i[1] > 0]
        temp_1 = []
        for value in temp:
            temp_1.append(value[1])
        best_near = temp[temp_1.index(min(temp_1, key=lambda x: abs(x - 94)))]
        while best_near[1] > first_line:
            temp.remove(best_near)
            temp_1 = []
            for value in temp:
                temp_1.append(value[1])
            best_near = temp[temp_1.index(min(temp_1, key=lambda x: abs(x - 94)))]
        best_near_copy = best_near
        if abs(best_near_copy[1] - first_line) > 20:
            best_near_copy[1] = first_line
        search_array = np.where(t[best_near_copy[1]:best_near_copy[1] + 200, :] > 150)
        right = [max(list(search_array[1])),
                    search_array[0][list(search_array[1]).index(max(list(search_array[1])))] + best_near[1]]
        left = [min(list(search_array[1])), search_array[0][-1] + best_near[1]]
        L1 = [left, best_near_copy]
        L2 = [right, best_near_copy]
        L3 = [[best_near_copy[0] - 20, best_near_copy[1]], [best_near_copy[0] + 20, best_near_copy[1]]]
        angle_left = self.angle_3(L1, L3)
        angle_right = self.angle_3(L2, L3)
        angle_center = 180 - angle_left - angle_right
        return angle_left, angle_center, angle_right
        return 0, 0, 0

    # 计算交点函数
    def cross_point(self, line1, line2):
        # 取直线坐标两点的x和y值
        x1 = line1[0]
        y1 = line1[1]
        x2 = line1[2]
        y2 = line1[3]

        x3 = line2[0]
        y3 = line2[1]
        x4 = line2[2]
        y4 = line2[3]

        # L2直线斜率不存在操作
        if (x4 - x3) == 0:
            k2 = None
            b2 = 0
            x = x3
            # 计算k1,由于点均为整数，需要进行浮点数转化
            k1 = (y2 - y1) * 1.0 / (x2 - x1)
            # 整型转浮点型是关键
            b1 = y1 * 1.0 - x1 * k1 * 1.0
            y = k1 * x * 1.0 + b1 * 1.0
        elif (x2 - x1) == 0:
            k1 = None
            b1 = 0
            x = x1
            k2 = (y4 - y3) * 1.0 / (x4 - x3)
            b2 = y3 * 1.0 - x3 * k2 * 1.0
            y = k2 * x * 1.0 + b2 * 1.0
        else:
            # 计算k1,由于点均为整数，需要进行浮点数转化
            k1 = (y2 - y1) * 1.0 / (x2 - x1)
            # 斜率存在操作
            k2 = (y4 - y3) * 1.0 / (x4 - x3)
            # 整型转浮点型是关键
            b1 = y1 * 1.0 - x1 * k1 * 1.0
            b2 = y3 * 1.0 - x3 * k2 * 1.0
            x = (b2 - b1) * 1.0 / (k1 - k2)
            y = k1 * x * 1.0 + b1 * 1.0
        return [x, y]

    def angle_3(self, L1, L2):
        a = L1[0]
        b = L1[1]
        c = L2[1]
        ang = math.degrees(
            math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
        temp = ang + 360 if ang < 0 else ang
        angle = abs(180 - temp)
        return angle if angle < 90 else 180 - angle

if __name__ == "__main__":
    conf = {"save_path":"test_rootanalysis",
            "cf_value":1} # 比例尺
    root = rootAnalysis(conf,original_img_name=None)
    root.start_analysis()
    # img = cv.imread('test.bmp', cv.IMREAD_GRAYSCALE)
    # root.img_analysis_b_m(img)



