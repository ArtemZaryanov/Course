import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os





# Переписать через класс и имплементировать в код
# Подумать над опредлением конусов(левые или правые) через сегментацию или способом Касаткина 

class LidarDataProccesing:
    def __init__(self,is_z):
        self.is_z = is_z
        pass
    def set_is_z(self,value:bool):
        self.is_z = value
        
    def get_indices_cone(self,lidar_data):
        # расстояние до агнета, Позиция (0,0)
        if self.is_z:
            r = np.hypot(lidar_data[:,0],lidar_data[:,1],1 - lidar_data[:,2])  
        else:
            r = np.hypot(lidar_data[:,0],lidar_data[:,1])  
        # Более надежно r = np.hypot(lidar_data_nsc[:,0],lidar_data_nsc[:,1])  
        # Проход по массиву
        cones_index = []
        cone_index = []
        is_cone_point = False
        is_end_cone_point = False
        for i in range(1,r.shape[0]-2):
            if is_end_cone_point:
                is_end_cone_point = False
                continue
            # Если нашли скачок, то это координаты конуса. Добавляем в отдельный массив
            if (np.abs(r[i-1]-r[i])>=0.5) and (is_cone_point==False) and (np.abs(r[i-1]-r[i])<=np.abs(r[0]-r[-1])):
                is_cone_point = True
    
            # Двигаемся  по координатам конуса 
            if is_cone_point == True:   
                cone_index.append(i)
            # Если произошел скачок во время отбора координат конуса
            if (np.abs(r[i]-r[i+1])>=1) and (is_cone_point==True):
                is_cone_point = False
                is_end_cone_point = True
                cones_index.append([cone_index])
                cone_index = []
        # print(f"Number of cones {len(cones_index)}")
        # for i,cone_index in zip(range(len(cones_index)),cones_index):
        #    print(f"Index of cone №{i}:{cone_index}")
        return cones_index

    # Получить все точки, которые принадлежат конусам
    def cone_detect(self,lidar_data,cones_index):
        # Отбор координат
        cones_coord = []
        for cone_index in cones_index:
            cones_coord.append(lidar_data[tuple(cone_index)])
        if self.is_z:
            return cones_coord
        else:
            # Нельзя использовать np.array, так как разные размерности
            # внутри массивов
            return [i[:,[0,1]] for i in cones_coord ]
    # Найти условный центр данных точек
    def cone_centr(self,cones_coord):
        return np.array(list(map( lambda a: np.mean(a, axis=0),cones_coord)))
    # Вычислить расстояние до них 
    
    def calculate_distance_cones(self,cones_centr):
        if self.is_z:
            dist = (cones_centr - np.array([0,0,1]))**2
        else:
            dist = (cones_centr - np.array([0,0]))**2
        dist = np.sum(dist, axis=1)
        dist = np.sqrt(dist)
        return dist

    def visulation_cone(self,cone_center):
        if self.is_z:
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(cone_center[:,0], cone_center[:,1], cone_center[:,2], marker='o')
            ax.scatter(0,0,1, marker='o')
            for center in cone_center:
                buf = np.append([center],[[0,0,1]],axis=0)
                ax.plot(buf[:,0],buf[:,1],buf[:,2],marker='o')
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            plt.show()
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(cone_center[:,0], cone_center[:,1], marker='o')
            ax.scatter(0,0, marker='o')
            for center in cone_center:
                buf = np.append([center],[[0,0]],axis=0)
                ax.plot(buf[:,0],buf[:,1],marker='o')
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            plt.show
    def get_cone_info(self,lidar_data):
        lidar_data_nsc =  np.transpose([lidar_data[0],lidar_data[1],lidar_data[2]])
        cone_indices = self.get_indices_cone(lidar_data_nsc)
        cone_coord = self.cone_detect(lidar_data_nsc,cone_indices)
        cone_center = self.cone_centr(cone_coord)
        dist = self.calculate_distance_cones(cone_center)
        return cone_center, dist
    def plot_disntane_function(self,lidar_data):
        if self.is_z:
            lidar_data_nsc =  np.transpose([lidar_data[0],lidar_data[1],lidar_data[2]])
            r = np.hypot(lidar_data_nsc[:,0],lidar_data_nsc[:,1],1 - lidar_data_nsc[:,2])  
        else:
            lidar_data_nsc =  np.transpose([lidar_data[0],lidar_data[1],lidar_data[2]])
            r = np.hypot(lidar_data_nsc[:,0],lidar_data_nsc[:,1])  
        f, axes = plt.subplots(nrows=1, ncols=1)
        axes.plot(r)
    def approx_data(self,cone_center,epsilon=1):
        from sklearn.metrics import pairwise_distances
        pairwise_distance_matrix = pairwise_distances(cone_center)
        cone_center_approx = []
        for i in range(len(pairwise_distance_matrix)):
            pnts = cone_center[np.where(pairwise_distance_matrix[i]<epsilon)]
            cone_center_approx.append(pnts.mean(axis=0))
        cone_center_approx = np.unique(cone_center_approx,axis=0)
        
        return cone_center_approx

class CameraDataProccesing:
    def __init__(self,is_cv2_imread,min_area,width=None,height = None):
        self.is_cv2_imread = is_cv2_imread
        self.is_up_left = None
        self.is_down_right = None
        # Зависит от разрешения Возможно лучше привести к единичному масштабу 
        self.min_area = min_area
        # Опорные цвета конусов. 
        # Зависит от сегментации. Сегментация Airsim
        # Проверка идет для RGB или HSV
        # Может меняться провертять каждый раз 
        self.sky_color_RGB = [158, 105, 215]
        self.ground_color_RGB = [ 14, 160, 234]
        self.left_cone_color_RGB =  [157,  65,  14]
        self.right_cone_color_RGB = [191,   0, 255]

        
        
        self.right_cone_color_HSV = None
        self.left_cone_color_HSV = None
        self.width = width
        self.height = height

    # Принимает на вход картинку с камеры
    # Получаем флаги
    def load_image(self):
        pass
    def convert_image_to_HSV(self,img):
        
        img_HSV = None
        # Если загружаем через cv2, то нужно
        # перевести BGR->RGB->HSV
        if self.is_cv2_imread:
            img_HSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        else:
            img_HSV = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
        return img_HSV
    def horisont_crop(self,img1):
        # добавить функицю или продумать насчет этого
        if self.is_cv2_imread:
            img = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
        else:
            img = img1.copy()
        sky,ground,left_cones,right_cones = self.color_image_split(img)
        v0 =  np.round((np.max(sky)/img.shape[1])%img.shape[0]).astype(int) - 5
        v1 = (np.round((np.max(sky)/img.shape[1])%img.shape[0]) + 5).astype(int)
        img[v0:v1,:] = self.sky_color_RGB
        return img
        
    def color_image_split(self,img_RGB):
        
        img_vector = None
        sky = None
        ground = None
        left_cones = None
        right_cones = None
        img_vector  = img_RGB.reshape(-1,img_RGB.shape[2])
        unique_elem, first_index,indices,counts = np.unique(img_vector, axis=0,
                                                            return_counts = True,
                                                            return_index = True, 
                                                            return_inverse = True)                                       
        for i in range(first_index.shape[0]):
            if np.array_equal(img_vector[first_index[i]],self.sky_color_RGB):
                sky = np.where(indices==i)
            if np.array_equal(img_vector[first_index[i]],self.ground_color_RGB):
                ground = np.where(indices==i)
            if np.array_equal(img_vector[first_index[i]],self.left_cone_color_RGB):
                left_cones = np.where(indices==i)
            if np.array_equal(img_vector[first_index[i]],self.right_cone_color_RGB):
                right_cones = np.where(indices==i)
                
                
                                                            
                                                        
                                                
        return sky,ground,left_cones,right_cones
    
    def color_clustering(self,img_HSV):
        # делим изображения на два с помощью цветовой сегментации
        # low левые конусы(внунтренние)
        # ршпр праые конусы(внешние)
        img_tresh_high = cv2.inRange(
            img_HSV, np.array([10, 135, 135]), np.array([15, 255, 255]))
        img_tresh_low = cv2.inRange(
            img_HSV, np.array([140, 135, 135]), np.array([150,255, 255]))
        return img_tresh_low,img_tresh_high
    def find_contours(self,img_HSV):
        # поиск границ
        img_edges = cv2.Canny(img_HSV,80, 160)
        # почитать про иерархию
        # построение контуров по найденным границам 
        #(раньше были просто пиксели, теперь кривые)
        contours,h = cv2.findContours(
            np.array(img_edges), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours,h
    def approx_contours(self,contours):
        # упрощение контуров (уменьшение количества точек в них) 
        #  с помощью алгоритма Дугласа-Пекера
        a_cs = []
        for c in contours:
            a_c = cv2.approxPolyDP(c,1, closed =True)
            a_cs.append(a_c)
        return a_cs
    def convexHull(self,contours):
        # преобразование контуров в выпуклые
        all_convex_hulls = []
        for ac in contours:
            all_convex_hulls.append(cv2.convexHull(ac))
        return all_convex_hulls
    
    def get_cone_and_bounding_rect(self,convex_hulls):
        # Исключаем контуры маленькой площади. Для оптимизации
        # Собираем расположение конусов и прямогоугольников,
        # ограничивающие контур
        cones = []
        bounding_rects = []
        
        for ch in convex_hulls:
            rect = cv2.boundingRect(ch)
            cones.append(ch)
            bounding_rects.append(rect)
        return cones,bounding_rects
    
    def get_cntr_p_bounding_rect(self,bounding_rects):
        cntr_ps = []
        for rect in bounding_rects:
            cntr_p = (rect[0] + rect[2]//2 ,rect[1] + rect[3])
            cntr_ps.append(cntr_p)
        return cntr_ps
    def get_bounding_rect(self,img1,is_RGB=False):
        if is_RGB:
            img_HSV = self.convert_image_to_HSV(img1)
        else:
            img_HSV = img1
        contours,h= self.find_contours(img_HSV)
        approx_contour = self.approx_contours(contours)
        all_convex_hulls = self.convexHull(approx_contour)
        cones,bounding_rects = self.get_cone_and_bounding_rect(
            all_convex_hulls)
         # Это проверка на цвет. 
        # Возможно стоит увеличить количество точек для проверки
        cntr_ps = self.get_cntr_p_bounding_rect(bounding_rects)
        return cntr_ps,bounding_rects,cones
    
    
    def get_label(self,img_res_rgb,cntr_ps):
        # Получение меток для SVM, используя инофрмацию о цвете
        labels = []
        for cntr_p in cntr_ps:
            y,x = cntr_p
            if np.array_equal(img_res_rgb[x-4,y],self.left_cone_color_RGB):
                labels.append(0)
            if np.array_equal(img_res_rgb[x-4,y],self.right_cone_color_RGB):
                labels.append(1)
        if not labels:
            print("Не найдены метки!!!!")
        return labels
class PinholeCamera:
    def __init__(self,control_points):
        self.H = None
        self.create_homogeneous_matrix(control_points)
        
    def to_homogeneous(self,cartesian_point):
        #homogeneous_point = np.append(cartesian_point.T,[[1]*len(cartesian_point)],axis=0).T
        homogeneous_point = np.append(cartesian_point.T,[1],axis=0).T
        return homogeneous_point 
    
    def to_сartesian(self,homogeneous_point):
        cartesian_point  = (homogeneous_point/homogeneous_point[-1])[0:-1]
        return cartesian_point 
    
    def get_spaces_coord(self,image_coord):
        return self.to_сartesian(np.dot(self.H,self.to_homogeneous(image_coord)))
    
    def vd(sel,x,y,X,Y):
        return np.array([[x,y,1,0,0,0,-X*x,-X*y,0],[0,0,0,x,y,1,-Y*x,-Y*y,0]])
    
    def create_homogeneous_matrix(self,control_points):
        assert isinstance(control_points,tuple), "points is not tuple"
        #(X,Y,x,y)
        X,Y,x,y = control_points
        D1 = self.vd(x[0],y[0],X[0],Y[0])
        D2 = self.vd(x[1],y[1],X[1],Y[1])
        D3 = self.vd(x[2],y[2],X[2],Y[2])
        D4 = self.vd(x[3],y[3],X[3],Y[3])
        D5 = [0]*8 +[1]
        M = np.vstack((D1,D2,D3,D4,D5))
        F = np.array([X[0],Y[0],X[1],Y[1],X[2],Y[2],X[3],Y[3],1])
        res = np.linalg.solve(M,F)
        self.H = res.reshape(3,3)
        