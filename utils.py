import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math
import scipy as sc
from queue import PriorityQueue

#################################################
#### FUNCTIONS ##################################
#################################################

def polar_angle(x,y):
    return math.atan2(y,x) % (2*math.pi)

def minkowski_sum(polygons, probs=None):
    if probs is None:
        probs = 1 / (len(polygons) * np.ones(len(polygons)))

    output_polygon = []
    first = np.array(probs) @ np.array([poly[0] for poly in polygons])
    output_polygon.append(first)

    pq = PriorityQueue()
    for poly, prob in zip(polygons, probs):
        for j in range(len(poly)- 1):
            vec = prob * (np.array(poly[j+1])- np.array(poly[j]))
            pq.put((polar_angle(vec[0], vec[1]), *vec))

    while not pq.empty():
        _, x, y = pq.get()
        output_polygon.append(output_polygon[-1] + np.array([x,y]))

    return np.array(output_polygon)

def rotation(x,y,ang):
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    arr = R @ np.array([x,y])
    return arr[0], arr[1]

def distance(first, second):
    return np.sqrt((first[0]-second[0])**2+(first[1]-second[1])**2)

def dist_to_segment(pt, start, end):
    line_vec = (end[0]-start[0], end[1]-start[1])
    pt_vec = (pt[0]-start[0], pt[1]-start[1])
    line_len = distance(line_vec, [0,0])
    line_unitvec = (line_vec[0]/line_len, line_vec[1]/line_len)
    pt_scaledvec = (pt_vec[0]/line_len, pt_vec[1]/line_len)
    t = line_unitvec[0]*pt_scaledvec[0] + line_unitvec[1]*pt_scaledvec[1]
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    nearest = (line_vec[0]*t, line_vec[1]*t)
    dist = distance(nearest,pt_vec)
    return dist

def area_of_tr(ax,ay,bx,by,cx,cy):
    return abs(ax*(by-cy) + bx*(cy-ay) + cx*(ay-by))/2.0

def Vorobyev_sim(polygons, probs, nx, ny, xlim, ylim):

    probs = 1 / (len(polygons)*np.ones(len(polygons))) if probs is None else np.array(probs)

    xx = np.linspace(*xlim,nx)
    yy = np.linspace(*ylim,ny)
    counts = np.zeros((nx, ny))
    for i,x in enumerate(xx):
        for j,y in enumerate(yy):
            for k, poly in enumerate(polygons):
                if poly.is_in(x,y): counts[i,j]+=probs[k]

    avg_area = np.array(probs) @ np.array([poly.area() for poly in polygons])
    t_space = np.linspace(0,1,1000)[1:]
    areas = np.zeros_like(t_space)
    for i,t in enumerate(t_space):
        areas[i] = np.count_nonzero(counts >= t) / (nx*ny) * (xlim[1]-xlim[0]) * (ylim[1]-ylim[0])

    diffs = np.abs(areas-avg_area)
    level = t_space[np.max(np.where(diffs == diffs.min()))]

    expectation = counts.T >= level

    fig, ax = plt.subplots(1,3, figsize=(9,4), gridspec_kw={'width_ratios': [44,2,44]}, constrained_layout=True)
    a0 = ax[0].pcolormesh(xx,yy,counts.T,cmap="hot")
    ax[0].set_title("Pokrývajúca funkcia $p_X=P(u\in X)$")
    plt.colorbar(a0, cax=ax[1])
    ax[2].pcolormesh(xx,yy,expectation, cmap="hot")
    ax[2].set_title("$E_V(X)$")
    plt.show()

def ODA_sim(polygons, probs, nx, ny, xlim, ylim):

    probs = 1 / (len(polygons)*np.ones(len(polygons))) if probs is None else np.array(probs)

    xx = np.linspace(*xlim,nx)
    yy = np.linspace(*ylim,ny)
    avg_dist = np.zeros((nx, ny))
    for i,x in enumerate(xx):
        for j,y in enumerate(yy):
            avg_dist[i,j] = np.array(probs) @ np.array([poly.oriented_dist(x,y) for poly in polygons])

    cmaps_extrem = max(avg_dist.max(), -avg_dist.min())
    fig, ax = plt.subplots(1,3, figsize=(9,4), gridspec_kw={'width_ratios': [44,2,44]}, constrained_layout=True)
    a0 = ax[0].pcolormesh(xx,yy,avg_dist.T,cmap="bwr", vmin=-cmaps_extrem, vmax=cmaps_extrem)
    ax[0].set_title("Priemerná orientovaná vzdialenosť $\overline{d}(u,X)$")
    ax[1] = plt.colorbar(a0, cax=ax[1])
    ax[2].pcolormesh(xx,yy,avg_dist.T<=0, cmap="hot")
    ax[2].set_title("$E_{ODA}(X)$")
    plt.show()

def values_to_set_indicator(canal_values):
    set_indicator = np.zeros((canal_values.shape[0], canal_values.shape[1], 256), dtype=np.uint8)
    for i,j in np.ndindex(canal_values.shape):
        for k in range(canal_values[i,j]+1):
            set_indicator[i,j,k] = 1
    return set_indicator

def set_indicator_to_values(set_indicator):
    return np.array(np.sum(set_indicator, axis=2)-1, dtype=np.uint8)

def imshow(img, ax=plt):
    if isinstance(img, RGBImage): ax.imshow(img.get_values())
    elif isinstance(img, BinaryImage): ax.imshow(img.values, cmap="gray", vmin=0, vmax=1)
    elif isinstance(img, GrayImage): ax.imshow(img.values, cmap="gray", vmin=0, vmax=255)
    elif len(img.shape) == 3: ax.imshow(img)
    elif np.max(img) <= 1: ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    else: ax.imshow(img, cmap="gray", vmin=0, vmax=255)

    ax.axis("off")

def bisection_on_measures(measure_function, expected_measure, tol=0.0001):
    left, right = 0,1
    while (right-left) > tol:
        mid = (left+right)/2
        measure_mid = measure_function(mid)
        if measure_mid == expected_measure:     break
        if measure_mid < expected_measure:      right = mid
        else: left = mid
    else:
        mid = left
    return mid


#################################################
#### CLASSES ####################################
#################################################


class BinaryImage:
    def __init__(self, values):
        self.values = np.array(values, dtype=np.uint8)
        self.height, self.width = self.values.shape
        self.measure = np.sum(self.values)


    def distance_matrix(self, type="oriented", norm=cv.DIST_L2):
        assert np.max(self.values)==1
        pos = cv.distanceTransform(np.abs(self.values-1), norm, cv.DIST_MASK_PRECISE)
        if type=="normal":
            return pos

        assert np.min(self.values)==0
        neg = cv.distanceTransform(self.values, norm, cv.DIST_MASK_PRECISE)
        return pos-neg

class GrayImage:
    def __init__(self, values_or_path):
        if isinstance(values_or_path, str):
            self.values = cv.imread(values_or_path, cv.IMREAD_GRAYSCALE)
        else:
            self.values = np.array(values_or_path, dtype=np.uint8)
        self.height, self.width = self.values.shape
        self.measure = np.sum(self.values) + self.height * self.width

    def set_indicator(self):
        return values_to_set_indicator(self.values)

    def threshold(self, t):
        return BinaryImage(np.array(self.values >= t, dtype=np.uint8))

    def distance_matrix(self, type="oriented"): #l_inf distance
        pos = sc.ndimage.distance_transform_cdt(np.abs(self.set_indicator()-1))
        if type=="normal":
            return np.array(pos, dtype=np.int32)

        neg = sc.ndimage.distance_transform_cdt(self.set_indicator())
        return np.array(pos-neg, dtype=np.int32)


class RGBImage:
    def __init__(self, values_or_path):
        if isinstance(values_or_path, str):
            img = cv.imread(values_or_path, cv.IMREAD_COLOR)
        else:
            img = np.array(values_or_path, dtype=np.uint8)
        self.canals = {col: GrayImage(img[:, :, i]) for i, col in enumerate(["blue", "green", "red"])}
        
    def get_values(self):
        return cv.merge([self.canals["red"].values, self.canals["green"].values, self.canals["blue"].values])

class BinaryRandomSet:
    def __init__(self, images: list[BinaryImage], probs=None):
        self.images = images
        self.height, self.width = self.images[0].height, self.images[0].width
        self.n = len(self.images)
        self.probs = 1 / self.n * np.ones(self.n) if probs is None else np.array(probs)
        assert len(self.probs) == len(self.images) and np.isclose(np.sum(self.probs),1)
        assert all([image.height == self.height and image.width == self.width for image in self.images])

        self.expected_measure = self.probs @ np.array([image.measure for image in self.images])
        self.coverage = np.sum([self.probs[i] * images[i].values for i in range(self.n)], axis=0, dtype=np.float32)

        self._vorobyev = None
        self._vorobyev_choosed_level = None
        self._oda_matrix = None

    def coverage_level_set(self, t):
        level_set = np.array(self.coverage >= t, dtype=np.uint8)
        return BinaryImage(level_set), np.count_nonzero(level_set)

    def vorobyev_expectation(self):
        if self._vorobyev is None:
            measure_function = lambda x:self.coverage_level_set(x)[1]
            self._vorobyev_choosed_level = bisection_on_measures(measure_function, self.expected_measure)
            self._vorobyev = self.coverage_level_set(self._vorobyev_choosed_level)[0]
        return self._vorobyev

    def oda_expectation(self):
        if self._oda_matrix is None:
            oda_matrix = np.zeros((self.height, self.width))
            for i,img in enumerate(self.images):
                oda_matrix += self.probs[i] * img.distance_matrix()
            self._oda_matrix = oda_matrix
        return BinaryImage(self._oda_matrix<=0)

class GrayRandomSet:
    def __init__(self, images: list[GrayImage]):
        # only uniform distribution!
        
        self.images = images
        self.height, self.width = self.images[0].height, self.images[0].width
        self.n = len(self.images)
        assert all([image.height == self.height and image.width == self.width for image in self.images])

        self.expected_measure = np.mean([image.measure for image in self.images])

        dtype = np.uint8 if self.n < 250 else np.uint16
        self.coverage = np.sum([image.set_indicator() for image in self.images], axis=0, dtype=dtype)

        self._vorobyev = None
        self._vorobyev_choosed_level = None
        self._oda_matrix = None

    def coverage_level_set(self, t):
        level_set = np.array(self.coverage >= t, dtype=np.uint8)
        return level_set, np.count_nonzero(level_set)

    def vorobyev_expectation(self):
        if self._vorobyev is None:
            levels = np.arange(self.n)
            measures = np.array([self.coverage_level_set(t)[1] for t in levels])
            self._vorobyev_choosed_level = levels[np.max(np.where((self.expected_measure - measures)<=0))]
            self._vorobyev = set_indicator_to_values(self.coverage_level_set(self._vorobyev_choosed_level)[0])
        return self._vorobyev

    def oda_expectation(self):
        if self._oda_matrix is None:
            oda_matrix = np.zeros((self.height, self.width, 256), dtype=np.float32)
            for i,img in enumerate(self.images):
                oda_matrix += img.distance_matrix() / self.n
            self._oda_matrix = oda_matrix
        return set_indicator_to_values(self._oda_matrix<=0)

class RGBRandomSet:
    def __init__(self, images: list[RGBImage]):
        # only uniform distribution!
        
        self.images = images
        canal_names = images[0].canals.keys()
        assert all([img.canals.keys() == canal_names for img in self.images])

        self.RCS = {c_name: GrayRandomSet([image.canals[c_name] for image in self.images]) for c_name in canal_names}

        self._vorobyev = None
        self._oda = None

    def vorobyev_expectation(self):
        if self._vorobyev is None:
            self._vorobyev = cv.merge([self.RCS[canal].vorobyev_expectation() for canal in ["red", "green", "blue"]])
        return self._vorobyev

    def oda_expectation(self):
        if self._oda is None:
            self._oda = cv.merge([self.RCS[canal].oda_expectation() for canal in ["red", "green", "blue"]])
        return self._oda

class Triangle:

    def __init__(self, n, start_angle=-np.pi/2):
        self.n = n
        angles = np.linspace(start_angle, start_angle+2*np.pi, n, endpoint=False)
        pts = np.column_stack([np.cos(angles), np.sin(angles)])
        first_idx = np.lexsort((pts[:,0], pts[:,1]))[0]
        self.points = np.concatenate([pts[first_idx:], pts[:first_idx],[pts[first_idx]]])

    def is_in(self,x,y):
        A1 = area_of_tr(x, y, self.points[1,0], self.points[1,1], self.points[2,0], self.points[2,1])
        A2 = area_of_tr(self.points[0,0], self.points[0,1], x, y, self.points[2,0], self.points[2,1])
        A3 = area_of_tr(self.points[0,0], self.points[0,1], self.points[1,0], self.points[1,1], x, y)
        return np.isclose(self.area() - (A1 + A2 + A3),0)

    def area(self):
        return 1/2 * self.n * np.sin(2*np.pi/self.n)

    def oriented_dist(self,x,y):
        srt = sorted(self.points[:-1], key=lambda pt:distance([x,y],pt))
        start = srt[0]
        end = srt[1]
        dst = dist_to_segment([x,y], start, end)
        return -dst if self.is_in(x,y) else dst


class Tetiva:
    def __init__(self,fixed=False):
        start, end = np.random.uniform(0, 2*np.pi, 2)
        if fixed:
            start = -np.pi/2
        pts = np.array([[np.cos(start),np.sin(start)], [np.cos(end), np.sin(end)]])
        first_idx = np.lexsort((pts[:,0], pts[:,1]))[0]
        self.points = np.concatenate([pts[first_idx:], pts[:first_idx],[pts[first_idx]]])

    def oriented_dist(self,x,y):
        return dist_to_segment([x,y],self.points[0],self.points[1])

class Rect:
    def __init__(self):
        angle = np.random.uniform(0, np.pi/2)
        self.x, self.y = np.cos(angle), np.sin(angle)
        self.points = np.array([[0,0], [self.x,0], [self.x,self.y], [0,self.y], [0,0]])

    def is_in(self,x,y):
        return (0 <= x <= self.x) and (0 <= y <= self.y)

    def area(self, p=1):
        return (self.x*self.y) ** p

    def oriented_dist(self,x,y):
        if self.is_in(x,y):
            return -min([x, y, self.x-x, self.y-y])
        if 0 <= x <= self.x:
            return np.min(np.abs([y,y-self.y]))
        if 0 <= y <= self.y:
            return np.min(np.abs([x, x-self.x]))
        return min([distance([x,y],pt) for pt in self.points[:-1]])


class Thresholder:
    def __init__(self, img):
        self.img = img
        self.min_value = img.values.min() + 1
        max_value = img.values.max()
        assert self.min_value < max_value
        counts = np.array([np.sum(self.img.values == val) for val in range(256)])
        self.rel_counts = (counts / np.sum(counts[self.min_value:max_value + 1]))[self.min_value:max_value + 1]
        self.thresh_images = np.array([self.img.threshold(val) for val in range(self.min_value, max_value + 1)])
        self.RS = {"uniform": BinaryRandomSet(self.thresh_images),
                   "weighted": BinaryRandomSet(self.thresh_images, probs=self.rel_counts)}

    def find_level(self, thresholded_image):
        for t, img in enumerate(self.thresh_images, start=self.min_value):
            if img.measure == thresholded_image.measure:
                return t

    def find_ODA_nearest(self, type):
        idx = np.argmin([np.sum(self.RS[type].oda_expectation().values != img.values) for img in self.thresh_images])
        return self.thresh_images[idx]

    def vorobyev_threshold(self, type):
        output = self.RS[type].vorobyev_expectation()
        return output

    def distance_threshold(self, type, distance_type="oriented", distance_norm=cv.DIST_L2, minimalization_norm="inf"):

        distance_matrix = np.zeros((self.img.height, self.img.width))
        for i, img in enumerate(self.thresh_images):
            distance_matrix += self.RS[type].probs[i] * img.distance_matrix(type=distance_type, norm=distance_norm)

        min_diff = np.inf
        min_level = 0
        for i, img in enumerate(self.thresh_images, start=self.min_value):
            if minimalization_norm == "inf":
                diff = np.max(np.abs(img.distance_matrix(type=distance_type, norm=distance_norm) - distance_matrix))
            else:
                diff = np.sum(np.square(img.distance_matrix(type=distance_type, norm=distance_norm) - distance_matrix))
            if diff < min_diff:
                min_diff = diff
                min_level = i
        return self.thresh_images[min_level]

    def oda_expectation(self, type):
        return self.RS[type].oda_expectation()