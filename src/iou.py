import numpy as np
import glob
import os
import cv2
import math

def euclidean_distance(point1, point2):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))

def calculate_distances(points):
    distances = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = euclidean_distance(points[i], points[j])
            distances.append((i, j, dist))
    return distances
def IoU(box1, box2):
    # box = (x1, y1, x2, y2)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # obtain x1, y1, x2, y2 of the intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # compute the width and height of the intersection
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)

    inter = w * h
    iou = inter / (box1_area + box2_area - inter)
    return iou

def calculate_median(numbers):
    if not numbers:
        raise ValueError("The list is empty")
    
    numbers.sort()
    n = len(numbers)
    mid = n // 2

    if n % 2 == 0:
        median = (numbers[mid - 1] + numbers[mid]) / 2
    else:
        median = numbers[mid]

    return median


file_path = '/home/work/YaiBawi/sohyun/YaiBawi/YOLOX_outputs/reid-rsKM/track_vis/2024_05_21_10_51_36.txt'

with open(file_path,'r') as f:
    lines = f.readlines()

n = 0
obj = []
ious = []
while True:
    if n >=1190:
        break
    next_5_lines = lines[n:n+5]
    for i, line in enumerate(next_5_lines):
        line = line.split(',')
        x = float(line[2])
        y = float(line[3])
        w = float(line[4])
        h = float(line[5])
        obj.append([x, y, w, h])
    dist = calculate_distances(obj)
    for (i,j,d) in dist:
        if d < 190:
            x1 = obj[i][0]
            x2 = obj[i][1]
            y1 = obj[i][0]+obj[i][2]
            y2 = obj[i][1]+obj[i][3]
            obj1 = [x1, x2, y1, y2]
            x1 = obj[j][0]
            x2 = obj[j][1]
            y1 = obj[j][0]+obj[j][2]
            y2 = obj[j][1]+obj[j][3]
            obj2 = [x1, x2, y1, y2]
            iou = IoU(obj1, obj2)
            if iou > 0:
                ious.append(iou)
                print("Frame: ", int(n/5))
                print("IOU: ", iou)

    obj = []
    n = n+5

mean = sum(ious)/len(ious)
med = calculate_median(ious)
print("IOU Mean: ", mean)
print("IOU Median: ", med)
print("IOU min: ", min(ious))

