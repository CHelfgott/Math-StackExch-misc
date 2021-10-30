# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 14:39:51 2021
Stack Exchange thread: https://math.stackexchange.com/questions/4283519/how-many-connected-components-in-this-random-graph

@author: Craig
"""
from matplotlib import collections as mc
import numpy as np
import pylab as pl
import random
from scipy import optimize

NUM_TESTS = 100
TEST_SIZE = 400

def build_points(size):
  points = np.zeros([size, 2], dtype=float)
  # Generate TEST_SIZE random points in [0,1]^2
  for j in range(size):
    for k in range(2):
      points[j][k] = random.random()
  return points

# points_* is an N-by-2 array of coordinates representing N pts in [0,1]^2
def get_dist_matrix(points_a, points_b):
  delta = np.expand_dims(points_a, axis=1) - np.expand_dims(points_b, axis=0)
  dsq = np.sum(delta * delta, axis=2)
  return np.sqrt(dsq)

# determines whether 3 points in the plane are oriented in counter-clockwise fashion
def ccw(point_a, point_b, point_c):
  return bool((point_c[1] - point_a[1]) * (point_b[0] - point_a[0]) > 
              (point_b[1] - point_a[1]) * (point_c[0] - point_a[0])) 
  
# Here segment_* is a 2-tuple of 2-dimensional arrays.
def do_intersect(segment_a, segment_b):
  if ccw(segment_a[0], segment_b[0], segment_b[1]) == \
    ccw(segment_a[1], segment_b[0], segment_b[1]):
      return False
  if ccw(segment_a[0], segment_a[1], segment_b[0]) == \
    ccw(segment_a[1], segment_a[1], segment_b[1]):
      return False
  return True

def print_bucketed_histogram(hist_dict, num_buckets):
  total_value = 0
  for val in hist_dict.values():
    total_value += val
  target_value = total_value / num_buckets
  
  start_bucket = None
  bucket_total = 0
  for key, value in sorted(hist_dict.items()):
    if start_bucket is None:
      start_bucket = key
    bucket_total += value
    if bucket_total > target_value:
      if start_bucket == key:
        print(f'{start_bucket}: {bucket_total}')
      else:
        print(f'{start_bucket}-{key}: {bucket_total}')
      bucket_total = 0
      start_bucket = None
  if start_bucket:
      if start_bucket == key:
        print(f'{start_bucket}: {bucket_total}')
      else:
        print(f'{start_bucket}-{key}: {bucket_total}')    
  

def main():
  colors = [(1,0,0,1),(0,1,0,1),(0,0,1,1)] * TEST_SIZE
  colors = np.array(colors)
  lines = []
  
  comp_sizes = {}
  comp_cnts = {}
  xsect_cnts = {}
  self_xsect_cnts = {}
  for _ in range(NUM_TESTS):
    red_points = build_points(TEST_SIZE)
    blue_points = build_points(TEST_SIZE)
    green_points = build_points(TEST_SIZE)

    # Build the distance matrices for all pairs of colors
    rg_dist = get_dist_matrix(red_points, green_points)
    gb_dist = get_dist_matrix(green_points, blue_points)
    br_dist = get_dist_matrix(blue_points, red_points)
  
    # Use those distances to get minimum-distance matchings for all pairs of colors
    _, rg_col_ind = optimize.linear_sum_assignment(rg_dist)
    _, gb_col_ind = optimize.linear_sum_assignment(gb_dist)
    _, br_col_ind = optimize.linear_sum_assignment(br_dist)
    
    # Capture the first test for display 
    if not lines:
      for i in range(TEST_SIZE):
        g_target = rg_col_ind[i]
        b_target = gb_col_ind[i]
        r_target = br_col_ind[i]
        lines.extend([
          [red_points[i,:], green_points[g_target,:]],
          [green_points[i,:], blue_points[b_target,:]],
          [blue_points[i,:], red_points[r_target,:]]          
        ])
      
    
    # Now start assigning vertices to components
    which_component = -1 * np.ones([TEST_SIZE, 3], dtype=int)
    component_cnt = 0
    for red_idx in range(TEST_SIZE):
      if which_component[red_idx, 0] >= 0:
        continue
      red_ptr = red_idx
      component_size = 0
      while which_component[red_ptr, 0] == -1:
        which_component[red_ptr, 0] = component_cnt
        green_ptr = rg_col_ind[red_ptr]
        which_component[green_ptr, 1] = component_cnt
        blue_ptr = gb_col_ind[green_ptr]
        which_component[blue_ptr, 2] = component_cnt
        red_ptr = br_col_ind[blue_ptr]
        component_size += 3
      comp_sizes[component_size] = comp_sizes.get(component_size, 0) + 1
      component_cnt += 1
    
    comp_cnts[component_cnt] = comp_cnts.get(component_cnt, 0) + 1

    # We now have a histogram of component sizes and counts.
    # Next to do intersection and self-intersection counts.
    xsect_count = 0
    self_xsect_count = 0
    
    # Note that all intersections must be between two edges that share exactly
    # one color.
    for i in range(TEST_SIZE):
      for j in range(TEST_SIZE):
        # First test RG/GB
        g_target = rg_col_ind[i]
        b_target = gb_col_ind[j]
        if g_target != j:
          if do_intersect((red_points[i,:], green_points[g_target,:]),
                          (green_points[j,:], blue_points[b_target,:])):
            xsect_count += 1
            if which_component[i,0] == which_component[j,1]:
              self_xsect_count += 1
 
        # Then GB/BR
        r_target = br_col_ind[i]
        if b_target != i:
          if do_intersect((blue_points[i,:], red_points[r_target,:]),
                          (green_points[j,:], blue_points[b_target,:])):
            xsect_count += 1
            if which_component[i,2] == which_component[j,1]:
              self_xsect_count += 1
              
        # Now BR/RG
        g_target = rg_col_ind[j]
        if r_target != j:
          if do_intersect((blue_points[i,:], red_points[r_target,:]),
                          (red_points[j,:], green_points[g_target,:])):
            xsect_count += 1
            if which_component[i,2] == which_component[j,0]:
              self_xsect_count += 1
              
    xsect_cnts[xsect_count] = xsect_cnts.get(xsect_count, 0) + 1
    self_xsect_cnts[self_xsect_count] = self_xsect_cnts.get(self_xsect_count, 0) + 1

  lc = mc.LineCollection(lines, colors=colors, linewidths=2)
  fig, ax = pl.subplots()
  ax.add_collection(lc)
  ax.autoscale()
  ax.margins(0.01)

  print(f'With {NUM_TESTS} tests on instances of size {TEST_SIZE}, we find:')
  print('Component count histogram: ')
#  print(sorted(comp_cnts.items()))
  print_bucketed_histogram(comp_cnts, 20)
  print('Component size histogram: ')
#  print(sorted(comp_sizes.items()))  
  print_bucketed_histogram(comp_sizes, 20)
  print('Intersection number histogram:')
#  print(sorted(xsect_cnts.items()))
  print_bucketed_histogram(xsect_cnts, 20)
  print('Component self-intersection number histogram:')
#  print(sorted(self_xsect_cnts.items()))
  print_bucketed_histogram(self_xsect_cnts, 20)
  
if __name__ == '__main__':
  main()  