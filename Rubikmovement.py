oo#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 19:37:19 2018

@author: kaustabh
"""




#making a single list representing all sides in a serial fashion

r = ['w1','w2','w3','w4','w5','w6','w7','w8','w9','b1','b2','b3','r1','r2','r3','g1','g2','g3','o1','o2','o3','b4','b5','b6','r4','r5','r6','g4','g5','g6','o4','o5','o6','b7','b8','b9','r7','r8','r9','g7','g8','g9','o7','o8','o9', 'y1','y2','y3','y4','y5','y6','y7','y8','y9']

real = r[:] 


# cube is the list and we can't use random.shuffle because certain colors stay together
    


def shift(r,L): #r is cube list ; L is 12 element array to be shifted
      
    t1 = r[L[0]]
    t2 = r[L[1]]
    t3 = r[L[2]]
    
    r[L[0]] , r[L[3]] = r[L[3]] , r[L[0]]
    r[L[1]] , r[L[4]] = r[L[4]] , r[L[1]]
    r[L[2]] , r[L[5]] = r[L[5]] , r[L[2]]
     
    r[L[3]] , r[L[6]] = r[L[6]] , r[L[3]]
    r[L[4]] , r[L[7]] = r[L[7]] , r[L[4]]
    r[L[5]] , r[L[8]] = r[L[8]] , r[L[5]]
    
    r[L[6]] , r[L[9]] = r[L[9]] , r[L[6]]
    r[L[7]] , r[L[10]] = r[L[10]] , r[L[7]]
    r[L[8]] , r[L[11]] = r[L[11]] , r[L[8]]
       
    r[L[9]] = t1
    r[L[10]] = t2
    r[L[11]] = t3
    
    return r

def rotate(r, L): #a face will also rotate
    
    t1 = r[L[0]]
    t2 = r[L[7]]
    
    r[L[0]] , r[L[6]] = r[L[6]] , r[L[0]]
    r[L[7]] , r[L[5]] = r[L[5]] , r[L[7]]
    r[L[6]] , r[L[4]] = r[L[4]] , r[L[6]]
    r[L[3]] , r[L[5]] = r[L[5]] , r[L[3]]
    r[L[2]] , r[L[4]] = r[L[4]] , r[L[2]]
    r[L[1]] , r[L[3]] = r[L[3]] , r[L[1]]
    r[L[2]] = t1
    r[L[1]] = t2
    
    return 
    
    

def right_c(r):
    
    L = [33,34,35,36,37,38,39,40,41,42,43,44]
    rt = [45,48,51,52,53,50,47,46]
    rotate(r,rt)
    return shift(r, L)

def right_ac(r):
    
    L = [33,34,35,36,37,38,39,40,41,42,43,44]
    L.reverse()
    rt = [45,48,51,52,53,50,47,46]
    rt.reverse()
    rotate(r,rt)
    return shift(r, L)

def left_ac(r):
    
    rt = [0,1,2,5,8,7,6,3]
    rotate(r,rt)
    L = [9,10,11,12,13,14,15,16,17,18,19,20]
    return shift(r, L)

def left_c(r):
    
    rt = [0,1,2,5,8,7,6,3]
    rt.reverse()
    rotate(r,rt)
    L = [9,10,11,12,13,14,15,16,17,18,19,20]
    L.reverse()
    return shift(r, L)

def up_c(r):
    
    rt = [9,21,33,34,35,23,11,10]
    rotate(r, rt)
    L = [0,3,6,12,24,36,45,48,51,18,30,42]
    return shift(r, L)

def up_ac(r):
    
    rt = [9,21,33,34,35,23,11,10]
    rt.reverse()
    rotate(r, rt)
    L = [0,3,6,12,24,36,45,48,51,18,30,42]
    L.reverse()
    return shift(r, L)

def down_ac(r):
    
    rt = [15,16,17,29,41,40,39,27]
    rotate(r,rt)
    L = [2,5,8,14,26,38,47,50,53,20,32,44]
    return shift(r, L)

def down_c(r):
    
    rt = [15,16,17,29,41,40,39,27]
    rt.reverse()
    rotate(r,rt)
    L = [2,5,8,14,26,38,47,50,53,20,32,44]
    L.reverse()
    return shift(r, L)

def front_c(r):
    
    rt = [12,24,36,37,38,26,14,13]
    rotate(r,rt)
    L = [11,23,35,45,46,47,39,27,15,8,7,6]
    return shift(r, L)

def front_ac(r):

    rt = [12,24,36,37,38,26,14,13]
    rt.reverse()
    rotate(r,rt)
    L = [11,23,35,45,46,47,39,27,15,8,7,6]
    L.reverse()
    return shift(r, L)

def back_c(r):
    
    rt= [\]
    rotate(r,rt)
    L = [0,3,6,9,21,33,45,48,51,15,27,39]
    return shift(r, L)

def back_ac(r):
    
    rt= [18,30,42,43,44,32,20,19]
    rt.reverse()
    rotate(r,rt)
    L = [0,3,6,9,21,33,45,48,51,15,27,39]
    L.reverse()
    return shift(r, L)

def shuffle(r):
    
    import random
    for i in range(random.randint(17,32)):
        random.choice([right_c(r),left_c(r),up_c(r),down_c(r),front_c(r),back_c(r),back_ac(r),right_ac(r),left_ac(r),up_ac(r),front_ac(r),down_ac(r)])
    return r

def reset():
    
    return real[:]

#print(shuffle(r))



    






    
    



