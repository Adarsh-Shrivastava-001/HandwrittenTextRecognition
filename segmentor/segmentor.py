import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
from PIL import Image

req_col=250
wht= 250

"""find major color"""
def find_start_row(rowpix):
    global l,w,arr,wht,erq_col
    for i in range(rowpix,l):
        if np.sum(arr[i])/(3*w)<req_col:
            return i
    
    return -1
    print("error")
    
def find_end_row(rowpix):
    global l,w,arr,wht,erq_col
    for i in range(rowpix,l):
        if np.sum(arr[i])/(3*w)>=wht:
            return i
    return -1
    print("error")
    
def start_col(startline,endline,colpix):
    global l,w,arr,wht,erq_col
    for i in range(colpix,w):
        if np.sum(arr[startline:endline,i:i+3,:])/(9*(endline-startline))<req_col:
            return i
    return -1
        

def end_col(startline,endline,colpix):
    global l,w,arr,wht,erq_col
    for i in range(colpix,w):
        if np.sum(arr[startline:endline,i:i+3,:])/(9*(endline-startline))>=wht:
            return i
    return -1
        

img=Image.open('hand2.png')

arr=np.array(img)
arr=arr[:,:,0:3]

for i in range(len(arr)):
    for j in range(len(arr[0])):
        if np.mean(arr[i][j])>200:
            arr[i][j]=[255,255,255]
        else:
            arr[i][j]=[0,0,0]
            

l=len(arr)
w=len(arr[0])



lines=[]

start_row=0

while True:
    start=find_start_row(start_row)
    if start==-1:
        break
    end=find_end_row(start)
    if  end==-1:
        break
    lines.append([arr[start-1:end+1,:,:],start-1,end+1])
    start_row=end
    
words=[]
for i in lines:
    start_cl=0
    word=[]
    while True:
        start=start_col(i[1],i[2],start_cl)
        if start==-1:
            break
        end=end_col(i[1],i[2],start)
        if  end==-1:
            break
        word.append(arr[i[1]:i[2],start-1:end,:])
        start_cl=end
    words.append(word)
    
    

    
plt.imshow(arr)


for i in words:
    for j in i:
        plt.imshow(j)
        plt.show()

