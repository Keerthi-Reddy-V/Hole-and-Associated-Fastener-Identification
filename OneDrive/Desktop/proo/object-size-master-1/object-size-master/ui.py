from tkinter import *
from PIL import Image,ImageTk
import os
win=Tk()
win.title("two way")
win.geometry("1500x750")
img=Image.open("aa.jpg")
img=img.resize((1500,750))
bg=ImageTk.PhotoImage(img)

lbl=Label(win,image=bg)
lbl.place(x=0,y=0)

def func():
    os.system("python FINAL.py")
    
label=Label(win,text="Diameter of the Hole Detector",font=("times",32,"bold underline"),bg="black",fg="white")
label.place(x=590,y=220)

labelb=Button(win,text=" Start Checking",command=func,font=("times",24,"bold"),bg="brown",fg="white",height=3,width=20)
labelb.place(x=550,y=420)


win.mainloop()
