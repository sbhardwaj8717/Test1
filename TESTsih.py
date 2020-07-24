import tkinter as tk
from tkinter import Message, Text
import cv2
import os
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font

window = tk.Tk()
# helv36 = tk.Font(family='Helvetica', size=36, weight='bold')
window.title("CamAttendance")

dialog_title = 'QUIT'
dialog_text = 'Are you sure?'
# answer = messagebox.askquestion(dialog_title, dialog_text)

# window.geometry('1280x720')
window.configure(background='white')

# window.attributes('-fullscreen', True)

window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

# path = "profile.jpg"

# Creates a Tkinter-compatible photo image, which can be used everywhere Tkinter expects an image object.
# img = ImageTk.PhotoImage(Image.open(path))

# The Label widget is a standard Tkinter widget used to display a text or image on the screen.
# panel = tk.Label(window, image = img)


# panel.pack(side = "left", fill = "y", expand = "no")

# cv_img = cv2.imread("img541.jpg")
# x, y, no_channels = cv_img.shape
# canvas = tk.Canvas(window, width = x, height =y)
# canvas.pack(side="left")
# photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv_img))
# Add a PhotoImage to the Canvas
# canvas.create_image(0, 0, image=photo, anchor=tk.NW)

# msg = Message(window, text='Hello, world!')

# Font is a tuple of (font_family, size_in_points, style_modifier_string)


message = tk.Label(window, text="Face-Recognition-Based-Attendance-Management-System", bg="Black", fg="white", width=50,
                   height=3, font=('times', 30, 'italic bold underline'))

message.place(x=100, y=20)

lbl = tk.Label(window, text="Enter ID", width=20, height=2, fg="white", bg="red", font=('times', 15, ' bold '))
lbl.place(x=250, y=200)

txt = tk.Entry(window, width=20, bg="red", fg="white", font=('times', 15, ' bold '))
txt.place(x=600, y=215)

lbl2 = tk.Label(window, text="Enter Name", width=20, fg="white", bg="red", height=2, font=('times', 15, ' bold '))
lbl2.place(x=250, y=300)

txt2 = tk.Entry(window, width=20, bg="red", fg="white", font=('times', 15, ' bold '))
txt2.place(x=600, y=315)

lbl3 = tk.Label(window, text="Notification : ", width=20, fg="white", bg="red", height=2,
                font=('times', 15, ' bold underline '))
lbl3.place(x=400, y=400)

message = tk.Label(window, text="", bg="red", fg="white", width=30, height=2, activebackground="yellow",
                   font=('times', 15, ' bold '))
message.place(x=700, y=400)

lbl3 = tk.Label(window, text="Attendance : ", width=20, fg="white", bg="red", height=2,
                font=('times', 15, ' bold  underline'))
lbl3.place(x=400, y=650)

message2 = tk.Label(window, text="", fg="white", bg="red", activeforeground="green", width=30, height=2,
                    font=('times', 15, ' bold '))
message2.place(x=700, y=650)


def clear():
    txt.delete(0, 'end')
    res = ""
    message.configure(text=res)


def clear2():
    txt2.delete(0, 'end')
    res = ""
    message.configure(text=res)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False

def TakeImages():
    Id = (txt.get())
    name = (txt2.get())
    if (is_number(Id) and name.isalpha()):
        # Read a Video Stream and Display It

        # Camera Object
        cam = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier('C:/Users/user/PycharmProjects/E-Yantra/haarcascade_frontalface_alt2.xml')
        face_data = []
        cnt = 0

        #user_name = input("enter your name")
        user_name=name
        #ctr=0
        while True:
            ret, frame = cam.read()
            if ret == False:
                print("Something Went Wrong!")
                continue

            #key_pressed = cv2.waitKey(1) & 0xFF  # Bitmasking to get last 8 bits
            #if key_pressed == ord('q'):  # ord-->ASCII Value(8 bit)
            #    break

            faces = face_cascade.detectMultiScale(frame, 1.3, 5)
            # print(faces)
            if (len(faces) == 0):
                cv2.imshow("Video", frame)
                continue
            for face in faces:
                x, y, w, h = face
                face_section = frame[y - 10:y + h + 10, x - 10:x + w + 10]
                face_section = cv2.resize(face_section, (100, 100))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                #ctr=ctr+1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.imwrite("C:/Users/user/PycharmProjects/E-Yantra/TrainingImage/ " + name + "." + Id + '.' + str(len(face_data)) + ".jpg", gray[y:y + h, x:x + w])
                if cnt % 10 == 0:
                    print("Taking picture ", int(cnt / 10))
                    face_data.append(face_section)
                cnt += 1

            cv2.imshow("Video", frame)
            cv2.imshow("Video Gray", face_section)

            key_pressed = cv2.waitKey(1) & 0xFF  # Bitmasking to get last 8 bits
            if key_pressed == ord('q'):  # ord-->ASCII Value(8 bit)
                break
            elif len(face_data)>30:
                break

        # Save the face data in a numpy file
        print("Total Faces", len(face_data))
        face_data = np.array(face_data)
        face_data = face_data.reshape((face_data.shape[0], -1))

        np.save("C:/Users/user/PycharmProjects/E-Yantra/FaceData/" + user_name + ".npy", face_data)
        print("Saved at FaceData/" + user_name + ".npy")
        print(face_data.shape)
        cam.release()
        cv2.destroyAllWindows()

        res = "Images Saved for ID : " + Id + " Name : " + name
        row = [Id, name]
        with open('C:/Users/user/PycharmProjects/E-Yantra/StudentDetails/StudentDetails.csv', 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message.configure(text=res)
    else:
        if (is_number(Id)):
            res = "Enter Alphabetical Name"
            message.configure(text=res)
        if (name.isalpha()):
            res = "Enter Numeric Id"
            message.configure(text=res)

def TrainAndTrackImages():
    def distance(v1, v2):
        # Eucledian
        return np.sqrt(((v1 - v2) ** 2).sum())

    def knn(train, test, k=5):
        dist = []

        for i in range(train.shape[0]):
            # Get the vector and label
            ix = train[i, :-1]
            iy = train[i, -1]
            # Compute the distance from test point
            d = distance(test, ix)
            dist.append([d, iy])
        # Sort based on distance and get top k
        dk = sorted(dist, key=lambda x: x[0])[:k]
        # Retrieve only the labels
        labels = np.array(dk)[:, -1]

        # Get frequencies of each label
        output = np.unique(labels, return_counts=True)
        # Find max frequency and corresponding label
        index = np.argmax(output[1])
        return output[0][index]

    ################################

    #cam = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier("C:/Users/user/PycharmProjects/E-Yantra/haarcascade_frontalface_alt2.xml")
    cam=cv2.VideoCapture(0)
    dataset_path = "C:/Users/user/PycharmProjects/E-Yantra/FaceData/"
    labels = []
    class_id = 0
    names = {}
    face_data = []
    labels = []

    for fx in os.listdir(dataset_path):
        if fx.endswith(".npy"):
            names[class_id] = fx[:-4]
            print("Loading file ", fx)
            data_item = np.load(dataset_path + fx)
            face_data.append(data_item)

            # Create Labels
            target = class_id * np.ones((data_item.shape[0],))
            labels.append(target)
            class_id += 1

    X = np.concatenate(face_data, axis=0)
    Y = np.concatenate(labels, axis=0)

    print(X.shape)
    print(Y.shape)

    # Training Set
    trainset = np.concatenate((X, Y.reshape(-1, 1)), axis=1)

    #recognizer = cv2.face.LBPHFaceRecognizer_create()
    #recognizer.read("C:/Users/user/PycharmProjects/E-Yantra/Trainner.yml")
    #face_cascade = cv2.CascadeClassifier("C:/Users/user/PycharmProjects/E-Yantra/haarcascade_frontalface_alt2.xml")
    df = pd.read_csv("C:/Users/user/PycharmProjects/E-Yantra/StudentDetails/StudentDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', 'Name', 'Date', 'Time']
    attendance = pd.DataFrame(columns=col_names)

    while True:

        #ret,frame = cam.read()
        #im = cv2.flip(im, -1)
        #graye = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #faces = face_cascade.detectMultiScale(frame,1.3,5)
        #for face in faces:
            #x,y,w,h=face
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (225, 0, 0), 2)
        Id = (txt.get())
            #Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
            #if (conf < 50):
        ts = time.time()
        date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
        aa = df.loc[df['Id'] == Id]['Name'].values
        tt = str(Id) + "-" + aa
        attendance.loc[len(attendance)] = [Id, aa, date, timeStamp]
            #else:
                #Id = 'Unknown'
                #tt = str(Id)
            #if (conf > 75):
                #noOfFile = len(os.listdir("C:/Users/user/PycharmProjects/E-Yantra/ImagesUnknown")) + 1
                #cv2.imwrite("C:/Users/user/PycharmProjects/E-Yantra/ImagesUnknown/Image" + str(noOfFile) + ".jpg",im[y:y + h, x:x + w])
        cv2.putText(frame, str(tt), (x, y + h), font, 1, (255, 255, 255), 2)
        attendance = attendance.drop_duplicates(subset=['Id'], keep='first')


        #if(len(faces)==0):
            #cv2.imshow("image", frame)
            #continue
        #if (cv2.waitKey(1) == ord('q')):
            #break
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour, Minute, Second = timeStamp.split(":")
    fileName = "C:/Users/user/PycharmProjects/E-Yantra/Attendance/Attendance_" + date + "_" + Hour + "-" + Minute + "-" + Second + ".csv"
    attendance.to_csv(fileName, index=False)
    cam.release()
    cv2.destroyAllWindows()
    # print(attendance)
    res = attendance
    message2.configure(text=res)
    while True:
        ret, frame = cam.read()
        if ret == False:
            print("Something Went Wrong!")
            continue

        key_pressed = cv2.waitKey(1) & 0xFF  # Bitmasking to get last 8 bits
        if key_pressed == ord('q'):  # ord-->ASCII Value(8 bit)
            break
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        if (len(faces) == 0):
            cv2.imshow("Faces Detected", frame)
            continue

        for face in faces:
            x, y, w, h = face
            face_section = frame[y - 10:y + h + 10, x - 10:x + w + 10]
            face_section = cv2.resize(face_section, (100, 100))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

            pred = knn(trainset, face_section.flatten())
            name = names[int(pred)]
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow("Faces Detected", frame)

    cam.release()
    cv2.destroyAllWindows()

clearButton = tk.Button(window, text="Clear", command=clear, fg="white", bg="red", width=20, height=2,
                        activebackground="Red", font=('times', 15, ' bold '))
clearButton.place(x=950, y=200)
clearButton2 = tk.Button(window, text="Clear", command=clear2, fg="white", bg="red", width=20, height=2,
                         activebackground="Red", font=('times', 15, ' bold '))
clearButton2.place(x=950, y=300)
takeImg = tk.Button(window, text="Take Images", command=TakeImages, fg="white", bg="red", width=20, height=3,
                    activebackground="Red", font=('times', 15, ' bold '))
takeImg.place(x=300, y=500)
trainImg = tk.Button(window, text="Train And Track Images", command=TrainAndTrackImages, fg="white", bg="red", width=20, height=3,
                     activebackground="Red", font=('times', 15, ' bold '))
trainImg.place(x=600, y=500)
#trackImg = tk.Button(window, text="Track Images", command=TrackImages, fg="white", bg="red", width=20, height=3,
 #                    activebackground="Red", font=('times', 15, ' bold '))
#trackImg.place(x=800, y=500)
quitWindow = tk.Button(window, text="Quit", command=window.destroy, fg="white", bg="red", width=20, height=3,
                       activebackground="Red", font=('times', 15, ' bold '))
quitWindow.place(x=900, y=500)
copyWrite = tk.Text(window, background=window.cget("background"), borderwidth=0,
                    font=('times', 30, 'italic bold underline'))
window.mainloop()

