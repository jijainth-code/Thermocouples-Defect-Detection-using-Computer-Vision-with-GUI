import os
import time
import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import tkinter as tk
import cv2
from PIL import Image, ImageTk , ImageDraw , ImageFont
from tkinter import ttk



model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp4/weights/last.pt', force_reload=True)

dict_value ={
    0: "dog",
    1: "person",
    2: "car",
    3: "tv",
    4: "car",
    5: "meatballs",
    6: "marinara sauce",
    7: "tomato soup",
    8: "chicken noodle soup",
    9: "french onion soup",
    10: "chicken breast",
    11: "ribs",
    12: "pulled pork",
    13: "hamburger",
    14: "cavity",
    15: "bad_weld",
    16: "good_weld"
}

class VideoStreamApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.configure(bg="#1c1c1c")
        
        self.video_capture = cv2.VideoCapture(0)  # Use the default camera (index 0)

        self.vid_canvas = tk.Canvas(window, width=500, height=400, bg="#1c1c1c")
        self.vid_canvas.pack()

        self.main_frame = tk.Frame(window, bg="#1c1c1c")
        self.main_frame.pack(fill="both", expand=True)

        self.text_frame = tk.Frame(self.main_frame, bg="#1c1c1c", width=500, height=self.video_capture.get(4), padx=5, pady=5)
        self.text_frame.pack(side="left", fill="both")

        self.text_label = tk.Label(self.text_frame, text="Click Capture for Result ! ", font=("Helvetica", 12), padx=5, pady=5, bg="#1c1c1c", fg="white", bd=2, relief="solid" , highlightbackground="white", highlightcolor="white")
        self.text_label.pack()

        self.capture_button = ttk.Button(self.text_frame, text="Capture", command=self.capture_photo, style="C.TButton")
        self.capture_button.pack(pady=10)  # Add padding to the button

        #New buttons for Good and Bad stripping 

        self.capture_button2 = ttk.Button(self.text_frame, text="bad Stripped", command=self.capture_photo_badStrip, style="C.TButton")
        self.capture_button2.pack(pady=10)  # Add padding to the button

        self.capture_button3 = ttk.Button(self.text_frame, text="good Stripped", command=self.capture_photo_goodStrip, style="C.TButton")
        self.capture_button3.pack(pady=10)  # Add padding to the button

        #New buttons for Good and Bad weld 

        self.capture_button2 = ttk.Button(self.text_frame, text="Good Weld", command=self.capture_photo_goodWeld, style="C.TButton")
        self.capture_button2.pack(pady=10)  # Add padding to the button

        self.capture_button2 = ttk.Button(self.text_frame, text="Bad Weld", command=self.capture_photo_badWeld, style="C.TButton")
        self.capture_button2.pack(pady=10)  # Add padding to the button

        
        

        self.photo_frame = tk.Frame(self.main_frame, bg="#1c1c1c", width=500, height=400, padx=5, pady=5)
        self.photo_frame.pack(side="right", fill="both")

        self.raw_capture_directory = "Capture_Data\capture_raw"
        if not os.path.exists(self.raw_capture_directory):
            os.makedirs(self.raw_capture_directory)

        self.pred_capture_directory = "Capture_Data\capture_pred"
        if not os.path.exists(self.pred_capture_directory):
            os.makedirs(self.pred_capture_directory)

        self.goodStrip_capture_directory = "Capture_Data\good_stripped"
        if not os.path.exists(self.goodStrip_capture_directory):
            os.makedirs(self.goodStrip_capture_directory)
        
        self.badStrip_capture_directory = "Capture_Data\ bad_stripped"
        if not os.path.exists(self.badStrip_capture_directory):
            os.makedirs(self.badStrip_capture_directory)

        self.goodWeld_capture_directory = "Capture_Data\good_weld"
        if not os.path.exists(self.goodWeld_capture_directory):
            os.makedirs(self.goodWeld_capture_directory)

        self.badWeld_capture_directory = "Capture_Data\ bad_weld"
        if not os.path.exists(self.badWeld_capture_directory):
            os.makedirs(self.badWeld_capture_directory)
        

        
        

        self.style = ttk.Style()
        self.style.configure("C.TButton", foreground="black", background="#4d4d4d", font=("Helvetica", 15), width=15, height=2)
        self.style.map("C.TButton", foreground=[("pressed", "white"), ("active", "purple")], background=[("pressed", "#4d4d4d"), ("active", "#4d4d4d")])

        self.update()
        self.window.mainloop()

    def update(self):
        ret, frame = self.video_capture.read()

        if ret:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((500, int(500 * frame.shape[0] / frame.shape[1]))))
            self.vid_canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(10, self.update)  # Update every 10 milliseconds

    def capture_photo(self):
        ret, frame = self.video_capture.read()

        if ret:
            # capturig and storing raw photos 
            timestamp = int(time.time())  # Get a unique timestamp
            photo_filename = f"captured_photo_{timestamp}.jpg"
            photo_path = os.path.join(self.raw_capture_directory, photo_filename)
            cv2.imwrite(photo_path, frame)
            self.display_captured_photo(photo_path)
    




    def capture_photo_badStrip(self):
        ret, frame = self.video_capture.read()

        if ret:
            # capturig and storing raw photos 
            timestamp = int(time.time())  # Get a unique timestamp
            photo_filename = f"captured_photo_{timestamp}.jpg"
            photo_path = os.path.join(self.badStrip_capture_directory, photo_filename)
            cv2.imwrite(photo_path, frame)

    def capture_photo_goodStrip(self):
        ret, frame = self.video_capture.read()

        if ret:
            # capturig and storing raw photos 
            timestamp = int(time.time())  # Get a unique timestamp
            photo_filename = f"captured_photo_{timestamp}.jpg"
            photo_path = os.path.join(self.goodStrip_capture_directory, photo_filename)
            cv2.imwrite(photo_path, frame)

    def capture_photo_goodWeld(self):
        ret, frame = self.video_capture.read()

        if ret:
            # capturig and storing raw photos 
            timestamp = int(time.time())  # Get a unique timestamp
            photo_filename = f"captured_photo_{timestamp}.jpg"
            photo_path = os.path.join(self.goodWeld_capture_directory,photo_filename)
            cv2.imwrite(photo_path, frame)


    def capture_photo_badWeld(self):
        ret, frame = self.video_capture.read()

        if ret:
            # capturig and storing raw photos 
            timestamp = int(time.time())  # Get a unique timestamp
            photo_filename = f"captured_photo_{timestamp}.jpg"
            photo_path = os.path.join(self.badWeld_capture_directory,photo_filename)
            cv2.imwrite(photo_path, frame)

    
                    







    def display_captured_photo(self, photo_path):
        captured_photo = Image.open(photo_path)


        #Computer Vision model code below !
        result = model(captured_photo)

        # Get the result with the highest confidence
        best_result = result.pred[0][0]


        # Extract the coordinates and confidence of the bounding box
        xmin, ymin, xmax, ymax = map(int, best_result[:4])
        confidence = best_result[4]
        prediction = best_result[5]# Get the prediction value
        # Map the prediction value to a dictionary key
        prediction_dict = {"dict_value": prediction}
        # Map the prediction value to a dictionary key and output the corresponding value
        predicted_class = dict_value[int(prediction)]
        
         # Set the font size and type
        font = ImageFont.truetype("arial.ttf", 50)



        # Draw the bounding box on the image
        draw = ImageDraw.Draw(captured_photo)
        draw.rectangle([xmin, ymin, xmax, ymax], outline="green", width=6)
        draw.text((xmin, ymin-20), predicted_class, fill="red" , font=font)
        
        #pil_image = Image.fromarray(np.squeeze(result.render()))
        #resized_captured_photo = pil_image.resize((500, int(500 * captured_photo.height / captured_photo.width)))
        #photo = ImageTk.PhotoImage(resized_captured_photo)
        
        pil_image = captured_photo.resize((500, int(500 * captured_photo.height / captured_photo.width)))
        photo = ImageTk.PhotoImage(pil_image)
        

        # capturig amd storing prediction photos 
        timestamp = int(time.time())
        photo_filename = f"captured_photo_{timestamp}.jpg"
        photo_path = os.path.join(self.pred_capture_directory, photo_filename)
        pil_image.save(photo_path)



        if hasattr(self, "photo_label"):
            self.photo_label.config(image=photo)
            self.photo_label.image = photo
            self.text_label.config(text=f" Prediction: {predicted_class} , Confidence: {confidence:.2f}")
        else:
            self.photo_label = tk.Label(self.photo_frame, image=photo)
            self.photo_label.pack()

    def __del__(self):
        if self.video_capture.isOpened():
            self.video_capture.release()

root = tk.Tk()
app = VideoStreamApp(root, " ThermoCouples Defect Detection ")