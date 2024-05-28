import PySimpleGUI as sg
import io
import base64
from PIL import Image
from datetime import datetime
import cv2
import socket
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import joblib
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pygame
import socket
from skimage.transform import resize 
from skimage.io import imread 


pygame.init()
infos = pygame.display.Info()
width_display = (infos.current_w,infos.current_h)
BORDER_COLOR = '#2c78c9'
DARK_HEADER_COLOR = '#010101'
img_HEIGHT = 650
img_WIDTH = 700

Nimg_HEIGHT = infos.current_h
Nimg_WIDTH = 800

image_SIZE = (700, 500)
img_size = (3264 , 2448)
Display_image_SIZE = (700, 500)


########### Function Popup Message #############

def get_popup(Message):
  sg.popup(Message, background_color='#282828',no_titlebar=True)

def get_popup_auto(Message):
  sg.popup_auto_close(Message, background_color='#282828',no_titlebar=True, auto_close_duration=1.5)

################################################

########### convert image to base64 image #############
def get_image64(filename):
    with open(filename, "rb") as img_file:
        image_data = base64.b64encode(img_file.read())
    buffer = io.BytesIO()
    imgdata = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(imgdata))
    new_img = img.resize(Display_image_SIZE)  # x, y
    new_img.save(buffer, format="PNG")
    img_b64 = base64.b64encode(buffer.getvalue())
    return img_b64

img_b64 = get_image64("Offline.jpg")
image_display = [
        [sg.T('Realtime Camera',
              font=('Helvetica', 15, "bold"),
              justification='center',
              text_color='#000000')],
        [sg.Image(data=img_b64, pad=(0, 0), key='image', size=Display_image_SIZE)] 
    ]
Image_Processed = [
         [sg.T('Image Processed',
              font=('Helvetica', 15, "bold"),
              justification='center',
              text_color='#000000')],
        [sg.Image(data=img_b64, pad=(0, 0), key='PImage',size=Display_image_SIZE)]
    ]

################################################



############# Matplotlib on PYSIMPLEGUI #####################
matplotlib.use('TkAgg')
fig = matplotlib.figure.Figure(figsize=(5, 4), dpi=100)
t = np.arange(0, 3, .01)
fig.add_subplot(111).plot(t, 2 * np.sin(2 * np.pi * t))

def draw_figure(canvas, figure):
   figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
   figure_canvas_agg.draw()
   figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
   return figure_canvas_agg
################################################
top_layout = [[sg.Column(image_display, vertical_alignment='center'), sg.Column(Image_Processed, vertical_alignment='center')]]

header_layout = [[
    sg.Column(
        [
            [sg.Text('', font='Any 22', key='timetext', background_color=BORDER_COLOR, size=(61, 1)),]
        ], pad=((0, 0), (0, 0)), background_color=BORDER_COLOR,
    )
]]

content_layout = [[
############# Save image options ###############
sg.Column(
    [
        [sg.T('D I S P L A Y  I M A G E',
              font=('Helvetica', 15, "bold"),
              text_color='#000000',background_color=BORDER_COLOR, visible=True)],
    ], pad=((0, 0), (0, 0)), background_color=BORDER_COLOR,
),

sg.Column(
    [
        [sg.T('Save Directory:',
              font=('Helvetica', 12),
              background_color=BORDER_COLOR)],
        [sg.Input(sg.user_settings_get_entry('-locImage-', ''),
                  key='-locImage-',
                  enable_events=True,
                  disabled=True,
                  use_readonly_for_disable=False,), sg.FolderBrowse()]
    ], pad=((0, 0), (0, 0)), background_color=BORDER_COLOR,
),
###############################################



##################    Decision   ################
sg.Column(
    [
        [sg.T('DECISION',
              size=(13, 1),
              font=('Helvetica', 15, "bold"),
              background_color=BORDER_COLOR,
              key='-decisionlabel-',
                text_color='#000000',
              justification='left')],
        [sg.Button('ANALYZE', button_color=('#000000', '#d8ff34'),size=(15, 1))],
        [sg.T('Label Hilang / Pudar: ', font=('Helvetica', 12), background_color=BORDER_COLOR),
         sg.Input(sg.user_settings_get_entry('-result_1-', ''), disabled = True,pad=(142,0),
                  key='-result_1-', size=(10, 1)),
        ],
        [sg.T('Botol Penyok: ', font=('Helvetica', 12), background_color=BORDER_COLOR),
         sg.Input(sg.user_settings_get_entry('-result_2-', ''), disabled = True,pad=(192,0),
                  key='-result_2-', size=(10, 1)),
        ],
        [sg.T('Waktu (S): ', font=('Helvetica', 12), background_color=BORDER_COLOR),
         sg.Input(sg.user_settings_get_entry('waktu', ''), disabled = True,pad=(217,0),
                  key='waktu', size=(10, 1)),
        ],
        [
            sg.Button('', key='kesimpulan',disabled=False, button_color=('#00aa01'),size=(50, 2)),
        ]
    ], background_color=BORDER_COLOR
),
sg.Column(
    [
        [sg.T('OUTPUT',
              size=(13, 1),
              font=('Helvetica', 15, "bold"),
              background_color=BORDER_COLOR,
              key='-decisionlabel-',
                text_color='#000000',
              justification='left')],
        [sg.Output(size=(50,10), key='-OUTPUT-')]
    ], background_color=BORDER_COLOR
),
]
]

###############################################

bottom_layout = [sg.Column(content_layout,background_color=BORDER_COLOR,size=(infos.current_w,500))]
clock_layout = [sg.Column(header_layout, background_color=BORDER_COLOR,size=(infos.current_w,50))]
layout = [[clock_layout, top_layout, bottom_layout]]


window = sg.Window('DVI - Decal Visual Inspection',
                   layout, finalize=True,
                   resizable=True,
                   no_titlebar=False,
                   margins=(0, 0),
                   grab_anywhere=True,
                    background_color='#2a2a2a',
                    element_justification='c',
                  location=(0, 0), right_click_menu=sg.MENU_RIGHT_CLICK_EXIT)

################################################



########### I M A G E  P R O C E S S I N G ##############

def Realtime_process_image():

    ret, frame = cap.read()
    frame = cv2.resize(frame, image_SIZE)
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresholded_image = gray_image.copy()
    otsu_threshold, image_result = cv2.threshold(thresholded_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold_value = 128
    thresholded_image[gray_image > threshold_value] = 255
    
    imgbytes = cv2.imencode('.png', thresholded_image)[1].tobytes()

    dataImage = base64.b64encode(imgbytes).decode('ascii')    
    content = base64.b64decode(dataImage)
    window['PImage'].update(data=content)

################################################
########### MACHINE LEARNING CORE ##############

def process_image(image_path):
    image1 = Image.open(image_path)

    # Langkah 1: Potong gambar
    width, height = image1.size
    x,y = 400,200
    cropped_image = image1.crop((x, y, width-300, height-100))

    # Langkah 2: Konversi ke grayscale
    gray_image = cv2.cvtColor(np.array(cropped_image), cv2.COLOR_BGR2GRAY)

    # Langkah 3: Thresholding
    thresholded_image = gray_image.copy()
    otsu_threshold, image_result = cv2.threshold(thresholded_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold_value = 120
    thresholded_image[gray_image > threshold_value] = 255

    plt.imshow(thresholded_image, cmap='gray') 
    plt.axis('off')
    #plt.show()
    now = datetime.now()
    filename = now.strftime("AnalyzedObject_%Y%m%d%H%M%S%f") + ".png"
    plt.savefig('Analyzed Files/'+ filename)

    return thresholded_image

def SVM_process_image(image_path):
    image1 = Image.open(image_path)

    # Langkah 1: Potong gambar
    width, height = image1.size
    x,y = 400,100
    cropped_image = image1.crop((x, y, width-300, height-800))

    # Langkah 2: Konversi ke grayscale
    gray_image = cv2.cvtColor(np.array(cropped_image), cv2.COLOR_BGR2GRAY)

    # Langkah 3: Thresholding
    thresholded_image = gray_image.copy()
    otsu_threshold, image_result = cv2.threshold(thresholded_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold_value = 128
    thresholded_image[gray_image > threshold_value] = 255

    return thresholded_image

def model_1(image):
    Categories=['Botol Bagus','Cat Pudar Botol'] 
    model_SVM = r'Version 2 Final_SVM.pkl'

    loaded_model = joblib.load(model_SVM)
    
    img=SVM_process_image(image)

    img_resize=resize(img,(150,150,3)) 
    l=[img_resize.flatten()] 
    CategoryClass = Categories[loaded_model.predict(l)[0]]

    probability=loaded_model.predict_proba(l) 
    for ind,val in enumerate(Categories): 
        print(f'{val} = {probability[0][ind]*100}%') 

    if CategoryClass == "Cat Pudar Botol":
        kesimpulan = 'Yes' #Botol Pudar?
    else :
        kesimpulan = "No" #Botol Bagus?
    return kesimpulan

def model_2(image):

    label = ['Bagus','Penyok'] 
    model_RF = r'Version_2_Final_RF.pkl'

    loaded_model = joblib.load(model_RF)
    img=process_image(image)

    img_resize=resize(img,(150,150,3)) 
    l=[img_resize.flatten()] 
    probability_Array=loaded_model.predict_proba(l)
    probability=label[loaded_model.predict(l)[0]]
    for ind,val in enumerate(label): 
        print(f'{val} = {probability_Array[0][ind]*100}%') 
    if probability == "Penyok":
        kesimpulan = 'Yes' #Botol Lecet?
    else :
        kesimpulan = "No" #Botol Bagus?
    return kesimpulan

finalresults = "default"

def cek_botol(image_path):
    print('\n')
    global finalresults
    start_time = time.time()
    process_image(image_path)
    result_1 = model_1(image_path)
    print('\n')
    result_2 = model_2(image_path)
    
    #model_3(result_image)

    if result_1 == 'Yes' or result_2 == 'Yes':
        kesimpulan = "Botol Rejected"
        finalresults = kesimpulan
        window['kesimpulan'].update(button_color='#ff0000')

    else:
        kesimpulan = "Botol Good"
        finalresults = kesimpulan
        window['kesimpulan'].update(button_color='#00aa01')

    waktu = time.time() - start_time

    window['-result_1-'].update(result_1)
    window['-result_2-'].update(result_2)
    window['kesimpulan'].update(kesimpulan)
    window['waktu'].update(waktu)

################################################


############## Camera Encoding #################
cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)
cap.set(6, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))

################################################

def save_image(directory, imageSaving, openimage):
    
    ret, frame = cap.read()
    frameShow = cv2.resize(frame, image_SIZE)
    if imageSaving:
        savedframe = cv2.resize(frame, img_size)
        now = datetime.now()
        filename = now.strftime("ObjectChecked_%Y%m%d%H%M%S%f") + ".png"
        new_file_name = os.path.join(directory, filename)
        cv2.imwrite(new_file_name, savedframe)
    if openimage:
        analyzed_img = r"Saved Images/"+ filename
        cek_botol(analyzed_img)

def capture_image():
    ret, frame = cap.read()
    frame = cv2.resize(frame, image_SIZE)
    imgbytes = cv2.imencode('.png', frame)[1].tobytes()
    dataImage = base64.b64encode(imgbytes).decode('ascii')    
    content = base64.b64decode(dataImage)
    window['image'].update(data=content)

def DeactivateCamera():
    imageSample = get_image64("Offline.jpg")
    window['image'].update(data=imageSample)

##################### NETWORK RELATED ##################
camera_realtime = 1

isSaving = sg.user_settings_get_entry('-IPSetting-', '')
directory = sg.user_settings_get_entry('-locImage-', '')
id = 1

########################################################

while True:
    window['timetext'].update(time.strftime('%H:%M:%S'))
    
    event, values = window.read(timeout=20)
    if event == 'EXIT' or event == sg.WIN_CLOSED:
        break  # exit button clicked
    if camera_realtime:
       Realtime_process_image()
       capture_image()
    if event == '-locImage-':
        sg.user_settings_set_entry('-locImage-', values['-locImage-'])
        directory = sg.user_settings_get_entry('-locImage-', '')
        TCPEnable = False
    elif event == '-isSaveImage-':
        sg.user_settings_set_entry('-isSaveImage-', values['-isSaveImage-'])
        isSaving = values['-isSaveImage-']
    if event == 'ANALYZE':
        save_image(directory, True,True)
    elif event == 'updateDevice':
        sg.user_settings_set_entry('-deviceName-', values['-deviceName-'])
        deviceName = values['-deviceName-']
        id = deviceName

window.close()
