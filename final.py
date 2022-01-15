import os

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from collections import defaultdict

from tkinter import *
from tkinter import ttk
import warnings
warnings.filterwarnings('ignore')

def fun1():


    data = defaultdict(list)
    with open('dermatology.data', 'r') as f:
        dataset = f.readlines()

    for line in dataset:
        values = [np.NaN if value == '?' else int(value) for value in line.split(',')]

        data['erythema'].append(values[0])
        '''
        Symptoms
        -> Circular, red bumps on the soles, palms, arms, face and legs that grow into circles, may look like targets
        -> Itchiness, in some cases
        -> Painful sores or blisters on the lips, mouth, eyes and genitals
        -> Red patches with pale rings inside the patch with purple centers and small blisters called target lesions
        -> Fever, Joint pain
        Take values -> 0, 1, 2, 3
        '''

        data['scaling'].append(values[1])
        '''
        Symptoms
        -> Patches of dry skin typically appear on the elbows and lower legs 
        -> Flaky scalp
        -> Itchy skin
        -> Polygon-shaped scales on the skin
        -> Scales that are brown, gray, or white
        -> Dry skin
        Take values -> 0, 1, 2, 3
        '''

        data['definite borders'].append(values[2])
        '''
        Symptoms
        -> Iregular or jagged or wavy borders around affected area
        Take values -> 0, 1, 2, 3
        '''

        data['itching'].append(values[3])
        '''
        Symptoms
        ->  Irritating sensation that makes you want to scratch your skin
        ->  Sometimes it can feel like pain
        ->  Either in one area or all over body
        ->  May have a rash or hives
        Take values -> 0, 1, 2, 3
        '''

        data['koebner phenomenon'].append(values[4])
        '''
        Symptoms
        ->  Due toa a cut or a burn
        ->  Discoloration of the skin
        ->  Develop on parts of the body where the skin is irritated by a waistband or belt buckle
        Take values -> 0, 1, 2, 3
        '''

        data['polygonal papules'].append(values[5])
        '''
        Symptoms
        ->  Appears as small, flat-topped, red-to-purple bumps with round or irregular shape
        ->  Few small bumps or many
        ->  There are white scales or flakes on them
        Take values -> 0, 1, 2, 3
        '''

        data['follicular papules'].append(values[6])
        '''
        Symptoms
        ->  Clusters of small red bumps or white-headed pimples that develop around hair follicles
        ->  Pus-filled blisters that break open and crust over
        ->  Itchy and burning skin
        Take values -> 0, 1, 2, 3
        '''

        data['oral mucosal involvement'].append(values[7])
        '''
        Symptoms
        ->  Affect the inside of the lips, cheeks, gums, tongue, and throat
        ->  Diffuse lip
        ->  Patches on the top layer of skin.
        ->  Painful
        Take values -> 0, 1, 2, 3
        '''

        data['knee and elbow involvement'].append(values[8])
        '''
        Symptoms
        ->  Pain, swelling and stiffness because of joint inflammation
        ->  Itchy, painful red patches or a silvery white buildup of dead skin cells
        ->  Most commonly on the knees, elbows and scalp
        Take values -> 0, 1, 2, 3
        '''

        data['scalp involvement'].append(values[9])
        '''
        Symptoms
        ->  Itchy, scaly, red patches on the scalp, hairline and ears
        ->  Hair loss or Male pattern baldness
        ->  Presence of head lice
        Take values -> 0, 1, 2, 3
        '''

        data['family history'].append(values[10])
        '''
        ->  Any family history for skin diseases, Yes or No
        Take values -> 0, 1
        '''

        data['melanin incontinence'].append(values[11])
        '''
        Symptoms
        ->  
        Take values -> 0, 1, 2, 3
        '''

        data['eosinophils in the infiltrate'].append(values[12])
        '''
        Symptoms
        ->  Pain and swelling and inflammation of the skin 
        ->  Especially of the arms and legs
        Take values -> 0, 1, 2, 3
        '''

        data['PNL infiltrate'].append(values[13])
        '''
        Symptoms
        ->  Itchiness, redness of the skin and pimple-like eruptions
        ->  On areas exposed to sunlight
        ->  May last up to several months
        Take values -> 0, 1, 2, 3
        '''

        data['fibrosis of the papillary dermis'].append(values[14])
        '''
        Symptoms
        ->  Swelling and tightening of the skin
        ->  Thickening and hardening of the skin
        ->  Skin that may feel "woody" and develop an orange-peel appearance 
        ->  Darkening (excess pigmentation)
        Take values -> 0, 1, 2, 3
        '''

        data['exocytosis'].append(values[15])
        '''
        Symptoms
        ->  Inflammatory cells within epidermis  
        Take values -> 0, 1, 2, 3
        '''

        data['acanthosis'].append(values[16])
        '''
        Symptoms
        ->  Dark, thickened, velvety skin in body folds
        ->  Creases in your armpits, groin and back of the neck
        ->  Skin changes usually appear slowly
        ->  Affected skin may also have an odor or itch
        Take values -> 0, 1, 2, 3
        '''

        data['hyperkeratosis'].append(values[17])
        '''
        Symptoms
        ->  thick spots on the bottom of the foot and hands
        ->  Whitish areas inside the mouth
        ->  Thickened Skin
        ->  Blisters
        ->  Red, Scaly Patches
        ->  Thickened spots of skin on the toes or top of the foot, and can appear as dull, rounded bumps
        Take values -> 0, 1, 2, 3
        '''

        data['parakeratosis'].append(values[18])
        '''
        Symptoms
        ->  Brown or red, scaly solid elevation of skin with no visible fluid
        Take values -> 0, 1, 2, 3
        '''

        data['clubbing of the rete ridges'].append(values[19])
        '''
        Symptoms
        ->  
        Take values -> 0, 1, 2, 3
        '''

        data['elongation of the rete ridges'].append(values[20])
        '''
        Symptoms
        ->  A flattened brown pigmented spot on the skin  
        Take values -> 0, 1, 2, 3
        '''

        data['thinning of the suprapapillary epidermis'].append(values[21])
        '''
        Symptoms
        ->  
        Take values -> 0, 1, 2, 3
        '''

        data['spongiform pustule'].append(values[22])
        '''
        Symptoms
        ->  Small bumps on the skin that contain fluid or pus
        ->  Appear as white bumps surrounded by red skin
        ->  Any part of the body but they most commonly form on the back, chest, and face.
        Take values -> 0, 1, 2, 3
        '''

        data['munro microabscess'].append(values[23])
        '''
        Symptoms
        ->   
        Take values -> 0, 1, 2, 3
        '''

        data['focal hypergranulosis'].append(values[24])
        '''
        Symptoms
        ->  Skin becomes thicker than normal 
        Take values -> 0, 1, 2, 3
        '''

        data['disappearance of the granular layer'].append(values[25])
        '''
        Symptoms
        ->   
        Take values -> 0, 1, 2, 3
        '''

        data['vacuolisation and damage of basal layer'].append(values[26])
        '''
        Symptoms
        ->  
        Take values -> 0, 1, 2, 3
        '''

        data['spongiosis'].append(values[27])
        '''
        Symptoms
        ->  Scaly patches of irritated skin
        ->  Rashes in the shape of coins
        ->  Reddened skin
        ->  Dandruff that's difficult to get rid of
        ->  Oozing and infection after scratching an affected area
        Take values -> 0, 1, 2, 3
        '''

        data['saw-tooth appearance of retes'].append(values[28])
        '''
        Symptoms
        ->  
        Take values -> 0, 1, 2, 3
        '''

        data['follicular horn plug'].append(values[29])
        '''
        Symptoms
        ->  Rash
        Take values -> 0, 1, 2, 3
        '''

        data['perifollicular parakeratosis'].append(values[30])
        '''
        Symptoms
        ->  
        Take values -> 0, 1, 2, 3
        '''

        data['inflammatory monoluclear inflitrate'].append(values[31])
        '''
        Symptoms
        ->  White blood cells collect at the site of injury to help clear away the debris
        Take values -> 0, 1, 2, 3
        '''

        data['band-like infiltrate'].append(values[32])
        '''
        Symptoms
        ->  Inflammation at or near the insertion site with swollen, taut skin with pain
        ->  Patchy band like pattern
        Take values -> 0, 1, 2, 3
        '''

        data['age'].append(values[33])

        data['class label'].append(values[34])

    df = pd.DataFrame(data, columns=data.keys())

    df['age'] = df['age'].replace(np.nan, df['age'].median())
    df['age'] = df['age'].replace(0.0, df['age'].median())
    X = df.drop('class label', axis=1)
    y = df['class label']  # target value

    xgb = XGBClassifier()
    xgb.fit(X, y)
    print(dataset1)

    di=[{"erythema":dataset1[0], "scaling":dataset1[1], "definite borders":dataset1[2], "itching":dataset1[3], "koebner phenomenon":dataset1[4], "polygonal papules":dataset1[5], "follicular papules":dataset1[6], "oral mucosal involvement":dataset1[7], "knee and elbow involvement":dataset1[8], "scalp involvement":dataset1[9], "family history":dataset1[10], "melanin incontinence":dataset1[12], "eosinophils in the infiltrate":dataset1[13], "PNL infiltrate":dataset1[14], "fibrosis of the papillary dermis":dataset1[15], "exocytosis":dataset1[16], "acanthosis":dataset1[17], "hyperkeratosis":dataset1[18], "parakeratosis":dataset1[19], "clubbing of the rete ridges":dataset1[20], "elongation of the rete ridges":dataset1[21], "thinning of the suprapapillary epidermis":dataset1[22], "spongiform pustule":dataset1[23], "munro microabscess":dataset1[24], "focal hypergranulosis":dataset1[25], "disappearance of the granular layer":dataset1[26], "vacuolisation and damage of basal layer":dataset1[27], "spongiosis":dataset1[28], "saw-tooth appearance of retes":dataset1[29], "follicular horn plug":dataset1[30], "perifollicular parakeratosis":dataset1[31], "inflammatory monoluclear inflitrate":dataset1[32], "band-like infiltrate":dataset1[33], "age":dataset1[11]}]
    df1=pd.DataFrame(di)
    yhat = xgb.predict(df1)
    print(yhat)
    class_labels = {1: 'psoriasis',
                    2: 'seboreic dermatitis',
                    3: 'lichen planus',
                    4: 'pityriasis rosea',
                    5: 'cronic dermatitis',
                    6: 'pityriasis rubra pilaris'}
    lbl=Label(top,text="Disease predicted: {}".format(class_labels[yhat[0]]),font=('Courier', -25, 'bold'), fg='green').grid(row=40,column=1,columnspan=7)






def table(name):
    xv = 1
    yv = 3
    for x in range(34):
        label = Label(top)
        #label.pack()
        label.config(text=str(x + 1) + ". " + name[x]+"\t\t"+ symptoms[x], font=('Helvetica',11,'italic'))
        label.grid(row=yv, column=0,sticky='w')

        radio = IntVar()
        if (x != 11):
            op = 4
            if (x == 10):
                op = 2
            for i in range(op):
                R1 = Radiobutton(top, text=i, variable=radio, value=i)
                #R1.pack(anchor=W)
                R1.grid(row=yv, column=xv,sticky='w')
                xv = xv + 1
        else:
            entryNum1 = Entry(top, textvariable=radio)
            entryNum1.grid(row=yv, column=xv,columnspan=1,sticky='w')
            yv = yv + 2;
        radiolist.append(radio)
        xv = 1;
        yv = yv + 1;


root = Tk()
root.title('Dermalyser')
#root.iconbitmap('c:/gui/codemy.ico')
root.geometry("500x400")
# Create A Main Frame
main_frame = Frame(root)
main_frame.pack(fill=BOTH, expand=1)
# Create A Canvas
my_canvas = Canvas(main_frame)
my_canvas.pack(side=LEFT, fill=BOTH, expand=1)
# Add A Scrollbar To The Canvas
my_scrollbar = ttk.Scrollbar(main_frame, orient=VERTICAL, command=my_canvas.yview)
my_scrollbar.pack(side=RIGHT, fill=Y)
# Configure The Canvas
my_canvas.configure(yscrollcommand=my_scrollbar.set)
my_canvas.bind('<Configure>', lambda e: my_canvas.configure(scrollregion = my_canvas.bbox("all")))
# Create ANOTHER Frame INSIDE the Canvas
top = Frame(my_canvas)
# Add that New frame To a Window In The Canvas
my_canvas.create_window((0,0), window=top, anchor="nw")

label1 = Label(top, font=('Helvetica', 10, 'bold'))

label1.config(text="Please enter values of Clinical and Histopathological attributes:", font=('Times', 16,'bold'))
#label1.pack()
label1.grid(row=0, column=0,sticky='w')
label2 = Label(top, bg="white", fg="#CD3333",font=('Helvetica', 12,'bold'))
#label2.pack()
label2.config(
    text="* NOTE *\nFrom the data set features, age and family history are continuous and ranged between 0-1,respectively.\n Every other feature(Clinical and Histopathological) was given a degree in range of 0 to 3 which,0 indicates that the feature was not present ,3 indicates the largest amount possible and 1,2 indicate the relative intermediate values.", font=('Helvetica', 10,'bold'))
label2.grid(row=1, column=0,pady=40,columnspan=7)
label = Label(top, font=('Helvetica',11,'bold'), bg="light green")
#label.pack()
label.config(text="Clinical Attributes: take values 0, 1, 2, 3, unless otherwise indicated--")
label.grid(row=2, column=0, columnspan=2, sticky='w')
label3 = Label(top).grid(row=15,column=0,columnspan=4)
label3 = Label(top, font=('Helvetica',11,'bold'), bg="light green")
label3.config(text="Histopathological Attributes: take values 0, 1, 2, 3--")
label3.grid(row=16, column=0, columnspan=2, sticky='w' )
name = ["erythema", "scaling", "definite borders", "itching", "koebner phenomenon", "polygonal papules",
        "follicular papules", "oral mucosal involvement", "knee and elbow involvement", "scalp involvement"
    , "family history, (0 or 1)", "Age (linear)", "melanin incontinence", "eosinophils in the infiltrate",
        "PNL infiltrate", "fibrosis of the papillary dermis", "exocytosis", "acanthosis", "hyperkeratosis"
    , "parakeratosis", "clubbing of the rete ridges", "elongation of the rete ridges",
        "thinning of the suprapapillary epidermis", "spongiform pustule", "munro microabcess", "focal hypergranulosis"
    , "disappearance of the granular layer", "vacuolisation and damage of basal layer", "spongiosis",
        "saw-tooth appearance of retes", "follicular horn plug", "perifollicular parakeratosis"
    , "inflammatory monoluclear inflitrate", "band-like infiltrate"]

symptoms=["(red bumps, Itchiness, Painful sores or blisters, Fever, Joint pain)", "(Flaky scalp, Itchy skin, Dry skin)", "(Iregular or jagged or wavy borders around affected area)", "(Irritating sensation that makes you want to scratch your skin, rash or hives)", "(Discoloration of the skin, develop in region of waistband or belt buckle)", "(small, flat-topped, red-to-purple bumps with white scales or flakes)",
"(small red bumps or white-headed pimples that develop around hair follicles, Itchy and burning skin)", "(Pain inside of the lips, cheeks, gums, tongue, and throat)", "(Pain, swelling and stiffness because of joint inflammation)", "(Itchy, scaly, red patches on the scalp, hairline and ears, Hair loss, head lice)",
 " ", " ", " ", "(Pain and swelling and inflammation of the skin)", "(Itchiness, redness of the skin and pimple-like eruptions)",
 "", "(Swelling, tightening, Thickening and hardening of the skin)", "(Inflammatory cells within epidermis)", "(Dark, thickened, velvety skin in body folds)", "(thick spots on the bottom of the foot and hands, Whitish areas inside the mouth)",
  "(Brown or red, scaly solid elevation of skin with no visible fluid)", "", "(A flattened brown pigmented spot on the skin)", " ", "(Small bumps on the skin that contain fluid or pus)",
 " ", "(Skin becomes thicker than normal)", " ", " ", "(Scaly patches of irritated skin, Rashes, Reddened skin, Dandruff)", " ", "(Rash)", " ", "(white blood cells collect at the site of injury to help clear away the debris)",
 "(Inflammation at or near the insertion site with swollen, taut skin with pain)"]
radiolist = []
dataset1=[]
table(name)
def selection():
    for x in radiolist:
        dataset1.append(x.get())
        print(x.get())
    fun1()


def reset():
    root.destroy()
    os.system('python final.py')


Label(top,height=5).grid(row=39,column=0)
Button(top, bg="light green", text="Predict Disease", command=selection, height=3, width=13).grid(row=40, column=0, sticky='E')
Button(top, bg="light green", text="New", command=reset, height=3, width=13).grid(row=40, column=0, sticky='W')
root.mainloop()