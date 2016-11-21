#!/usr/bin/env python
from PIL import ImageTk
import numpy as np
import tkFileDialog
from Tkinter import *
from keras.models import load_model
from keras.preprocessing.image import array_to_img

class App(object):
    def __init__(self, master):
        self.frame = frame = master#Frame(master)
        #frame.pack()
        self.generator = None
        self.view_position = None
        self.last_positions = None
        self.last_samples = None
        self.open_generator_button = Button(
            frame, text='Open Generator', command=self.open_generator
        )
        self.open_generator_button.grid(row=0, column=0)
        self.radius_entry = Entry(frame)
        self.radius_entry.insert(0, '0.01')
        self.radius_entry.grid(row=0, column=1)
        self.update_button = Button(
            frame, text='Random View', command=self.set_random_view_position
        )
        self.update_button.grid(row=0, column=2)
        self.quit_button = Button(
            frame, text='Quit', fg='red', command=frame.quit
        )
        self.quit_button.grid(row=0, column=3)
        self.image_frame = Frame(frame)
        self.image_frame.grid(row=1, columnspan=4)
        self.images = []
        self.last_radius = None
        self.num_rows = self.num_cols = 5
        self.setup_images(self.num_rows, self.num_cols)

    def setup_images(self, num_rows, num_cols):
        for y in range(num_rows):
            for x in range(num_cols):
                i = y * num_rows + x
                label = Label(self.image_frame)
                label.grid(row=y, column=x)
                label.bind('<Button-1>', self.on_clicked_image)
                label.index = i
                label.coords = dict(row=y, column=x)
                self.images.append(label)

    def open_generator(self):
        filename = tkFileDialog.askopenfilename()#(**self.file_opt)
        if filename:
            print('Loading {}...'.format(filename))
            self.generator = load_model(filename)
            print('Loaded.')
            self.set_view_position(np.ones(self.input_shape) * 0.5)
            self.random_walk()

    def on_click_sample(self):
        self.set_random_view_position()

    def set_random_view_position(self):
        try:
            r = float(self.radius_entry.get())
        except ValueError:
            r = 0.01
        r_low = min(r, 1. - r)
        r_hi = max(r, 1. - r)
        self.set_view_position(np.random.uniform(r_low, r_hi, self.input_shape))

    def set_view_position(self, p):
        self.view_position = p.copy()
        self.update_images()

    def translate_view_position(self, dp):
        self.set_view_position(self.view_position + dp)

    def randomize_image_offsets(self, radius):
        self.image_offsets = np.random.uniform(
            -radius, radius, (len(self.images),) + self.input_shape
        )
        self.last_radius = radius

    def update_images(self):
        if not self.generator:
            return
        try:
            r = float(self.radius_entry.get())
        except ValueError:
            r = 0.01
        if r != self.last_radius:
            self.randomize_image_offsets(r)
        offset_positions = self.image_offsets + self.view_position
        samples = self.generator.predict(offset_positions, verbose=False)
        for label, sample in zip(self.images, samples):
            tkimage = ImageTk.PhotoImage(image=array_to_img(sample))
            label.configure(image=tkimage)
            label.img = tkimage
        self.frame.update()
        self.last_positions = offset_positions

    def random_walk(self):
        dp = 0.001
        num_offsets = len(self.image_offsets)
        self.translate_view_position(
            np.random.uniform(-dp, dp, self.input_shape)
        )
        self.image_offsets += np.random.uniform(-dp, dp, self.image_offsets.shape)
        self.image_frame.after(200, self.random_walk)

    def on_clicked_image(self, event):
        offset = self.image_offsets[event.widget.index]
        self.set_view_position(offset + self.view_position)

    def samples_in_radius(self, num_samples, center, radius):
        offsets = np.random.uniform(-radius, radius, (num_samples,) + self.input_shape)
        zs = offsets + center[None,:]
        preds = self.generator.predict(zs)
        return zs, preds

    @property
    def input_shape(self):
        return self.generator.input_shape[1:]

if __name__ == '__main__':
    root = Tk()
    root.wm_title('Generator Viewer')
    app = App(root)
    root.mainloop()
    root.destroy()
