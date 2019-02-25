import imageio
from tkinter import *
from PIL import Image, ImageTk
import random
import torch
from DCGAN_MNIST import *

netG.load_state_dict(torch.load('Networks/DCGANG_128xFaces5.pt'))


def Board():
    _, img = netG.forward(create_noise(1))
    img = img.detach().cpu().numpy()[0][0]
    img = ((img - img.min()) * (1/img.max() - img.min()) * 255)

    def create_vector():
        arr = np.empty((1, 128, 1, 1))
        for i in range(128):
            arr[0][i][0][0] = sliders[i].get()
        ten = torch.tensor(arr)
        return ten.float()

    def update_picture(x):
        _, img = netG.forward(create_vector())
        img = img.detach().cpu().numpy()[0]
        img = ((img - img.min()) * (1 / img.max() - img.min()) * 50).astype('uint8')
        img = img.transpose((1, 2, 0))
        photo = ImageTk.PhotoImage(image=Image.fromarray(img, 'RGB').resize((256, 256), Image.ANTIALIAS))

        label.configure(image=photo)
        label.image = photo

    def interpolate():
        ndx = random.randint(0, 123)
        direction = []
        arr = [-1, 1]
        for i in range(5):
            direction.append(arr[random.randint(0, 1)])
        for i in range(10):
            for x in range(5):
                sliders[ndx].set(sliders[ndx].get() + (direction[x] * 0.03))
                ndx += 1
            update_picture(1)
            ndx -= 5
            time.sleep(0.05)

    def randomize():
        for x in sliders:
            x.set(random.uniform(-1, 1))

    def reset():
        for x in sliders:
            x.set(0)

    tk = Tk()

    photo = ImageTk.PhotoImage(image=Image.fromarray(img))
    label = Label(image=photo, height=256, width=256)
    label.grid(row=0, column=0)

    sliders = []
    for i in range(128):
        sliders.append(Scale(tk, from_=1, to=-1, resolution=0.01, command=update_picture))
        sliders[i].grid(row=int(i/32), column=i % 32 + 2)

    create = Button(text="interpolate", command=interpolate).grid(row=1, column=0)
    rand = Button(text='randomize', command=randomize).grid(row=2, column=0)
    zero = Button(text='reset', command=reset).grid(row=3, column=0)

    tk.mainloop()


if __name__ == '__main__':
    Board()
