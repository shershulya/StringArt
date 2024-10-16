
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rescale, rotate
from skimage.io import imread
from functools import partial
from scipy.fft import fft, ifft
from skimage import exposure
import random
import re
from os import listdir
from os.path import isfile, join
import sys

"""### Cчитывание картинки """
def read_img(filename):
    image = imread(filename, as_gray=True)

    if image.shape[0] > 1000 and image.shape[1] > 1000:
        scaling_factor = 700 / image.shape[0] 
        print("Image was rescaled. Scaling factor: ", scaling_factor)
        image = rescale(image, scaling_factor, anti_aliasing=True)

    # plt.rcParams['figure.figsize'] = [10, 10]

    # plt.axis('off')
    # plt.imshow(image, cmap=plt.cm.Greys_r)

    return image

"""### Форматирвоание под тип квадратной картинки с круглым наполнением внутри """
def procces_to_circle(image):
    round_size = int(np.floor(np.sqrt((max(image.shape[0], image.shape[1])) ** 2 / 2.0)))

    a1 = (image.shape[0] - round_size) // 2
    a2 = (image.shape[1] - round_size) // 2

    cropped = image[a1:image.shape[0] - a1, a2:image.shape[1] - a2]
    rounded = np.zeros((round_size, round_size))

    for i, row in enumerate(rounded):
        for j, line in enumerate(row): 
            rounded[i][j] = cropped[i][j]

    radius = round_size // 2
    xpr, ypr = np.mgrid[:round_size, :round_size] - radius

    out_circle = (xpr ** 2 + ypr ** 2) > radius ** 2
    rounded[out_circle] = 0.

    # plt.imshow(rounded, cmap=plt.cm.Greys_r)

    return rounded

"""### Получение синограмы с помощью прямого преоброзования радона пунтк 1 алгоритма из задания """
def radon_transform(image, steps):
    print(image.dtype)
    radon = np.zeros((steps, len(image)), dtype='float64')
    for s in range(steps):
        rotation = rotate(image, -s*180/steps).astype('float64')
        radon[:,s] = sum(rotation)
    return radon

def get_sinogram(rounded, theta):
    theta_x = np.linspace(theta[0], theta[1], max(rounded.shape), endpoint=False)
    sinogram = radon_transform(rounded, rounded.shape[0])
    # plt.imshow(sinogram, cmap=plt.cm.Greys_r)
    return sinogram, theta_x

"""###Выравнивание и паддинг с учетом "круглости" картинки """
def cicrcle_padding(sinogram):
    diagonal = int(np.ceil(np.sqrt(2) * sinogram.shape[0]))
    pad = diagonal - sinogram.shape[0]
    old_center = sinogram.shape[0] // 2
    new_center = diagonal // 2
    pad_before = new_center - old_center
    pad_width = ((pad_before, pad - pad_before), (0, 0))
    radon_image = np.pad(sinogram, pad_width, mode='constant', constant_values=0)
    return radon_image


"""###Свертка проекций с рамп-фильтром пунтк 2 алгоритма из задания """
def ramp_filter(radon_image):

    img_shape = radon_image.shape[0]

    # Resize image to next power of two (but no less than 64) for
    # Fourier analysis; speeds up Fourier and lessens artifacts
    size = max(64, int(2 ** np.ceil(np.log2(2 * img_shape))))

    n = np.concatenate((np.arange(1, size / 2 + 1, 2, dtype=int),
                            np.arange(size / 2 - 1, 0, -2, dtype=int)))
    f = np.zeros(size)
    f[0] = 0.25
    f[1::2] = -1 / (np.pi * n) ** 2

    # # Computing the ramp filter from the fourier transform of its
    # # frequency domain representation lessens artifacts and removes a
    # # small bias as explained in [1], Chap 3. Equation 61
    fourier_filter = 2 * np.real(fft(f))         # ramp filter
    filter = fourier_filter[:, np.newaxis]

    pad_width = ((0, size - img_shape), (0, 0))
    img = np.pad(radon_image, pad_width, mode='constant', constant_values=0)

    projection = fft(img, axis=0) * filter
    radon_filtered = np.real(ifft(projection, axis=0)[:img_shape, :])

    # plt.imshow(radon_filtered, cmap=plt.cm.Greys_r)
    return radon_filtered

"""### Прореживание проекций по пространственной переменной(как я понял этот шаг) пунтк 3 алгоритма из задания """
def thinning(radon_filtered, n_strings = 45):
    # print(radon_filtered.shape)

    n = int(radon_filtered.shape[1] / n_strings)
    # print(int(n))

    reduced = np.zeros(radon_filtered.shape)

    for i, row in enumerate(radon_filtered):
        for j, line in enumerate(row):
            if j % n == 0 and i % n == 0:
                reduced[i][j] = line

    # plt.imshow(reduced, cmap=plt.cm.Greys_r)
    return reduced

"""### Нормировка разбитая на куски по шагам пунтк 4 алгоритма из задания """
def normolizing(reduced):
    non_zero = np.zeros(reduced.shape)

    for i, row in enumerate(reduced):
        for j, line in enumerate(row):
            non_zero[i][j] = line if line > 0 else 0

    # plt.imshow(non_zero, cmap=plt.cm.Greys_r)

    normolized = exposure.adjust_gamma(non_zero)

    # plt.imshow(normolized, cmap=plt.cm.Greys_r)
    return normolized

"""###Бинарное распыление синограммы пунтк 5 алгоритма из задания """
def binarization(normolized):
    binary_img = np.zeros(normolized.shape)

    val = [0, 1]
    for i, row in enumerate(normolized):
        for j, line in enumerate(row):
            l = random.choices(val, weights=[1 - line, line], k=1)
            binary_img[i][j] = l[0]

    # plt.imshow(binary_img, cmap=plt.cm.Greys_r)
    return binary_img

"""###Обратное преобразование радона пукнт 6 алгоритма из задания """
def back_projection(binary_img, theta):
    angles_count = len(theta)
    
    img_shape = binary_img.shape[0]
    output_size = int(np.floor(np.sqrt((img_shape) ** 2 / 2.0)))
    reconstructed = np.zeros((output_size, output_size))
    radius = output_size // 2
    xpr, ypr = np.mgrid[:output_size, :output_size] - radius
    x = np.arange(img_shape) - img_shape // 2

    for col, angle in zip(binary_img.T, np.deg2rad(theta)):
        t = ypr * np.cos(angle) - xpr * np.sin(angle)
        interpolant = partial(np.interp, xp=x, fp=col, left=0, right=0)
        reconstructed += interpolant(t)


    out_reconstruction_circle = (xpr ** 2 + ypr ** 2) > radius ** 2
    reconstructed[out_reconstruction_circle] = 0.

    reconstructed = reconstructed * np.pi / (2 * angles_count)

    return reconstructed

def get_stringart(src_file, n_strings = 75, theta = [0, 180], draw_add_info = True, name = "_"):
    image = read_img(src_file)
    rounded = procces_to_circle(image)
    sinogram, theta_x = get_sinogram(rounded, theta)
    circled = cicrcle_padding(sinogram)
    radon_filtered = ramp_filter(circled)
    reduced_img = thinning(radon_filtered, n_strings)
    normolized_img = normolizing(reduced_img)
    binary_img = binarization(normolized_img)
    reconstructed = back_projection(binary_img, theta_x)

    if draw_add_info:
        f = plt.figure(figsize=(10, 10))
        f.add_subplot(2,2, 1)
        plt.imshow(sinogram, cmap=plt.cm.Greys_r)
        f.add_subplot(2,2, 2)
        plt.imshow(radon_filtered, cmap=plt.cm.Greys_r)
        f.add_subplot(2,2, 3)
        plt.imshow(reduced_img, cmap=plt.cm.Greys_r)
        f.add_subplot(2,2, 4)
        plt.imshow(binary_img, cmap=plt.cm.Greys_r)
        plt.savefig("addInfo/" + name + "_info.png")
        plt.show(block=True)
        # plt.savefig("addInfo/info.png")

    return reconstructed

def main():
    if len(sys.argv) == 1:
        print("No arguments passed")
        print("For list of arguments use -h or -help")
        print("Using deafult params")
    else:
        cmd = sys.argv[1]
        if (cmd == '-h' or cmd == '-help' or cmd == '--help' or cmd == '--h'):
            print("filename")
            print("     Получить String Art из данного файла")
            print("filename num_threads")
            print("     Получить String Art из данного файла c заданным количесвтом нитей")
            print("     @:param num_threads: Количество нитей в итоговом изображении")
            print("filename num_threads end_angles")
            print("     Получить String Art из данного файла c заданным количесвтом нитей")
            print("     @:param num_threads: Количество нитей в итоговом изображении")
            print("     @:param end_angles: Угол до которого делается радон)")
            return 0

    filename = "testdata/GirlwithPearl.png"
    theta = [0., 180.]
    n = 70
    if (len(sys.argv) == 1):
        onlyfiles = [f for f in listdir("testdata") if isfile(join("testdata", f))]
        print(onlyfiles)
        for file in onlyfiles:
            filename = "testdata/" + file
            result = re.search(r'/\w+.\w+', filename)
            # print(result[0] if result else 'Not found')
            name = result[0][1:len(result[0]) - 4]
            print(name)

            art = get_stringart(filename, n_strings=n, theta=theta, name=name)
            f = plt.figure(figsize=(10, 10))
            plt.axis('off')
            plt.imshow(art, cmap=plt.cm.Greys_r)
            plt.savefig("results/" + name + "_result.png")
            plt.show(block=True)
        return 0

    if (len(sys.argv) == 2):
        filename = sys.argv[1]

    if (len(sys.argv) == 3):
        filename = sys.argv[1]
        n = int(sys.argv[2])

    if (len(sys.argv) == 4):
        filename = sys.argv[1]
        n = int(sys.argv[2])
        theta = [0., float(sys.argv[3])]

    result = re.search(r'/\w+.\w+', filename)
    # print(result[0] if result else 'Not found')
    name = result[0][1:len(result[0]) - 4]
    print(name)

    art = get_stringart(filename, n_strings=n, theta=theta, name=name)
    f = plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(art, cmap=plt.cm.Greys_r)
    plt.savefig("results/" + name + "_result.png")

    return 0
  
if __name__== "__main__":
  main()

