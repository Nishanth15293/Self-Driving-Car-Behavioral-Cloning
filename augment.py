import numpy as np
import cv2
import matplotlib.pyplot as plt

def augment_batch(batch_samples, PICS_PATH, DISPLAY_SAMPLES=False):
    images = []
    angles = []
    correction = 0.25

    def augment_img(image, angle, left_img, left_angle, right_img, right_angle):
        images.extend([image, left_img, right_img])
        angles.extend([angle, left_angle, right_angle])

        flipped_img, flipped_ang = flip(image, angle)
        flipped_img1, flipped_ang1 = flip(left_img, left_angle)
        flipped_img2, flipped_ang2 = flip(right_img, right_angle)
        images.extend([flipped_img, flipped_img1, flipped_img2])
        angles.extend([flipped_ang, flipped_ang1, flipped_ang2])

        if DISPLAY_SAMPLES:
            plt.figure()
            plt.title("flip")
            plt.imshow(flipped_img)
            plt.show()
        for i in range(3):
            bright_img = augment_brightness(image)
            images.append(bright_img)
            angles.append(angle)
            if DISPLAY_SAMPLES:
                plt.figure()
                plt.title("bright")
                plt.imshow(bright_img)
                plt.show()

            shadow_img = augment_shadow(image)
            images.append(shadow_img)
            angles.append(angle)
            if DISPLAY_SAMPLES:
                plt.figure()
                plt.title("shadow")
                plt.imshow(shadow_img)
                plt.show()

            shift_img, shift_ang = augment_shift(image, angle)
            images.append(shift_img)
            angles.append(shift_ang)
            if DISPLAY_SAMPLES:
                plt.figure()
                plt.title("shift")
                plt.imshow(shift_img)
                plt.show()
    for batch_sample in batch_samples:
        center_img = cv2.imread(PICS_PATH + batch_sample[0].split('\\')[-1])
        left_img = cv2.imread(PICS_PATH + batch_sample[1].split('\\')[-1])
        right_img = cv2.imread(PICS_PATH + batch_sample[2].split('\\')[-1])

        center_angle = float(batch_sample[3])
        left_angle = center_angle + correction
        right_angle = center_angle - correction

        augment_img(center_img, center_angle, left_img, left_angle, right_img, right_angle)
    return images, angles

def flip(image, angle):
    img = np.copy(image)
    return np.fliplr(img), -angle

def augment_brightness(image):
    img = np.copy(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img = np.array(img, dtype=np.float64)
    random_bright = .5 + np.random.uniform()
    img[:, :, 2] = img[:, :, 2] * random_bright
    img[:, :, 2][img[:, :, 2] > 255] = 255
    img = np.array(img, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img

def augment_shadow(image):
    img = np.copy(image)
    topY = 320*np.random.uniform()
    topX = 0
    bottomX = 160
    bottomY = 320*np.random.uniform()
    hls = cv2.cvtColor(img,cv2.COLOR_RGB2HLS)
    mask = 0*hls[:,:,1]
    mgridX = np.mgrid[0:img.shape[0],0:img.shape[1]][0]
    mgridY = np.mgrid[0:img.shape[0],0:img.shape[1]][1]
    mask[((mgridX - topX) * (bottomY - topY) - (bottomX - topX) * (mgridY - topY) >= 0)] = 1
    if np.random.randint(2) == 1:
        brightness = .5
        check_one = mask == 1
        check_zero = mask == 0
        if np.random.randint(2) == 1:
            hls[:, :, 1][check_one] = hls[:, :, 1][check_one] * brightness
        else:
            hls[:, :, 1][check_zero] = hls[:, :, 1][check_zero] * brightness
    img = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)
    return img

def augment_shift(image,steer,shift_interval=0.006):
    img = np.copy(image)
    x_shift = shift_interval * np.random.uniform() - shift_interval / 2
    turn_angle = steer + x_shift / shift_interval * 2 * .2
    y_shift = 40 * np.random.uniform() - 40 / 2
    Trans_M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    (h, w) = img.shape[:2]
    img = cv2.warpAffine(img, Trans_M, (w, h))
    return img, turn_angle