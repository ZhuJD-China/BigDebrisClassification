import imageio
import imgaug as ia

image = imageio.imread("XieTouDataSet_Bin/normal/normal_0.jpg")

print("Original:")
ia.imshow(image)

from imgaug import augmenters as iaa

ia.seed(4)

rotate = iaa.Affine(rotate=(-5, 5))
image_aug = rotate(image=image)

print("Augmented:")
ia.imshow(image_aug)

import numpy as np
import os

images = []
path_image = os.listdir("XieTouDataSet_Bin/defect")
for i in range(len(path_image)):
    image = imageio.imread("XieTouDataSet_Bin/defect/defect_" + str(i) + ".jpg")
    images.append(image)
len(images)
images_aug = rotate(images=images)
ia.imshow(images_aug[1])

for i in range(len(image_aug)-1):
    print(i)
    imageio.imwrite("XieTouDataSet_Bin/defect_aug/defect_aug_" + str(i) + ".jpg", images_aug[i])



# seq = iaa.Sequential([
#     iaa.Crop(px=(1, 16), keep_size=False),
#     iaa.Fliplr(0.5),
#     iaa.GaussianBlur(sigma=(0, 3.0))
# ])

seq4 = iaa.Sequential([
    # iaa.CropAndPad(percent=(-0.2, 0.2), pad_mode="edge"),  # crop and pad images
    iaa.AddToHueAndSaturation((-250, 250)),  # change their color
    iaa.ElasticTransformation(alpha=1000, sigma=300),  # water-like effect
    # iaa.Cutout()  # replace one squared area within the image by a constant intensity value
], random_order=True)

images_aug4 = seq4(images=images)
ia.imshow(images_aug4[22])
for i in range(len(images_aug4)):
    print(i)
    imageio.imwrite("XieTouDataSet_Bin/defect_aug/defect_aug7_" + str(i) + ".jpg", images_aug4[i])



# shuffle(path_image)
type(path_image)

images_aug = rotate(images=path_image)

print("Augmented batch:")
ia.imshow(np.hstack(images_aug))
