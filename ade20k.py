"""Pascal ADE20K Semantic Segmentation Dataset."""
import os
import torch
import numpy as np

from PIL import Image
import torch.utils.data as data

def _get_ade20k_pairs(folder, mode='train'):
    img_paths = []
    mask_paths = []
    if mode == 'train':
        img_folder = os.path.join(folder, 'images/training')
        mask_folder = os.path.join(folder, 'annotations/training')
    else:
        img_folder = os.path.join(folder, 'images/validation')
        mask_folder = os.path.join(folder, 'annotations/validation')
    filename_list=os.listdir(img_folder)
    filename_list.sort()
    for filename in filename_list:
        basename, _ = os.path.splitext(filename)
        if filename.endswith(".jpg"):
            imgpath = os.path.join(img_folder, filename)
            maskname = basename + '.png'
            maskpath = os.path.join(mask_folder, maskname)
            if os.path.isfile(maskpath):
                img_paths.append(imgpath)
                mask_paths.append(maskpath)
            else:
                print('cannot find the mask:', maskpath)

    return img_paths, mask_paths



class ADE20KSeg(data.Dataset):
    id_map={
                    4:{"class":1,"name":"floor","color":(0, 128, 0)},
                    29:{"class":2,"name":"carpet","color":(0, 0, 128)},
                    255:{"class":0,"name":"background","color":(111, 74, 0)},
                }
    train_id_to_color=[(v["class"],v["color"]) for k,v in id_map.items()]
    train_id_to_color.sort(key=lambda x : x[0])
    train_id_to_color=list(map(lambda x : x[1],train_id_to_color))
    train_id_to_color=np.array(train_id_to_color)
    def __init__(self,
                 root,
                 split='train',
                 transform=None):

        assert os.path.exists(root), "Please setup the dataset using ../datasets/ade20k.py"
        self.images, self.masks = _get_ade20k_pairs(root, split)
        assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of:" + root + "\n")
        print('Found {} images in the folder {}'.format(len(self.images), root))
        self.transform=transform

    def getitem(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])
        return img, target
    
    def __getitem__(self,index):
        # size_=0
        # while size_<257:
        #     img, target=self.getitem(index)
        #     width, height = img.size
        #     size_=min(width,height)
        #     index=np.random.randint(self.__len__())

        img, target=self.getitem(index)
        if self.transform is not None:
            img, target = self.transform(img, target)
            target=self.encode_target(target)
        return img, target

    def __len__(self):
        return len(self.images)

    @classmethod
    def encode_target(cls, mask):
        vectorized_map = np.vectorize(lambda x: cls.id_map[x]["class"] if x in cls.id_map else x)
        new_mask=vectorized_map(mask)
        return new_mask

    @classmethod
    def decode_target(cls, target):
        # target[target == 255] = 19
        #target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]

if __name__ == '__main__':
    import cv2
    train_dataset = ADE20KSeg("/Users/eureka/Documents/cv/dataset/ADEChallengeData2016",split="validation")
    
    #example
    a=train_dataset.__getitem__(30)
    image,mask=a
    image=np.array(image)
    mask=np.array(mask)
    print(mask)
    print(train_dataset.__len__())
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # #遍历
    # min_value=9999
    # count=0
    # for i in range(train_dataset.__len__()):
    #     image,mask=train_dataset.__getitem__(i)

    #     image=np.array(image)
    #     mask=np.array(mask)
    #     w,h,_ = image.shape
    #     size_=min(w,h)
    #     if size_<256:
    #         count+=1
    # print(count)
    