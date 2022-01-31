from skimage.io import imread
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tensorflow_examples.models.pix2pix import pix2pix
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import glob
import h5py
import ast
import os
from .models import Image as ImageModel
from django.core.files import File


class LaiUnet():

    def __init__(self,
                 dirs    = [''],
                 rand    = None,
                 repeat  = 1,
                 split   = 0.8,
                 batch   = 10 ,
                 name    = 'Lai',
                 mode    = 'init',
                 height  = 256,
                 width   = 256,
                 cropH   = 1,
                 cropW   = 1,
                 ):

        self.height = height
        self.width  = width
        self.mode = mode
        self.dirs = dirs
        self.rand = rand
        self.repeat = repeat
        self.split  = split
        self.batch  = batch
        self.name   = name
        self.dataset_train = []
        self.dataset_test  = []
        self.cropH = cropH
        self.cropW = cropW
        self.tf_train      = None
        self.tf_test       = None

        self.Init_Dir()
        self.make_dataset()
        self.make_model()

    def Init_Dir(self):

        def raw_url(url):
            return url.replace('_In.jpg','')

        for current_dir in self.dirs:
            self.dataset_train += list(map(raw_url,glob.glob(os.path.join(current_dir,'*_In.jpg'))))*( self.repeat if (self.repeat,int) else 1 )

        if len(self.dataset_train) > 0:
            self.dataset_train,self.dataset_test = ( self.dataset_train[ : int(len(self.dataset_train)*self.split) ],
                                                    self.dataset_train[ int(len(self.dataset_train)*self.split) : ])

        if self.rand is not None:
            random.Random( self.rand if isinstance(self.rand, int) else np.random.randint(1000) ).shuffle(self.dataset_train)

    def make_dataset(self):

        if len(self.dataset_train) > 0:
            self.tf_train = tf.data.Dataset.from_tensor_slices(self.dataset_train)
            self.tf_train = self.tf_train.map(self.load_image_train, num_parallel_calls  = -1)
            self.tf_train = self.tf_train.shuffle(100).batch(self.batch)

        if len(self.dataset_test) > 0:
            self.tf_test = tf.data.Dataset.from_tensor_slices(self.dataset_test)
            self.tf_test = self.tf_test.map(self.load_image_test, num_parallel_calls = -1)
            self.tf_test = self.tf_test.batch(self.batch)

    @tf.function
    def load_image_train(self, filename):

        in_img = tf.cast(tf.image.decode_jpeg(tf.io.read_file(filename+'_In.jpg'), channels = 3), tf.float32)
        ou_img = tf.cast(tf.image.decode_jpeg(tf.io.read_file(filename+'_Ou.jpg'), channels = 3), tf.float32)


        height = tf.cast( tf.cast(tf.shape(in_img)[0], tf.float32)*self.cropH,  dtype=tf.int32) if self.cropH < 1 else self.cropH
        width  = tf.cast( tf.cast(tf.shape(in_img)[1], tf.float32)*self.cropW,  dtype=tf.int32) if self.cropW < 1 else self.cropW


        img_stacked = tf.stack([in_img,ou_img], axis = 0)
        cropped = tf.image.random_crop( img_stacked, size = [2, height, width ,3])

        in_img, ou_img = cropped[0], cropped[1]

        in_img = tf.image.resize( in_img , (self.height+30,self.width+30) )# , method = 'nearest')
        ou_img = tf.image.resize( ou_img , (self.height+30,self.width+30) )# , method = 'nearest')
        img_stacked = tf.stack([in_img,ou_img], axis = 0)
        cropped = tf.image.random_crop( img_stacked, size = [2,self.height,self.width,3])

        in_img, ou_img = cropped[0], cropped[1]

        in_img = in_img/255.0
        ou_img_f = (tf.round( tf.image.rgb_to_grayscale(tf.cast(ou_img, tf.float32))/255.0)-1)*-1

        return in_img, ou_img_f

    @tf.function
    def load_image_test(self, filename):

        in_img = tf.cast(tf.image.decode_jpeg(tf.io.read_file(filename+'_In.jpg'), channels = 3), tf.float32)
        ou_img = tf.cast(tf.image.decode_jpeg(tf.io.read_file(filename+'_Ou.jpg'), channels = 3), tf.float32)

        height = tf.cast( tf.cast(tf.shape(in_img)[0], tf.float32)*self.cropH,  dtype=tf.int32) if self.cropH < 1 else self.cropH
        width  = tf.cast( tf.cast(tf.shape(in_img)[1], tf.float32)*self.cropW,  dtype=tf.int32) if self.cropW < 1 else self.cropW

        img_stacked = tf.stack([in_img,ou_img], axis = 0)
        cropped = tf.image.random_crop( img_stacked, size = [2, height, width ,3])

        in_img, ou_img = cropped[0], cropped[1]

        in_img = tf.image.resize( in_img , (self.height+30,self.width+30) )#, method = 'nearest')
        ou_img = tf.image.resize( ou_img , (self.height+30,self.width+30) )#, method = 'nearest')
        img_stacked = tf.stack([in_img,ou_img], axis = 0)
        cropped = tf.image.random_crop( img_stacked, size = [2,self.height,self.width,3])

        in_img, ou_img = cropped[0], cropped[1]

        in_img = in_img/255.0
        ou_img_f = (tf.round( tf.image.rgb_to_grayscale(tf.cast(ou_img, tf.float32))/255.0)-1)*-1

        return in_img, ou_img_f

    def make_model(self, mode = 'init'):

        generator_Pix2Pix = pix2pix.unet_generator(3, norm_type='instancenorm')
        input_net  = tf.keras.layers.InputLayer(input_shape=[self.height , self.width , 3])
        output_net = tf.keras.layers.Conv2D(2, 3, strides = 1, padding='same')
        self.loss  = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
        self.opt   = tf.keras.optimizers.Adam()
        self.model = tf.keras.Sequential([ input_net , generator_Pix2Pix, output_net ])
        if mode == 'init':
            self.model.compile(optimizer = self.opt, loss = self.loss, metrics=['accuracy'])

    def save_model(self, path, remove = True, save_optimizer = True, attrs = []):
        if os.path.exists(path) and remove:
            os.remove(path)
            print('Dropping previous Model...')
        info = {}
        for attr in attrs:
            info[attr] = getattr(self, attr)
        keys = str(list(info.keys()))
        self.model.save_weights( path )
        h5 = h5py.File( path, 'a')
        h5.attrs.create('info', keys)
        for key in info.keys():
            h5.attrs.create( key, str(info[key]) )                
        h5.close()
        if save_optimizer:
            optimizer=tf.keras.backend.batch_get_value(self.model.optimizer.weights)
            opt_path = os.path.join(os.path.dirname(path),'-opt.'.join(os.path.basename(path).split('.')))
            if os.path.exists(opt_path) and remove:
                os.remove(opt_path); print('Dropping Optimizator...')
            h5 = h5py.File( opt_path , 'a')
            h5.attrs.create('opt.', str(len(optimizer)))
            for ido,opt in enumerate(optimizer):
                shape = opt.shape if ido != 0 else (1,)
                h5.create_dataset('opt_'+str(ido+1), data=opt, shape=shape,chunks=shape,
                                    compression='gzip', compression_opts=9)
            h5.close()

    def load_model( self, path, load_optimizer = True):

        if not os.path.isfile(path):
            print('Model is not Loaded, Path not exists.')
            return None

        h5 = h5py.File( path, 'r')
        if 'info' in h5.attrs:
            info = ast.literal_eval(h5.attrs['info'])
            for attr in info:
                setattr(self, attr, ast.literal_eval(h5.attrs[attr]))
        h5.close()
        opt_path=os.path.join(os.path.dirname(path),'-opt.'.join(os.path.basename(path).split('.')))
        if load_optimizer and os.path.exists(opt_path):
            h5 = h5py.File(opt_path,'r')
            len_opt = ast.literal_eval(h5.attrs['opt.'])
            optimizer = []
            for x in range(len_opt):
                optimizer.append(h5.get('opt_'+str(x+1))[()])
            optimizer[0] = optimizer[0].reshape(())
            h5.close()
            if len(self.model.optimizer.get_weights()) == 0:
                rand_in = np.random.rand(1,self.height,self.width,3)
                rand_ou = np.random.rand(1,self.height,self.width)
                self.model.fit( x = rand_in , y = rand_ou, verbose = 0)
            self.model.optimizer.set_weights(optimizer)
        self.model.load_weights(path)

    def make_predictions(self, tf_data = 'tf_test', counter = 5, threshold = 0.5):

        dataset = getattr(self, tf_data, None)
        if dataset is None: return

        plt.rcParams['figure.figsize'] = [20, 12]; 
        cc  = 0
        pos = 1

        for In,Ou in dataset:
            pred = tf.image.grayscale_to_rgb(tf.cast(tf.nn.softmax(self.__call__(In),
                                            axis = 3)[...,0:1]> threshold, tf.float32))
            Ou = tf.image.grayscale_to_rgb( tf.cast( Ou < threshold, tf.float32 ) )
            for i in range(In.shape[0]):
                c_pred = pred[i,...]; c_in   =   In[i,...]; c_ou   =   Ou[i,...]
                final  = tf.concat(( c_in, c_ou, c_pred ), axis = pos )
                plt.imshow(final); plt.show()
                cc += 1
                if cc == counter: return

    def fit(self,epochs = 10):
        self.model.fit( self.tf_train, 
                        epochs = epochs , 
                        validation_data = self.tf_test , 
                        steps_per_epoch = np.ceil(len(self.dataset_train) / self.batch) )

    def __call__(self, Input):
        return self.model.predict(Input)

def Predict_LAI(model, img, coord = [], plot = False):

    if len(coord) == 0:
        y,x,h,w = (0,0,) + img.shape[:2]
    else:
        y,x,h,w = tuple( coord)

    sub_img = img[y:h, x:w, :]
    h,w,_   = tuple( sub_img.shape )
    img_r   = tf.cast( tf.image.resize( sub_img, (256,256), method = 'bicubic' ) , tf.float32 )[None,...]/255.0
    output = tf.image.resize( ( model( img_r )[0,:,:,0:1] > 0.5 ).astype('float32') , (h,w) ).numpy()[...,0]
    output = cv2.dilate(output,np.ones((3,3)),iterations = 3)
    FVC    = np.round( ( output>=0.5 ).sum()/(h*w)*100,2 )
    plt.imshow(output)

    distance_1 = np.sqrt( ( img.shape[0]/2 - (h-y)/2)**2 + ( img.shape[1]/2 - (w-x))**2 )*4.33/1000
    distance_2 = 10
    angle      = np.round( np.arctan(distance_2/distance_1) ,2)
    print('Angle:',  np.round(angle*180/np.pi,2),'degree', angle ,'radian')
    print('FVC: ', FVC, '%' )
    
    m = 0.077
    
    LAI = FVC*m + angle/np.pi
    
    print('LAI: ', np.round(LAI,2))

    detected_output_image = sub_img
    output_image = tf.image.grayscale_to_rgb( tf.cast(output[...,None], 'uint8')*255).numpy()
    # plt.rcParams['figure.figsize'] = [20, 12]
    # plt.imshow( tf.concat( ( sub_img,tf.zeros((h,50,3), dtype='uint8'), tf.image.grayscale_to_rgb( tf.cast(output[...,None], 'uint8')*255).numpy() ) , axis = 1) )
    # plt.figure()
    # plt.rcParams['figure.figsize'] = [6, 6]
    # plt.imshow( img )
    # plt.show()
    # plt.figure()

    coords = (x,y,h,w)
    
    return LAI, FVC, detected_output_image, output_image, coords


def detect(image_path, image_id, X, Y, Latitude, Longitude):
    SITE_ROOT = os.path.dirname(os.path.realpath(__file__)).replace('\\', '/')
    app_path = SITE_ROOT
    main_folder = SITE_ROOT.split('/')[-1]
    SITE_ROOT = SITE_ROOT.replace(main_folder, '')
    SITE_ROOT = SITE_ROOT[:-1]

    dirs = ['']
    Lai = LaiUnet( 
                    dirs   = dirs,
                    rand   = 4,
                    repeat = 100,
                    split  = 0.8,
                    batch  = 10,
                    name   = 'Lai',
                    mode   = 'init',
                    cropH  = 0.45,
                    cropW  = 0.125, 
                )

    Lai_model = Lai.load_model(app_path + '/' + 'Lai_model.h5')
    img = imread(SITE_ROOT + image_path) #Input Image
    # box_img = [1745,1600,2701,1965] # [x, y, x+width, y+height, ]
    box_img = [X, Y, Latitude, Longitude] # [x, y, x+width, y+height, ]
    
    LAI, FVC, detected_output_image, output_image, coords = Predict_LAI(Lai, img, coord = box_img)

    file_extension = image_path.split('.')[-1]
    image_path = image_path[:-len(file_extension)]
    image_path = image_path[:-1]
    
    cv2.imwrite(SITE_ROOT + image_path + '_detected_output.jpg', detected_output_image)
    cv2.imwrite(SITE_ROOT + image_path + '_output.jpg', output_image)

    output_image_path = image_path + '_output.jpg'
    detected_output_image_path = image_path + '_detected_output.jpg'

    image_object = ImageModel.objects.get(id=image_id)
    image_object.path = image_path + '.jpg'
    
    image_object.LAI = LAI
    image_object.FVC = FVC
    image_object.X = X
    image_object.Y = Y
    image_object.Latitude = Latitude
    image_object.Longitude = Longitude
    
    image_object.detected_path = image_path + '_detected_output.jpg'
    image_object.output_path = image_path + '_output.jpg'

    image_object.detected_output_image_file = File(detected_output_image)
    image_object.detected_image_file = File(output_image)
    image_object.save()

    return detected_output_image_path, output_image_path, LAI, FVC, coords
