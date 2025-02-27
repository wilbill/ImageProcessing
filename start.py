import tensorflow as tf

import numpy as np  #for PIL IMAGE PROCESSING
from PIL import Image, ImageFilter #FOR PIL IMAGE PROCESSING

#decoding image data from a file in tensorflow
def decode_image(filename, image_type, resize_shape, channels=0):
    value = tf.io.read_file(filename) #reading the image binary data from its file and returning a string of raw byte data of image
    #checking if its a png or jpeg image and decoding the image value to pixel (channel=3 =>RGB format). if channel= an int btn 0-255=>b&w or grayscale, channel=4=> RGBA
    if image_type== 'png':
        decoded_image = tf.io.decode_png(value, channels=channels)
    elif image_type=='jpeg':
        decoded_image  = tf.io.decode_jpeg(value, channels=channels)
    else:
        decoded_image = tf.io.decode_image(value, channels=channels)

    # we need a specified resize_shape, so checking if the resize_shape is valid and the image_type is also valid
    if resize_shape is not None and image_type=='png' or resize_shape is not None and image_type=='jpeg':
        decoded_image=tf.image.resize(decoded_image, resize_shape) #Resize fn takes original image decoded data n new image size w/c is a list/tuple of ints => new_height & new_width in that order

    return decoded_image

# Return a dataset created from the image file paths..
#useful if we want to deal with multiple images, not a single image, also map fn instead of "for-loop" coz its efficient and does parallel processing
def get_dataset(image_paths, image_type, resize_shape, channels):
    # CODE HERE
    filename_tensor = tf.constant(image_paths)
    dataset = tf.data.Dataset.from_tensor_slices(filename_tensor) #creating dataset from tensor of image files

    #wrapper fn to help us with image decoding to each image file in the dataset
    def _map_fn(filename):
        return decode_image(filename, image_type, resize_shape, channels=channels)
    return dataset.map(_map_fn)

#Iterator to extract data from image_dataset (ie iterator to extract data from a pixel array dataset
#fn uses iterator object to get decoded image from a dataset
def get_image_data(image_paths, image_type=None, resize_shape=None, channels=0):
    dataset = get_dataset(image_paths, image_type, resize_shape, channels)
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset) #making an iterator for the dataset
    next_image = iterator.get_next() #setting next element tensor to extract data from dataset

    #Execution now
    #we need to return a list of our image pixel data
    image_data_list = []
    with tf.compat.v1.Session() as sess: #Putting our iteration and execution code within the scope of a tf.compat.v1.Session.
        for i in range(len(image_paths)):
            image_data = sess.run(next_image)
            image_data_list.append(image_data)

    return image_data_list

#Using the PIL library to extract and modify data from an image. (we do largescale img processing in tensor, but pill is for more fine graining
#here, we will do basic resizing anf filtering in pill
def pil_resize_image(image_path, resize_shape, image_mode='RGBA', image_filter=None):
    im = Image.open(image_path)
    converted_image = im.convert(image_mode)
    resized_im = converted_image.resize(resize_shape, Image.LANCZOS)
    if image_filter is not None:
        resized_im = resized_im.filter(image_filter)

    im_data = resized_im.getdata()
    return np.asarray(im_data)




