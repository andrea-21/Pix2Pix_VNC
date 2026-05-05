
import tensorflow as tf

def Generator(normalizationFunction = ['standard'], _f = 2, n_canali = 1):
    #inputs = tf.keras.Input((image_h, image_w,1))
    # Versione meno complessa, in alternativa vanno cambiati i conv2D con conv3D
    inputs = tf.keras.layers.Input(shape=(512, 512, n_canali))
     
    x = tf.keras.layers.ZeroPadding2D(((0, 0), (0, 0)))(inputs)
    x = tf.keras.layers.Conv2D(16*_f, (3,3), padding = "same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(16*_f, (3,3), padding = "same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    a1 = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.MaxPooling2D((2,2))(a1)
    #-->(128,128,16)
    x = tf.keras.layers.Conv2D(32*_f, (3,3), padding = "same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(32*_f, (3,3), padding = "same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    a2 = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.MaxPooling2D((2,2))(a2)
    #-->(64,64,32)
    x = tf.keras.layers.Conv2D(48*_f, (3,3), padding = "same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(48*_f, (3,3), padding = "same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    a3 = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.MaxPooling2D((2,2))(a3)
    #-->(32,32,48)
    x = tf.keras.layers.Conv2D(64*_f, (3,3), padding = "same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(64*_f, (3,3), padding = "same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    a4 = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.MaxPooling2D((2,2))(a4)

    #Middle
    x = tf.keras.layers.Conv2D(64*_f, (3,3), padding = "same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(64*_f, (3,3), padding = "same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    #Middle
    x = tf.keras.layers.UpSampling2D((2,2))(x)
    x = tf.keras.layers.Concatenate()([x, a4])

    x = tf.keras.layers.Conv2D(64*_f, (3,3), padding = "same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(64*_f, (3,3), padding = "same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.UpSampling2D((2,2))(x)
    x = tf.keras.layers.Concatenate()([x, a3])

    x = tf.keras.layers.Conv2D(48*_f, (3,3), padding = "same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(48*_f, (3,3), padding = "same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.UpSampling2D((2,2))(x)
    x = tf.keras.layers.Concatenate()([x, a2])

    x = tf.keras.layers.Conv2D(32*_f, (3,3), padding = "same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(32*_f, (3,3), padding = "same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.UpSampling2D((2,2))(x)
    x = tf.keras.layers.Concatenate()([x, a1])

    x = tf.keras.layers.Conv2D(16*_f, (3,3), padding = "same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(16*_f, (3,3), padding = "same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(1,(1,1), padding = "same")(x)
    #x = tf.keras.layers.Cropping2D(((0, 0), (0, 0)))(x)
    if normalizationFunction[0] == 'standard':
        x = tf.keras.layers.Activation("linear")(x)
    elif normalizationFunction[0] == 'minmax':
        x = tf.keras.layers.Activation("sigmoid")(x)
    return tf.keras.Model(inputs, x)


#model = unet_like(image_h = train_y.shape[1], image_w = train_y.shape[2], n_classes = train_y.shape[3])
# model = unet_like(image_h = 32, image_w = 32, n_classes = 11)

#model = unet_like(image_h = 512, image_w = 512)

def CreateModel(n_canali = 1):
    model = Generator(n_canali)
    
    model.summary()
    
    return model

#Generator loss

def generator_loss(disc_generated_output, gen_output, target, _lambda = 700, loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # Mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  total_gen_loss = gan_loss + (_lambda * l1_loss)

  return total_gen_loss, gan_loss, l1_loss

# LOSS ESPERIMENTO CON HINGE AL POSTO DI CROSS (BOCCIATA?)
def generator_loss2(disc_generated_output, gen_output, target, _lambda = 700):
    gan_loss = -tf.reduce_mean(disc_generated_output)  # Hinge loss

    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))  # L1 loss for structure

    total_gen_loss = gan_loss + (_lambda * l1_loss)  # Keep L1 but scale adaptively

    return total_gen_loss, gan_loss, l1_loss