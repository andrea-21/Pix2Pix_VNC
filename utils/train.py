import time
import tensorflow as tf
from IPython import display

from plots.plots import *
from models.generator import *
from models.discriminator import *

# allena su un batch singolo
@tf.function
def train_step(input_image, target, _lambda, generator, discriminator, generator_optimizer, discriminator_optimizer, sigma):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    # Aggiunta rumore alle immagini date al discriminatore
    target = target*sigma
    gen_output = gen_output*sigma

    # Loss discriminatore su immagini reali e false
    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    # Loss generatore totale (disc_loss_falsi + lamba*(loss_gen))
    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target, _lambda = _lambda)

    # Loss discriminatore totale (disc_loss_falsi + disc_loss_reali)
    disc_total_loss, disc_real_loss, disc_generated_loss = discriminator_loss(disc_real_output, disc_generated_output)

  # CALCOLO GRADIENTI E AGGIORNAMENTO PARAMETRI RETE SULLE LOSS (SEPARATI DISC E GEN)
  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_total_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))
  
  # Trasfromazione output modello in probabilità
  probs_gen = tf.sigmoid(disc_generated_output)
  probs_real = tf.sigmoid(disc_real_output)

  return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_total_loss, disc_real_loss, disc_generated_loss, probs_gen, probs_real



# train_ds è il full dataset
#rifare con epoche e batch 
def fit(train_ds, val_ds, _lambda, generator, discriminator, generator_optimizer, discriminator_optimizer, losses_history, losses_metrics, f1_score_metrics, noise_scheduler, epoch):
    example_input, example_target = next(iter(val_ds.take(1)))
    start = time.time()

    # target.shape=(8, 512, 512, 1)
    # Alla fine genera errore "W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence" ma non dovrebbe essere un bug, ma solo la fine del dataset
    for step, (input_image, target) in enumerate(train_ds):

      if (step) % 15 == 0:
         display.clear_output(wait=True)
  
         if step != 0:
           train_time = time.time()-start
           print(f'Time taken for 15 batchs: {train_time:.2f} sec\n')
           print(f'\tTime taken for each batch: {train_time/15:.2f} sec\n')

         start = time.time()
  
         # Funzione da attivare se vogliamo esempi di generazione durante il training
         generate_images(generator, example_input, example_target)
         print(f"Batch: {step}")

      # Rumore moltiplicato con input del discriminatore 
      sigma =  noise_scheduler.generate_noise_map(input_image, epoch)

      gen_total_loss, gen_gan_loss, gen_l1_loss, \
      disc_total_loss,  disc_real_loss, \
      disc_generated_loss, probs_gen, probs_real = train_step(input_image, target, _lambda, generator, discriminator, generator_optimizer, discriminator_optimizer, sigma)
      
      # Appende loss al vettore delle loss_history       
      # Appende loss alla metrica per il calcolo della loss dell'epoca attuale
      losses = {
          "gen_total": gen_total_loss,
          "gen_gan": gen_gan_loss,
          "gen_l1": gen_l1_loss,
          "disc_total": disc_total_loss,
          "disc_real": disc_real_loss,
          "disc_fake": disc_generated_loss
      }

      
      for key, value in losses.items():
          losses_history[key].append(value)
          losses_metrics[key].update_state(value)

      # Appendo i valori delle f1_score su immagini reali e su immagini generate
      probs_gen = tf.reshape(probs_gen, [probs_gen.shape[0], -1])
      probs_real = tf.reshape(probs_real, [probs_real.shape[0], -1])

      f1_score_metrics["f1_gen"].update_state(tf.zeros_like(probs_gen), probs_gen)
      f1_score_metrics["f1_real"].update_state(tf.ones_like(probs_real), probs_real)

      # Training step
      if (step+1) % 15 == 0:
        print('.', end='', flush=True)
    train_time = time.time()-start
    print(f'Time taken for last batch group: {train_time:.2f} sec\n')
    
    return losses_history, losses_metrics, f1_score_metrics


def fit_test(train_ds, val_ds, _lambda, generator, discriminator, generator_optimizer, discriminator_optimizer, losses_history, losses_metrics, f1_score_metrics, noise_scheduler, epoch):
    example_input, example_target = next(iter(val_ds.take(1)))
    start = time.time()

    # Alla fine genera errore "W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence" ma non dovrebbe essere un bug, ma solo la fine del dataset
    for step, (input_image, target) in enumerate(train_ds):
  
      print(f"Batch: {step}")
      print("INPUT IMAGES:\n")
      for input_im in enumerate(input_image):
         print(input_im)
      
      print("TARGET IMAGES:\n")
      for target_im in enumerate(target):
         print(target_im)


    train_time = time.time()-start
    print(f'Time taken for last batch group: {train_time:.2f} sec\n')
    
    return None, None, None

import tensorflow as tf
import math


class NoiseScheduler:

    def __init__(self,
                 initial_noise=0.2,
                 final_epoch=50,
                 decay_type='linear'):
        """
        Parameters
        ----------
        initial_noise : float
            Valore iniziale del fattore di rumore

        final_epoch : int
            Epoca in cui il rumore diventa ~0

        decay_type : str
            'linear' oppure 'exponential'
        """

        self.initial_noise = initial_noise
        self.final_epoch = final_epoch
        self.decay_type = decay_type.lower()

        if self.decay_type not in ['linear', 'exponential']:
            raise ValueError(
                "decay_type deve essere 'linear' o 'exponential'"
            )

    def get_noise_factor(self, epoch):

        # Decadimento lineare
        if self.decay_type == 'linear':

            factor = max(
                0.0,
                1.0 - (epoch / self.final_epoch)
            )

            return self.initial_noise * factor

        # Decadimento esponenziale
        elif self.decay_type == 'exponential':

            # k scelto in modo da avere valore ~0 a final_epoch
            k = 5.0 / self.final_epoch

            return self.initial_noise * math.exp(-k * epoch)

    def generate_noise_map(self, images, epoch):
        """
        Genera la mappa di rumore.

        Parameters
        ----------
        images : tensor
            Batch immagini
            Shape: (batch, 512, 512, 1)

        epoch : int

        Returns
        -------
        noise_map : tensor
            Stessa shape di images
        """

        noise_factor = self.get_noise_factor(epoch)

        noise_map = tf.random.normal(
            shape=tf.shape(images),
            mean=1.0,
            stddev=noise_factor,
            dtype=images.dtype
        )

        return noise_map
