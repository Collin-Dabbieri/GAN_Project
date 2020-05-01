# Similar to convgan.py but instead making it a Wasserstein GAN
# The general motivation here is that with a vanilla GAN the discriminator is attempting to give a probability that a given image is real
# If the discriminator is very good at knowing when the generator is giving it a fake image, it will give probabilities very close to zero
# With sigmoid activation this will have a very small gradient, so if the discriminator gets smarter than the generator, it will be hard for the generator to catch up
# In a Wassersteing GAN, the discriminator is replaced with a critic that scores images on their realness, real images are given positive scores
# and fake images are given negative scores
# Loss=-(1/n)SUM{y_i p_i}
# y_i=1 if real
# y_i=-1 if fake
# p_i is the score
# the critic wants labels and scores to have the same sign
# Problems:
#  - the function is unbounded, the critic is incentivized to pick high p for true images and highly negative p for fake images
#    this can create giant gradients for the generator
# Solutions:
#  - could clip weights to be in +- 0.01
#  - could train critic for 5 epochs, then train the generator for 1 epoch


import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import pickle
from tensorflow.keras.layers import InputLayer, Dense, LSTM, Dropout, Activation, Conv2D, Flatten, Reshape, Conv2DTranspose, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import backend
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.constraints import Constraint
import argparse

FONTSIZE=18
plt.rcParams['figure.figsize']=(10,10)
plt.rcParams['font.size']=FONTSIZE


#Wasserstein loss and ClipConstraint class are from https://machinelearningmastery.com/how-to-code-a-wasserstein-generative-adversarial-network-wgan-from-scratch/
def wasserstein_loss(y_true,y_pred):
    '''
    Define Wasserstein Loss Function
    '''
    return backend.mean(y_true*y_pred)

class ClipConstraint(Constraint):
    '''
    This class will be used as a Keras constraint
    it will clip the weights to +-constraint after each mini-batch
    '''
    def __init__(self,clip_value):
        self.clip_value=clip_value
    def __call__(self,weights):
        return backend.clip(weights, -self.clip_value, self.clip_value)
    def get_config(self):
        return {'clip_value': self.clip_value}

def build_critic(args):
    '''
    clip_weight - Clip limit value for weights
    init_weight_sigma - standard deviation of RandomNormal initialization of weights
    critic_filters - list with number of filters for each conv layer of critic
    critic_kernels - list of kernel sizes for each conv layer of critic
    critic_strides - list of strides for each conv layer of critic
    '''
    
    if not args.load_pretrained:
        const = ClipConstraint(args.clip_weight)
        init=RandomNormal(stddev=args.init_weight_sigma)

        model=Sequential()

        input_shape=(args.image_height,args.image_width,args.num_channels)

        for i in range(len(args.critic_filters)):
            if i==0: 
                model.add(Conv2D(filters=args.critic_filters[i],
                                 kernel_size=(args.critic_kernels[i],args.critic_kernels[i]),
                                 strides=(args.critic_strides[i],args.critic_strides[i]),
                                 padding='valid',
                                 kernel_constraint=const,
                                 input_shape=input_shape,
                                 kernel_initializer=init,
                                 name='Critic_Conv_'+str(i+1)))
                model.add(LeakyReLU(alpha=0.2))
            else:
                model.add(Conv2D(filters=args.critic_filters[i],
                                 kernel_size=(args.critic_kernels[i],args.critic_kernels[i]),
                                 strides=(args.critic_strides[i],args.critic_strides[i]),
                                 padding='valid',
                                 kernel_constraint=const,
                                 kernel_initializer=init,
                                 name='Critic_Conv_'+str(i+1)))
                model.add(LeakyReLU(alpha=0.2))

        model.add(Flatten())
        model.add(Dense(1,activation='linear',name='Critic_Output'))
        # the critic has a linear activation function in the output to give a 'realness score' to the image

        opt = RMSprop(lr=args.lrate)
        model.compile(loss=wasserstein_loss,optimizer=opt)
        
    elif args.load_pretrained:
        
        model=load_model(args.pretrained_critic_name)
        opt = RMSprop(lr=args.lrate)
        model.compile(loss=wasserstein_loss,optimizer=opt)
    
    return model

def build_generator(args):
    '''
    init_weight_sigma - standard deviation for initialized weight values
    generator_filters - list of filters for the initial dense layer and the subsequent Conv2DTranspose layers
    generator_starting_shape - (rows,columns) size of Dense layer image upsampled from latent space
    generator_kernel_height - list with heights of conv kernels
    generator_kernel_width - list with widths of conv kernels
    generator_stride_height - list with heights of conv strides
    generator_stride_width - list with widths of conv strides
    '''
    if not args.load_pretrained:
        init=RandomNormal(stddev=args.init_weight_sigma)

        model=Sequential()

        for i in range(len(args.generator_filters)):

            if i==0:
                #first we upsample the latent space to a single dense layer
                n_nodes=args.generator_starting_shape[0]*args.generator_starting_shape[1]*args.generator_filters[i]
                model.add(Dense(units=n_nodes,
                                kernel_initializer=init,
                                input_dim=args.latent_dim,
                                name='Generator_Dense'))
                model.add(LeakyReLU(alpha=0.2))
                model.add(Reshape((args.generator_starting_shape[0],args.generator_starting_shape[1],args.generator_filters[i])))

            else:
                #kernel size should be a factor of the stride size
                model.add(Conv2DTranspose(filters=args.generator_filters[i],
                                          kernel_size=(args.generator_kernel_height[i-1],args.generator_kernel_width[i-1]),
                                          strides=(args.generator_stride_height[i-1],args.generator_stride_width[i-1]),
                                          padding='same',
                                          kernel_initializer=init,
                                          name='Generator_Conv_'+str(i)
                                          ))
                model.add(LeakyReLU(alpha=0.2))

        #collapse it back to one image
        model.add(Conv2D(filters=args.num_channels,
                         kernel_size=(7,7),
                         activation='tanh',
                         padding='same',
                         kernel_initializer=init))
        
    elif args.load_pretrained:
        
        model=load_model(args.pretrained_gen_name)
    
    return model

def build_gan(critic,generator,args):
    '''
    critic - critic Sequential model
    generator - generator Sequential model
    lrate - learning rate for training
    '''
    critic.trainable=False #in the GAN step, only the generator should be trained
    model=Sequential()
    model.add(generator)
    model.add(critic)
    opt = RMSprop(lr=args.lrate)
    model.compile(loss=wasserstein_loss, optimizer=opt)
    return model
    
def generate_filename(args):
    '''
    this should not include the directory for the file, just the name that comes from the chosen hyperparams
    '''
    gen_strides_str='Gstrides_'
    for i in range(len(args.generator_stride_height)):
        height=args.generator_stride_height[i]
        width=args.generator_stride_width[i]
        gen_strides_str+=str(height)
        gen_strides_str+='-'
        gen_strides_str+=str(width)
        gen_strides_str+='_'
    
    critic_strides_str='Cstrides_'
    for i in args.critic_strides:
        critic_strides_str+=str(i)
        critic_strides_str+='_'
    
    gen_filters_str='Gfilters_'
    for i in args.generator_filters:
        gen_filters_str+=str(i)
        gen_filters_str+='_'
    
    critic_filters_str='Cfilters_'
    for i in args.critic_filters:
        critic_filters_str+=str(i)
        critic_filters_str+='_'
        
    critic_boost_str='Cboost_'+str(args.critic_boost)+'_'
    
    weight_clip_str='WeightClip_'+str(args.clip_weight)
    
        
    fbase='WassGan_'+args.mapping+'_'+gen_filters_str+critic_filters_str+gen_strides_str+critic_strides_str+critic_boost_str+weight_clip_str
    
    return fbase

def generate_real_samples(ins,sample_size):
    '''
    ins-training data
    sample_size - number of samples
    '''
    
    idx_all=np.arange(ins.shape[0])
    idx_pick=np.random.choice(idx_all,size=sample_size,replace=False)
    
    X=ins[idx_pick,:,:]
    y=np.ones((sample_size, 1)) #real images have a value of 1
    
    return X,y
    

def generate_fake_samples(gen_model,latent_dim,sample_size):
    '''
    gen_model - generator Sequential model
    latent_dim - size of the latent dimension
    sample_size - number of samples
    '''
    
    gen_input=generate_latent_space(latent_dim,sample_size)
    X=gen_model.predict(gen_input)
    y=-np.ones((sample_size,1)) #fake images have a value of -1
    
    return X,y

def generate_latent_space(latent_dim,sample_size):
    '''
    latent_dim - size of the latent dimenstion
    sample_size - number of samples
    '''
    
    gen_input=np.random.randn(latent_dim*sample_size)
    gen_input=gen_input.reshape((sample_size,latent_dim))
    
    return gen_input

def plot_images(generator,latent_space,filename_base,epoch,args,n=5):
    '''
    generator - generator Sequential model
    latent_space - latent space for generator
    filename_base 
    epoch
    plots_dir - path to directory for saving plots
    '''
    
    generated_plots=generator.predict(latent_space)
    
    fig,axs=plt.subplots(n,n)
    count=-1
    for i in range(n):
        for j in range(n):
            count+=1
            #turn off axis
            axs[i,j].axis('off')
            #plot first measure of image
            axs[i,j].imshow(generated_plots[count, :, :, 0])
    #reduce space between plots
    plt.subplots_adjust(wspace=0.01,hspace=0.01)        
    fig.suptitle('Epoch {:.0f}/{:.0f}'.format(epoch+1,args.epochs))
    epoch_str=str(epoch+1).zfill(3) #this turns 1 into 001
    plot_path=args.plots_dir+filename_base+'_e_'+str(epoch_str)+'.png'
    plt.savefig(plot_path,bbox_inches='tight')
    plt.clf()

def execute_exp(args):
    '''
    -fname_train - filename for ins
    -batch_size - size of training batches
    -epochs - number of training epochs
    -critic_boost - number of training batches of the critic for each batch of the generator
    -latent_dim - size of the latent space for the generator
    '''
    
    critic_model=build_critic(args)
    gen_model=build_generator(args)
    gan_model=build_gan(critic_model,gen_model,args)
    
    fp = open(args.fname_train, "rb")
    ins = pickle.load(fp)
    fp.close()
    assert(ins.shape[1]==args.image_height),"image shape in ins must match image shape in args"
    assert(ins.shape[2]==args.image_width),"image shape in ins must match image shape in args"
    assert(ins.shape[3]==args.num_channels),"image shape in ins must match image shape in args"
    
    half_batch=int(args.batch_size/2)
    batches_per_epoch=int(ins.shape[0]/args.batch_size)
    
    #we'll generate one latent space for all of our saved plots
    #so we can see how generated images for the same latent space evolve over time
    #25 because the plots will be a 5x5 grid
    latent_space_plots=generate_latent_space(args.latent_dim,25)
    
    filename_base=generate_filename(args) #filename for saved results, models, plots

    #save some of the results that won't change during training
    results={}
    results['critic_summary']=critic_model.summary()
    results['generator_summary']=gen_model.summary()
    results['args']=args
    results['fbase']=filename_base
    
    critic_real_losses=[]
    critic_fake_losses=[]
    generator_losses=[]
    
    
    for i in range(args.epochs): #batches will be stochastically generated so this isn't strictly necessary
        for j in range(batches_per_epoch): #but expressing things in terms of epochs and batch_size is better intuitively
            #plus this will let us save checkpoints after specified epochs
        
            #train critic
            for boost in range(args.critic_boost):
                #the critic is trained more times than the generator because WGANs create large gradients for the generator
                X_real,y_real=generate_real_samples(ins,half_batch) #generate real images
                critic_loss_real=critic_model.train_on_batch(X_real,y_real) #train critic on real images
                critic_real_losses.append(critic_loss_real)
                
                X_fake,y_fake=generate_fake_samples(gen_model,args.latent_dim,half_batch) #generate fake images
                critic_loss_fake=critic_model.train_on_batch(X_fake,y_fake) #train critic on fake images
                critic_fake_losses.append(critic_loss_fake)

            #train generator
            X_gan=generate_latent_space(args.latent_dim,args.batch_size)
            y_gan=np.ones((args.batch_size, 1)) #fake images are supposed to have a label of -1
            # but we give inverted labels here because we want to reward the generator when the critic is mistaken
            gen_loss=gan_model.train_on_batch(X_gan,y_gan)
            generator_losses.append(gen_loss)
            
            if args.verbose:
                print('Epoch: %d, Batch: %d/%d, C loss real: %f, C loss fake: %f, G loss: %f' % (i+1,j+1,batches_per_epoch,critic_loss_real,critic_loss_fake,gen_loss))
            
        #plot some generated images after each epoch
        plot_images(gen_model,latent_space_plots,filename_base,i,args)
            
        if (i+1) in args.checkpoints:
            #if this epoch is a checkpoint, save models and results
            
            #save results
            results['critic_real_loss']=critic_real_losses
            results['critic_fake_loss']=critic_fake_losses
            results['generator_loss']=generator_losses
            
            fp=open(args.results_dir+"%s_results_e%03d.pkl" % (filename_base,i+1),'wb')
            pickle.dump(results,fp)
            fp.close()
            
            #save models
            generator_filename = args.results_dir+'%s_Gen_epoch_%03d.h5' % (filename_base,i+1)
            gen_model.save(generator_filename)
            
            critic_filename=args.results_dir+'%s_Critic_epoch_%03d.h5' % (filename_base,i+1)
            critic_model.save(critic_filename)
        

def create_parser():
    parser=argparse.ArgumentParser(description='WGAN')
    
    #general info
    parser.add_argument('-image_height',type=int,default=96,help='Number of Rows in the image')
    parser.add_argument('-image_width',type=int,default=96,help='Number of Columns in the image')
    parser.add_argument('-num_channels',type=int,default=4,help='Number of Image Channels')
    parser.add_argument('-fname_train',type=str,default='ins_convgan_final_fantasy_fifths.pkl',help='filename for ins')
    parser.add_argument('-epochs',type=int,default=750,help='Number of Training Epochs')
    parser.add_argument('-batch_size',type=int,default=64,help='Training Batch Size')
    parser.add_argument('-lrate',type=float,default=0.00005,help='Learning Rate')
    parser.add_argument('-verbose',default=True,help='Boolean for verbosity of experiment')
    parser.add_argument('-init_weight_sigma',type=float,default=0.02,help='Standard Deviation of RandomNormal initialization of weights')
    parser.add_argument('-clip_weight',type=float,default=0.01,help='Clip limit value for weights')
    parser.add_argument('-plots_dir',type=str,default='./plots/',help='path to directory for plots')
    parser.add_argument('-results_dir',type=str,default='./results/',help='path to directory for results')
    parser.add_argument('-mapping',type=str,default='fifths',help='mapping of ins (linear or fifths)')
    parser.add_argument('-checkpoints',nargs='+',type=int,default=[1,50,100,150,200,250,300,350,400,450,500,550,600,650,700,750],help='model checkpoints')
    
    #critic info
    parser.add_argument('-critic_filters',nargs='+',type=int,default=[64,64],help='number of filters in each conv layer of critic')
    parser.add_argument('-critic_kernels',nargs='+',type=int,default=[2,4],help='list of kernel sizes for each conv layer of critic')
    parser.add_argument('-critic_strides',nargs='+',type=int,default=[2,2],help='list of strides for each conv layer of critic')
    parser.add_argument('-critic_boost',type=int,default=5,help='train critic for this many batches for each batch of gen training')
    
    #generator_info
    parser.add_argument('-latent_dim',type=int,default=20,help='Size of Generator Latent Space')
    parser.add_argument('-generator_starting_shape',nargs='+',type=int,default=[4,12],help='(rows,columns) size of Dense layer image upsampled from latent space')
    parser.add_argument('-generator_filters',nargs='+',type=int,default=[48,48,48],help='list of filters for the initial dense layer and the subsequent Conv2DTranspose layers')
    parser.add_argument('-generator_kernel_height',nargs='+',type=int,default=[12,2],help='list with heights of conv kernels')
    parser.add_argument('-generator_kernel_width',nargs='+',type=int,default=[2,4],help='list with widths of conv kernels')
    parser.add_argument('-generator_stride_height',nargs='+',type=int,default=[12,2],help='list with heights of conv strides')
    parser.add_argument('-generator_stride_width',nargs='+',type=int,default=[2,4],help='list with widths of conv strides')
    
    #load pretrained
    parser.add_argument('-load_pretrained',default=False,help='Boolean for whether or not to load pretrained models')
    parser.add_argument('-pretrained_critic_name',type=str,default='./results1/WassGan_fifths_Gfilters_48_48_48_Cfilters_64_64_Gstrides_12-2_2-4_Cstrides_2_2_Critic_epoch_750.h5',help='path for pretrained critic')
    parser.add_argument('-pretrained_gen_name',type=str,default='./results1/WassGan_fifths_Gfilters_48_48_48_Cfilters_64_64_Gstrides_12-2_2-4_Cstrides_2_2_Gen_epoch_750.h5',help='path for pretrained generator')
    
    return parser
    

def check_args(args):

    assert(len(args.generator_starting_shape)==2),"generator_starting_shape must have length 2"
        
    #kernel sizes should be factors of stride sizes (maybe this gives a warning instead of an error)
    for i in range(len(args.critic_kernels)):
        assert(args.critic_kernels[i]%args.critic_strides[i]==0),"Critic kernel size must be a factor of stride size"
        
    for i in range(len(args.generator_kernel_height)):
        
        assert(args.generator_kernel_height[i]%args.generator_stride_height[i]==0),"Generator kernel height must be a factor of stride height"
        assert(args.generator_kernel_width[i]%args.generator_stride_width[i]==0),"Generator kernel width must be a factor of stride width"
    
    #len(generator_kernel_height) should be 1 less than len(generator_filters)
    assert(len(args.generator_kernel_height)==len(args.generator_filters)-1),"len(generator_kernel_height) should be 1 less than len(generator_filters)"
    
    #len(generator_kernel_width) should be 1 less than len(generator_filters)
    assert(len(args.generator_kernel_width)==len(args.generator_filters)-1),"len(generator_kernel_width) should be 1 less than len(generator_filters)"
    
    #len(generator_stride_height) should be 1 less than len(generator_filters)
    assert(len(args.generator_stride_height)==len(args.generator_filters)-1),"len(generator_stride_height) should be 1 less than len(generator_filters)"
    
    #len(generator_stride_width) should be 1 less than len(generator_filters)
    assert(len(args.generator_stride_width)==len(args.generator_filters)-1),"len(generator_stride_width) should be 1 less than len(generator_filters)"
    
    #len(critic_filters)==len(critic_kernels)==len(critic_strides)
    assert(len(args.critic_filters)==len(args.critic_kernels)),"critic filters, strides and kernels must be the same length"
    assert(len(args.critic_filters)==len(args.critic_strides)),"critic filters, strides and kernels must be the same length"
    
    #final size of generated image should be correct
    height=args.generator_starting_shape[0]
    width=args.generator_starting_shape[1]
    for i in range(len(args.generator_stride_height)):
        height*=args.generator_stride_height[i]
        width*=args.generator_stride_width[i]
        
    assert(height==args.image_height),"Generator must produce images with the same shape as ins"
    assert(width==args.image_width),"Generator must produce images with the same shape as ins"
    
    #mapping must be 'linear' or 'fifths'
    assert((args.mapping=='linear') or (args.mapping=='fifths')),"mapping must be linear or fifths"

#################################################################
if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    check_args(args)
    execute_exp(args)