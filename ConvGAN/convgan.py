# Here we're going to combine our argparse script with a convgan class to do a few things
# this will be able to either take premade training data or create training data given a glob-ready string, something like '../Final_Fantasy/*.mid'
# the argparse will contain all of the hyperparameters you could ever want
# it will train a GAN and output images and/or MIDI files periodically during training


import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import pickle
from tensorflow.keras.layers import InputLayer, Dense, LSTM, Dropout, Activation, Conv2D, Flatten, Reshape, Conv2DTranspose
from tensorflow.keras.models import Sequential
import argparse

FONTSIZE=18
plt.rcParams['figure.figsize']=(10,10)
plt.rcParams['font.size']=FONTSIZE



def build_discriminator(args):
    '''
    args for build_discriminator
    -image_height
    -image_width
    -num_channels
    -filters
    -disc_h_strides
    -disc_w_strides
    -activation
    -disc_h_kernels
    -disc_w_kernels
    -l2
    -dropout
    -verbose
    -lrate
    '''
    model=Sequential()
    input_shape=(args.image_height,args.image_width,args.num_channels)

    for i in range(len(args.filters)):

        if i==0:
            model.add(Conv2D(args.filters[i],
                             strides=(args.disc_h_strides[i],args.disc_w_strides[i]),
                             input_shape=input_shape,
                             activation=args.activation,
                             kernel_size=(args.disc_h_kernels[i],args.disc_w_kernels[i]),
                             padding='same',
                             use_bias=True,
                             kernel_initializer='truncated_normal',
                             bias_initializer='zeros',
                             name='DC'+str(i),
                             kernel_regularizer=tf.keras.regularizers.l2(args.l2)
                             ))
        else:
            model.add(Conv2D(args.filters[i],
                             strides=(args.disc_h_strides[i],args.disc_w_strides[i]),
                             activation=args.activation,
                             kernel_size=(args.disc_h_kernels[i],args.disc_w_kernels[i]),
                             padding='same',
                             use_bias=True,
                             kernel_initializer='truncated_normal',
                             bias_initializer='zeros',
                             name='DC'+str(i),
                             kernel_regularizer=tf.keras.regularizers.l2(args.l2)
                             ))
        model.add(Dropout(args.dropout))

    model.add(Flatten())
    model.add(Dense(1,activation='sigmoid'))
    if args.verbose>0:
        print(model.summary())
    opt=tf.keras.optimizers.Adam(lr=args.lrate,beta_1=0.9,beta_2=0.999,epsilon=None,decay=0.0,amsgrad=False)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model

def build_generator(args):
    '''
    args for build_generator
    -image_height
    -image_width
    -generator_feature_space
    -latent_dim
    -activation
    -num_channels
    -verbose
    -gen_starting_fraction starting size is final image size divided by this number
    -gen_h_kernels list of kernel sizes for generator
    -gen_w_kernels
    -gen_h_strides list of strides for generator
    -gen_w_strides
    '''
    #latent space is a vector of length 100
    model=Sequential()
    n_nodes=int( int(args.image_height/args.gen_starting_fraction)*int(args.image_width/args.gen_starting_fraction)*args.generator_feature_space  )
    model.add(Dense(n_nodes,
                     input_dim=args.latent_dim,
                     activation=args.activation,
                     ))
    
    model.add(Reshape((int(args.image_height/args.gen_starting_fraction),int(args.image_width/args.gen_starting_fraction),args.generator_feature_space)))
    
    for i in range(len(args.gen_h_strides)):
        
        model.add(Conv2DTranspose(args.generator_feature_space,
                                  kernel_size=(args.gen_h_kernels[i],args.gen_w_kernels[i]),
                                  strides=(args.gen_h_strides[i],args.gen_w_strides[i]),
                                  padding='same',
                                  activation=args.activation
                                  ))
    
    model.add(Conv2D(args.num_channels, (7,7), activation='sigmoid', padding='same'))
    
    if args.verbose>0:
        print(model.summary())

    return model
    
    
def generate_latent_space(n_samples,latent_dim):
    
    xinput=np.random.randn(n_samples*latent_dim)
    xinput=xinput.reshape((n_samples,latent_dim))
    
    return xinput

def generate_fake_samples(G,n_samples,latent_dim):
    
    xinput=generate_latent_space(n_samples,latent_dim)
    
    X=G.predict(xinput)
    
    y=np.zeros((n_samples,1))
    
    return X,y

def build_gan(D,G,args):
    '''
    args for build_gan
    -verbose
    -lrate
    '''
    D.trainable=False
    
    model=Sequential()
    model.add(G)
    model.add(D)
    if args.verbose>0:
        print(model.summary())
    opt=tf.keras.optimizers.Adam(lr=args.lrate,beta_1=0.9,beta_2=0.999,epsilon=None,decay=0.0,amsgrad=False)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    
    return(model)

def generate_real_samples(ins, n_samples):
    # choose random instances
    ix = np.random.randint(0, ins.shape[0], n_samples)
    # retrieve selected images
    X = ins[ix]
    # generate 'real' class labels (1)
    y = np.ones((n_samples, 1))
    return X, y

# evaluate the discriminator, plot generated images
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=100):
    # prepare real samples
    X_real, y_real = generate_real_samples(dataset, n_samples)
    # evaluate discriminator on real examples
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    # prepare fake examples
    x_fake, y_fake = generate_fake_samples(g_model, n_samples, latent_dim)
    # evaluate discriminator on fake examples
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    
    return acc_real,acc_fake
    
def save_plot(xinput,G,epoch, n=5):
    
    #xinput is input latent space
    #G is generator
    #this way we can evolve our plots from the same latent space during training
    x_fake=G.predict(xinput)
    
    # plot images
    fig,axs=plt.subplots(n,n)
    count=-1
    for i in range(n):
        for j in range(n):
            count+=1
            # turn off axis
            axs[i,j].axis('off')
            # plot raw pixel data
            axs[i,j].imshow(x_fake[count, :, :, 0])
    # save plot to file
    plt.subplots_adjust(wspace=0.01,hspace=0.01)
    filename = './plots/generated_plot_e%03d.png' % (epoch)
    plt.savefig(filename,bbox_inches='tight')
    plt.close()
    
def build_training_data(args):
    '''
    args for build_training_data
    -premade_training (1 if just loading in premade training data, 0 if generating training data)
    -fname_train
    '''
    
    if args.premade_training==1:
        fp = open(args.fname_train, "rb")
        ins = pickle.load(fp)
        fp.close()
        
    return ins

def generate_fname(args):
    
    gen_strides_str='Gstrides_'
    for i in args.gen_h_strides:
        gen_strides_str+=str(i)
        gen_strides_str+='_'
        
    disc_strides_str='Dstrides'
    for i in args.disc_h_strides:
        disc_strides_str+='_'
        disc_strides_str+=str(i)
        
    l2_str='l2'+str(args.l2)+'_'
    dropout_str='drop'+str(args.dropout)
    
    fbase='ConvGan_'+args.mapping+'_'+gen_strides_str+disc_strides_str
    
    return fbase
    
    
    
def execute_exp(args):
    '''
    args for execute_exp
    -n_batch
    -n_epochs
    -verbose
    -checkpoints (list of epoch numbers where you want to generate plots and output the model)
    -gen_boost_factor multiplicative for extra generator training
    '''
    
    D=build_discriminator(args)
    G=build_generator(args)
    C=build_gan(D,G,args)
    fbase=generate_fname(args)
    
    ins=build_training_data(args)
    
    d_losses=[]
    g_losses=[]
    
    accs_real=[]
    accs_fake=[]

    batch_per_epoch=int(ins.shape[0]/args.n_batch)
    half_batch=int(args.n_batch/2)
    
    xinput=generate_latent_space(25,args.latent_dim) #generate a single latent space that will be used for making plots during training
    
    for i in range(args.n_epochs):
        for j in range(batch_per_epoch):
            X_real, y_real = generate_real_samples(ins,half_batch)
            X_fake, y_fake = generate_fake_samples(G,half_batch,args.latent_dim)
            X, y = np.vstack((X_real, X_fake)), np.vstack((y_real, y_fake))
            d_loss, _ = D.train_on_batch(X, y)
            d_losses.append(d_loss)
            
            for k in range(args.gen_boost_factor):
            
                X_gan = generate_latent_space(args.n_batch,args.latent_dim)
                y_gan = np.ones((args.n_batch, 1))
                g_loss = C.train_on_batch(X_gan, y_gan)
                g_losses.append(g_loss)
            
            if args.verbose>0:
                print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, batch_per_epoch, d_loss, g_loss))
                
        if (i+1)%args.plot_every==0:
            acc_real,acc_fake=summarize_performance(i+1, G, D, ins, args.latent_dim)
            accs_real.append(acc_real*100)
            accs_fake.append(acc_fake*100)
            save_plot(xinput,G,i+1)
            
        if i+1 in args.checkpoints:
            # save the generator model tile file
            filename = './results/%s_epoch_%03d.h5' % (fbase,i+1)
            G.save(filename)
            
            #write out results at each checkpoint too
            results={}
            results['args']=args
            results['d_loss']=d_losses
            results['g_loss']=g_losses
            results['acc_real']=accs_real
            results['acc_fake']=accs_fake
            results['epoch']=i+1
    
            #Save results
            results['fname_base']=fbase
            fp=open("./results/%s_results_e%03d.pkl" % (fbase,i+1),'wb')
            pickle.dump(results,fp)
            fp.close()
    
    
def create_parser():
    
    parser=argparse.ArgumentParser(description='Image Learner')
    
    parser.add_argument('-image_height',type=int,default=96,help='Number of Rows in the image')
    parser.add_argument('-image_width',type=int,default=96,help='Number of Columns in the image')
    parser.add_argument('-num_channels',type=int,default=4,help='Number of Image Channels')
    parser.add_argument('-filters',nargs='+',type=int,default=[64,64],help='List of Conv filters for Discriminator')
    parser.add_argument('-disc_h_strides',nargs='+',type=int,default=[12,2],help="list of strides for discriminator's conv layers")
    parser.add_argument('-disc_w_strides',nargs='+',type=int,default=[2,12],help="list of strides for discriminator's conv layers")
    parser.add_argument('-activation',type=str,default='elu',help='Activation of Nodes and Filters')
    parser.add_argument('-disc_h_kernels',nargs='+',type=int,default=[12,2],help="list of kernel sizes for discriminator's conv layers")
    parser.add_argument('-disc_w_kernels',nargs='+',type=int,default=[2,12],help="list of kernel sizes for discriminator's conv layers")
    parser.add_argument('-l2',type=float,default=0.0001,help='l2 regularization value')
    parser.add_argument('-dropout',type=float,default=0.4,help='dropout probability')
    parser.add_argument('-verbose',type=int,default=1,help='verbosity of experiment')
    parser.add_argument('-lrate',type=float,default=0.001,help='learning rate')
    parser.add_argument('-generator_feature_space',type=int,default=128,help='Number of Conv filters for generator layers')
    parser.add_argument('-latent_dim',type=int,default=100,help='length of latent space for generator')
    parser.add_argument('-gen_starting_fraction',type=int,default=24,help='low res starting image for generator will be this fraction of output image')
    parser.add_argument('-gen_h_kernels',nargs='+',type=int,default=[12,2],help='list of height kernel sizes for generator convolutions')
    parser.add_argument('-gen_w_kernels',nargs='+',type=int,default=[2,12],help='list of width kernel sizes for generator convolutions')
    parser.add_argument('-gen_h_strides',nargs='+',type=int,default=[12,2],help='list of height strides for generator convolutions')
    parser.add_argument('-gen_w_strides',nargs='+',type=int,default=[2,12],help='list of width strides for generator convolutions')
    parser.add_argument('-premade_training',type=int,default=1,help='1 if premade data, 0 if making data here (not yet added)')
    parser.add_argument('-fname_train',type=str,default='ins_convgan_final_fantasy.pkl',help='filename for ins')
    parser.add_argument('-n_batch',type=int,default=64,help='number of samples per batch')
    parser.add_argument('-n_epochs',type=int,default=500,help='number of training epochs')
    parser.add_argument('-checkpoints',type=int,nargs='+',default=[1,10,100,250,500],help='list of epochs for saving model checkpoints')
    parser.add_argument('-mapping',type=str,default='linear',help='mapping of pitch to the y-axis (linear or fifths)')
    parser.add_argument('-plot_every',type=int,default=1,help='generate plots after every X epochs')
    parser.add_argument('-gen_boost_factor',type=int,default=1,help='multiplicative for extra generator training (2 means train generator twice as long as discriminator)')
    
    return parser

def check_args(args):
    
    #images must be squares
    assert(args.image_height==args.image_width),"Images must be squares"
    
    #make sure stride lists are same length as filters list
    assert(len(args.filters)==len(args.disc_h_strides)),"Discriminator stride vectors must be same length as filters vector"
    assert(len(args.filters)==len(args.disc_h_kernels)),"Discriminator kernel vectors must be same length as filters vector"
    
    #low res gen starting space must have integer number of nodes
    assert( (args.image_height/args.gen_starting_fraction)%1==0),'generator low res starting size must be an integer'
    assert( (args.image_width/args.gen_starting_fraction)%1==0),'generator low res starting size must be an integer'
    
    #make sure the output for the generator is the right size
    initial_image_size=args.image_height/args.gen_starting_fraction
    final_image_height=initial_image_size
    final_image_width=initial_image_size
    for i in range(len(args.gen_h_strides)):
        final_image_height*=args.gen_h_strides[i]
        final_image_width*=args.gen_w_strides[i]
    assert(final_image_height==args.image_height),"Generator must output images that are the same size as real images"
    assert(final_image_width==args.image_width),"Generator must output images that are the same size as real images"
    assert(len(args.gen_h_strides)==len(args.gen_h_kernels)),"Generator stride vector must be same length as kernel vector"
    assert(len(args.gen_w_strides)==len(args.gen_w_kernels)),"Generator stride vector must be same length as kernel vector"
    assert(len(args.gen_h_strides)==len(args.gen_w_strides)),"Generator height and width stride vectors must be same length"
    assert(len(args.gen_h_kernels)==len(args.gen_w_kernels)),"Generator height and width kernel vectors must be same length"
    
    #make sure discriminator strides and kernels are same length
    assert(len(args.disc_h_strides)==len(args.disc_w_strides)),"Discriminator height and width stride vectors must be same length"
    assert(len(args.disc_h_kernels)==len(args.disc_w_kernels)),"Discriminator height and width kernel vectors must be same length"
    assert(len(args.disc_h_strides)==len(args.disc_h_kernels)),"Discriminator stride vectors must be same length as kernel vectors"
    
#################################################################
if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    check_args(args)
    execute_exp(args)
            
    
    
    

    