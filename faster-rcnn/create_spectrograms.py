import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
import scipy.io
from numpy.lib import stride_tricks
#import PIL.Image as Image
import os
from scipy import signal
from sklearn import preprocessing
import matplotlib.image as Image

""" short time fourier transform of audio signal """
#def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
def stft(sig, frameSize, overlapFac=0.999, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    
    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(np.floor(frameSize/2.0).astype(np.int)), sig)    
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)).astype(np.int) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))
    
    frames = stride_tricks.as_strided(samples, shape=(cols, frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win
    
    return np.fft.rfft(frames)    
    
""" scale frequency axis logarithmically """    
def logscale_spec(spec, sr=44100, factor=20., alpha=1.0, f0=0.9, fmax=1):
    spec = spec[:, 0:256]
    timebins, freqbins = np.shape(spec)
    scale = np.linspace(0, 1, freqbins) #** factor
    
    # http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=650310&url=http%3A%2F%2Fieeexplore.ieee.org%2Fiel4%2F89%2F14168%2F00650310
    scale = np.array(map(lambda x: x * alpha if x <= f0 else (fmax-alpha*f0)/(fmax-f0)*(x-f0)+alpha*f0, scale))
    scale *= (freqbins-1)/max(scale)

    newspec = np.complex128(np.zeros([timebins, freqbins]))
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = [0.0 for i in range(freqbins)]
    totw = [0.0 for i in range(freqbins)]
    for i in range(0, freqbins):
        if (i < 1 or i + 1 >= freqbins):
            newspec[:, i] += spec[:, i]
            freqs[i] += allfreqs[i]
            totw[i] += 1.0
            continue
        else:
            # scale[15] = 17.2
            w_up = scale[i] - np.floor(scale[i])
            w_down = 1 - w_up
            j = int(np.floor(scale[i]))
           
            newspec[:, j] += w_down * spec[:, i]
            freqs[j] += w_down * allfreqs[i]
            totw[j] += w_down
            
            newspec[:, j + 1] += w_up * spec[:, i]
            freqs[j + 1] += w_up * allfreqs[i]
            totw[j + 1] += w_up
    
    for i in range(len(freqs)):
        if (totw[i] > 1e-6):
            freqs[i] /= totw[i]
    
    return newspec, freqs

""" plot spectrogram"""
#def plotstft(audiopath, binsize=2**10, plotpath=None, colormap="gray", channel=0, name='tmp.png', alpha=1, offset=0):
def plotstft(audiopath, binsize=2**10, plotpath=None, colormap="gray", channel=0, name='tmp.png', alpha=1, offset=0):
    #samplerate, samples = wav.read(audiopath)
    #samples = samples[:, channel]
    samplerate=1
    samples=scipy.io.loadmat('./preliminary/TRAIN/'+audiopath)
    samples=samples['data']
    im=[]
    for i in range(samples.shape[0]):
      widths = np.arange(1, 2**8+1)
      cwtmatr = signal.cwt(samples[i,:], signal.ricker, widths)
#      plt.imshow(cwtmatr, cmap='PRGn',aspect='auto')
#      plt.show()
      ims=cwtmatr
      
#      impng=np.array(ims,dtype='float32')
#      impng = Image.fromarray(impng) 
#      image = image.convert('L')
#      image.save(name[:-4]+'.png')
      Image.imsave(name[:-4]+str(i)+'.png',ims)
      
     # s = stft(samples[i,:], binsize)

#      sshow, freq = logscale_spec(s, factor=1, sr=samplerate, alpha=alpha)
#      sshow = sshow[2:, :]
#      ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel
#      timebins, freqbins = np.shape(ims)
#    
#      ims = np.transpose(ims)
#    # ims = ims[0:256, offset:offset+768] # 0-11khz, ~9s interval
#      ims = ims[0:256, :] # 0-11khz, ~10s interval
    #print "ims.shape", ims.shape
      im.append(ims)
    ims={'Spectrum':ims}
    scipy.io.savemat(name,ims)


file = open('./preliminary/reference.txt', 'r')
datapath='./preliminary/TRAIN/'
result_path='./preliminary/TRAIN/pngaugm/'
label0_path='./preliminary/TRAIN/pngaugm/0/'
label1_path='./preliminary/TRAIN/pngaugm/1/'
if not os.path.isdir(result_path):
    os.mkdir(datapath)
if not os.path.isdir(label0_path):
    os.mkdir(label0_path)
if not os.path.isdir(label1_path):
    os.mkdir(label1_path)
for iter, line in enumerate(file.readlines()[0:]): # first line of traininData.csv is header (only for trainingData.csv)
    filename,label = line.split( )
    samples=scipy.io.loadmat(datapath+filename+'.mat')
    samples=samples['data']
    im=[]
    for i in range(samples.shape[0]):
      widths = np.arange(1, 2**8+1)
      cwtmatr = signal.cwt(samples[i,:], signal.ricker, widths)
      cwtmatr=np.concatenate((np.flip(cwtmatr,axis=0),cwtmatr),axis=0)
#      plt.imshow(cwtmatr, cmap='PRGn',aspect='auto')
#      plt.show()
      ims=cwtmatr
      
#      impng=np.array(ims,dtype='float32')
#      impng = Image.fromarray(impng) 
#      image = image.convert('L')
#      image.save(name[:-4]+'.png')
      if label == '0':
          Image.imsave(label0_path+filename+str(i).zfill(2)+'.png',ims)
      else:
          Image.imsave(label1_path+filename+str(i).zfill(2)+'.png',ims)
          
      im.append(ims)
    
    imc=np.concatenate(im,axis=1)
    imc=np.reshape(imc,(imc.shape[0],5000,12))
    imm={'Spectrum':imc}
    scipy.io.savemat(result_path+filename+'.mat',imm)
   # filename = filepath[:-4]
   
   # wavfile = 'tmp.wav'
   # os.system('mpg123 -w ' + wavfile + ' '+ os.path.join(os.path.abspath('..'),'trainingdata',filepath))
    """
    for augmentIdx in range(0, 20):
        alpha = np.random.uniform(0.9, 1.1)
        offset = np.random.randint(90)
        plotstft(wavfile, channel=0, name='/home/brainstorm/data/language/train/pngaugm/'+filename+'.'+str(augmentIdx)+'.png',
                 alpha=alpha, offset=offset)
    """
    # we create only one spectrogram for each speach sample
    # we don't do vocal tract length perturbation (alpha=1.0)
    # also we don't crop 9s part from the speech
  #  plotstft(wavfile, channel=0, name=os.path.join(os.path.abspath('..'),'trainingdata','pngaugm/')+filename+'.png', alpha=1.0)
#    plotstft(filename, channel=0, name=os.path.join('./preliminary/TRAIN/pngaugm/')+filename+'.mat', alpha=1.0)
  #  os.remove(wavfile)
    print(("processed %d files" % (iter + 1)))