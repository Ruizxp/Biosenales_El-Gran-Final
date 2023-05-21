
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import scipy.signal as signal

def sonido_probando123(lista_archivos,ruta_carpeta):
    sound_list={}
    for archivo in lista_archivos:
        # Verificar que el archivo sea del tipo 
        if archivo.endswith('.wav'):   
            y, sr = librosa.load(os.path.join(ruta_carpeta,archivo))
            sound_list[os.path.join(ruta_carpeta,archivo)]=[y,sr]
    return sound_list



def wnoisest(coeff):
    stdc = np.zeros((len(coeff),1))
    for i in range(0,len(coeff)):
        stdc[i] = (np.median(np.absolute(coeff[i])))/0.6745
    return stdc

def threshold(coeff):
    Num_samples = 0
    for i in range(0,len(coeff)):
        Num_samples = Num_samples + coeff[i].shape[0]
    
    thr = np.sqrt(2*(np.log(Num_samples)))
    return thr

def wthresh(coeff):
    y   = list()
    s = wnoisest(coeff)
    thr = threshold(coeff)
    for i in range(0,len(coeff)):
        y.append(np.multiply(coeff[i],np.abs(coeff[i])>(thr*s[i])))
    return y

def grafiquelo(promedios,xlim,ylim):
    f=np.linspace(0,1000,len(promedios[0]))
    for i in promedios:
        plt.plot(f,i)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend(["Sanos","Crackles","Wheezes"])

def mfreqz(b,a,order,nyq_rate = 1):
    
    """
    Plot the impulse response of the filter in the frequency domain

    Parameters:
        
        b: numerator values of the transfer function (coefficients of the filter)
        a: denominator values of the transfer function (coefficients of the filter)
        
        order: order of the filter 
                
        nyq_rate = nyquist frequency
    """
    
    w,h = signal.freqz(b,a);
    h_dB = 20 * np.log10 (abs(h));
    
    plt.figure();
    plt.subplot(311);
    plt.plot((w/max(w))*nyq_rate,abs(h));
    plt.ylabel('Magnitude');
    plt.xlabel(r'Normalized Frequency (x$\pi$rad/sample)');
    plt.title(r'Frequency response. Order: ' + str(order));
    [xmin, xmax, ymin, ymax] = plt.axis();
    
    #plt.xlim((40,60))
    
    plt.grid(True);
    
    plt.subplot(312);
    plt.plot((w/max(w))*nyq_rate,h_dB);
    plt.ylabel('Magnitude (db)');
    plt.xlabel(r'Normalized Frequency (x$\pi$rad/sample)');
    plt.title(r'Frequency response. Order: ' + str(order));
    plt.grid(True)
    plt.grid(True)
    
    
    plt.subplot(313);
    h_Phase = np.unwrap(np.arctan2(np.imag(h),np.real(h)));
    plt.plot((w/max(w))*nyq_rate,h_Phase);
    plt.ylabel('Phase (radians)');
    plt.xlabel(r'Normalized Frequency (x$\pi$rad/sample)');
    plt.title(r'Phase response. Order: ' + str(order));
    plt.subplots_adjust(hspace=0.5);
    plt.grid(True)
    plt.show()