import numpy as np
from statsmodels.tsa.stattools import acovf

LONGITUD = 44100*2

def recortar(audio):
    if len(audio) >= LONGITUD:
        return audio[:LONGITUD]
    else:
        return np.pad(audio, (0, LONGITUD - len(audio)))

def calcularFFT(audio):
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    audio = recortar(audio)
    x = audio - np.mean(audio)
    autocov = acovf(x,fft=True)
    return np.fft.rfft(autocov)
    ##return autocov

def determinar_espectro(audio_data, tipo = ""):
    if isinstance(audio_data, list):
        espectros = []

        for audio in audio_data:
            espectros.append(np.abs(calcularFFT(audio)))

        espectro = np.mean(espectros, axis = 0)
        espectro /= np.max(espectro)
        if tipo:
           np.save(tipo, espectro)
    
        return espectro
    
    espectro = np.abs(calcularFFT(audio_data))
    
    if np.max(espectro) != 0 :
        return (espectro / np.max(espectro))
    else:
        return espectro
        
