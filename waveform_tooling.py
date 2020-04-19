import librosa
import numpy as np
import scipy as sp
import datetime
import math, random

def getWavelengthMap(input, lowerLimit = 16, upperLimit = 300):
    if (lowerLimit < 16):
        lowerLimit = 16

    if (lowerLimit > 250):
        lowerLimit = 250

    if (upperLimit < 16):
        upperLimit = 16

    if (upperLimit > 300):
        upperLimit = 300
    
    waveLengthMap = []
    
    for i in range(0, len(input) - 880, 440):
        corrMap = np.correlate(input[i:i + 880], input[i:i + 440]) 
        waveLengthMap.append(np.argmax(corrMap[lowerLimit:upperLimit]) + lowerLimit)
        
    return sp.signal.medfilt(np.array(waveLengthMap, dtype = np.int32), 11)

def lookupWavelength(index, waveLengthMap):
    mapIndex = index // 440

    if (mapIndex >= len(waveLengthMap)):
        mapIndex = len(waveLengthMap) - 1
    
    return waveLengthMap[mapIndex]

def findNextBestCandidate(y, start, candidates, windowSize):
    corrMap = np.correlate(y[start:(start + windowSize * 2)], y[start:(start + windowSize)])
    return candidates[np.argmax(corrMap[candidates])]

def uniform(y, lastWaveformPoint, bestCandidate, oversampling = 2, encoderWidth = 600):
    waveform = np.interp(np.linspace(0, bestCandidate, bestCandidate * oversampling), \
                     np.arange(0, encoderWidth), \
                     y[lastWaveformPoint:lastWaveformPoint + encoderWidth])
    
    if(encoderWidth - bestCandidate * oversampling > 0):
        tailCompression = 20
        tailDist = np.linspace(0, 1, encoderWidth - bestCandidate * oversampling)
        
        tailDist = tailDist / (tailDist * tailCompression + 1)        
        
        tailGrid = (encoderWidth - bestCandidate) * tailDist / oversampling
        
        tail = np.interp(tailGrid, \
                         np.arange(0, encoderWidth - bestCandidate), \
                         y[lastWaveformPoint + bestCandidate:lastWaveformPoint + encoderWidth])
        
        waveform = np.concatenate((waveform, tail))
    
    return waveform[0:encoderWidth]

def split(y, wavelengthMap, encoderWidth = 600):
    zeroCrossings = np.nonzero(librosa.zero_crossings(y))
    
    lastWaveformPoint = 0
    
    result = []
    
    waveformCandidates = []
    
    counter = 0

    for crossingPoint in zeroCrossings[0].tolist():
        windowSize = int(lookupWavelength(crossingPoint, wavelengthMap) * 1.1)
        
        if(crossingPoint + windowSize * 2 >= len(y) or crossingPoint + encoderWidth >= len(y)):
            break
               
        if (crossingPoint - lastWaveformPoint > windowSize / 4 and crossingPoint - lastWaveformPoint < windowSize):
            waveformCandidates.append(crossingPoint - lastWaveformPoint)
            
        if (crossingPoint - lastWaveformPoint > windowSize):
            waveformCandidates = [c for c in waveformCandidates if y[lastWaveformPoint + c] > 0]
            
            if(len(waveformCandidates) > 0):
                bestCandidate = findNextBestCandidate(y, lastWaveformPoint, waveformCandidates, max(windowSize, max(waveformCandidates)))
                
                waveform = uniform(y, lastWaveformPoint, bestCandidate)
                result.append(waveform)

                lastWaveformPoint = lastWaveformPoint + bestCandidate
                
                for i in range(0, len(waveformCandidates)):
                    waveformCandidates[i] = waveformCandidates[i] - bestCandidate
                
                waveformCandidates = [c for c in waveformCandidates if c > windowSize / 4]
                
                counter += 1
            else:
                lastWaveformPoint = crossingPoint
                waveformCandidates = []

    return result

def merge(waveforms, oversampling = 2):
    output = np.empty(0)
    
    tail = None
    for waveform in waveforms:
        waveform = np.interp(np.linspace(0, len(waveform), len(waveform) // oversampling), \
                             np.arange(0, len(waveform)), waveform)
        zeroCrossings = np.nonzero(librosa.zero_crossings(waveform))
        
        if(len(zeroCrossings) == 0):
            output = np.concatenate((output, waveform))
            tail = None
        else:
            waveformEnd = zeroCrossings[0][len(zeroCrossings[0]) - 1]
            
            if(tail is not None and len(tail) > 0):
                cfLength = min([int(len(waveform) * 0.1), len(tail)])
                fadeIn = np.linspace(0, 1, cfLength)
                output = np.concatenate((output, waveform[0:cfLength] * fadeIn + tail[0:cfLength] * (1 - fadeIn)))
                output = np.concatenate((output, waveform[cfLength:waveformEnd]))
            else:
                output = np.concatenate((output, waveform[0:waveformEnd]))
            
            tail = waveform[waveformEnd:len(waveform)]
            
    
    return output