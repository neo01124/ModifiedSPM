#!/usr/bin/python

#
# Varun Jewalikar
#
# Feature Extraction Tool

#from getLyrics import getLyricsMain
import math
import numpy as np
import essentia.streaming as stream
import essentia.standard as standard
import essentia
import os
import ast
import datetime
import json
import copy
import shutil
from subprocess import call
import subprocess
import time as pytime
from time import gmtime, strftime
import glob
from numpy import log10, square, sqrt
from scipy.stats import norm
from time import time,clock

train_dir = './EssentiaTrainFeatures'
test_dir = './EssentiaTestFeatures'
numBoundaries = 252
UnvoicedPlosives = ['p','t','k']
VoicedPlosives = ['b','d','g','B','D','G']
UnvoicedAffricates = ['tS']
UnvoicedFricatives = ['f','T','s','x']
Nasals = ['m','n','J']
Liquids = ['l','r','rr','L']
Semivowels = ['j','j\\','w','I','U']	
Vowels = ['a','e','o','u','i'];
Silence = ['Sil'];
transitionTypes = ['UnvoicedPlosives','VoicedPlosives','UnvoicedAffricates','UnvoicedFricatives','Nasals','Liquids','Semivowels','Vowels','Silence']

def segFileRead(segFile):
	print segFile
	try:
		numbers = np.genfromtxt(segFile,skiprows=3)
		handle = open(segFile,'r')
		lines = handle.readlines()[3:]
	except:
		numbers = np.genfromtxt(segFile,skiprows=5)
		handle = open(segFile,'r')
		lines = handle.readlines()[4:]
	phoneList = []
	for line in lines:
		phone = line.split('\t')
		phoneList.append(phone[0])
	boundaries = numbers[:,2]
	boundaries = boundaries[1:-1]
	phoneList = phoneList[1:-1]
	return (phoneList,boundaries)

def featureExtractDry(audioFile,  nameList):
	#nameListout = copy.deepcopy(nameList)
	#print audioFile
	boundaryFile = audioFile.replace('.wav','.seg')
	[phones,times] = segFileRead(boundaryFile)
	#print boundaryFile
	#print phones
	#print times
	timeStarts = times - (125.0/1000)
	timeEnds = times + (125.0/1000)
	timeStarts = essentia.array(timeStarts)
	timeEnds = essentia.array(timeEnds)
	
	#print len(timeStarts)
	#print len(phones)
	for idx,start_time in enumerate(timeStarts):
		end_time = timeEnds[idx]
		
		#Simple check to ensure that boundaries for feature extraction are within the length of the file
		#print phones
		#print idx
		first = phones[idx]
		if (idx+1) < len(phones):
			second = phones[idx+1]
		else:
			break
		#print first
		#print second
		#UnvoicedPlosives VoicedPlosives UnvoicedAffricates UnvoicedFricatives Nasals Liquids Semivowels Vowels Silence
		if first in UnvoicedPlosives:
			featureFileName = 'UnvoicedPlosives'
		elif first in VoicedPlosives:
			featureFileName = 'VoicedPlosives'
		elif first in UnvoicedAffricates:
			featureFileName = 'UnvoicedAffricates'
		elif first in UnvoicedFricatives:
			featureFileName = 'UnvoicedFricatives'
		elif first in Nasals:
			featureFileName = 'Nasals'
		elif first in Liquids:
			featureFileName = 'Liquids'
		elif first in Semivowels:
			featureFileName = 'Semivowels'
		elif first in Vowels:
			featureFileName = 'Vowels'
		elif first in Silence:
			featureFileName = 'Silence'

		if second in UnvoicedPlosives:
			featureFileName += '_to_UnvoicedPlosives'
		elif second in VoicedPlosives:
			featureFileName += '_to_VoicedPlosives'
		elif second in UnvoicedAffricates:
			featureFileName += '_to_UnvoicedAffricates'
		elif second in UnvoicedFricatives:
			featureFileName += '_to_UnvoicedFricatives'
		elif second in Nasals:
			featureFileName += '_to_Nasals'
		elif second in Liquids:
			featureFileName += '_to_Liquids'
		elif second in Semivowels:
			featureFileName += '_to_Semivowels'
		elif second in Vowels:
			featureFileName += '_to_Vowels'
		elif second in Silence:
			featureFileName += '_to_Silence'
		#print featureFileName
		nameList[featureFileName] +=1
	return nameList 
	
def scaleData(indata, maxval, minval):
	try:
		(x,y) = indata.shape
	except:
		y = 1
	if y == 1:
		outdata = indata - np.min(indata)
		outdata = (outdata/(np.max(outdata) - np.min(outdata)))*(maxval-minval)
		outdata = outdata + minval
	else:
		for indx in np.arange(y):
			outdata = indata
			outdata[:,indx] = outdata[:,indx] - np.min(indata[:,indx])
			outdata[:,indx] = (outdata[:,indx]/(np.max(outdata[:,indx]) - np.min(outdata[:,indx])))*(maxval-minval)
			outdata[:,indx] = outdata[:,indx] + minval
			#outdata[:,indx] = outs
	return outdata

	
def featureExtractMain(audioFile, phones, times, trueTimes, nameList, nameListcln, frameLength, hop):
	
	#metadata calculation
	calcdata = standard.MetadataReader(filename = audioFile)
	metadata = calcdata.compute()
	
	#algorithm parameter declaration
	samplingRate = metadata[-2]
	trueTimeStarts = trueTimes - (125.0/1000)
	trueTimeEnds = trueTimes + (125.0/1000)
	trueTimeStarts = essentia.array(trueTimeStarts)
	trueTimeEnds = essentia.array(trueTimeEnds)

	hmmTimeStarts = times - (125.0/1000)
	hmmTimeEnds = times + (125.0/1000)
	hmmTimeStarts = essentia.array(hmmTimeStarts)
	hmmTimeEnds = essentia.array(hmmTimeEnds)

	frameLength = frameLength*samplingRate
	hop = hop*samplingRate
	# Setup parameters 
	#print hop
	#print frameLength
	if frameLength%2 ==1:
		frameLength +=1
	#print frameLength
	
	
	#algorithm initialisation
	loader = standard.MonoLoader(filename = audioFile)
	audio = loader.compute()
	length = int(len(audio)/samplingRate)
	harmonicPeaks = standard.HarmonicPeaks()
	pitch = standard.PitchYinFFT(frameSize = int(frameLength), sampleRate = samplingRate)
	specContrast = standard.SpectralContrast(frameSize= int(frameLength), sampleRate=samplingRate)
	spectrum = standard.Spectrum(size = int(frameLength)) #size is frameSize
	#mfcc = standard.MFCC(lowFrequencyBound=40, sampleRate=samplingRate) # MFCC
	hpcp = standard.HPCP(sampleRate = samplingRate, size = 36)# HPCP
	hpcp2 = standard.HPCP(sampleRate = samplingRate, size = 120);# HPCP for hiresfeatures
	lowLevelSpectralExtractor = standard.LowLevelSpectralExtractor(frameSize= int(frameLength), hopSize= int(hop), sampleRate=samplingRate)
	spectralPeaks = standard.SpectralPeaks(sampleRate=samplingRate)
	barkBands = standard.BarkBands(sampleRate=samplingRate) 
	hfc = standard.HFC(sampleRate=samplingRate)
	mfcc = standard.MFCC(sampleRate=samplingRate)
	pitchSalience = standard.PitchSalience(sampleRate=samplingRate)
	silenceRate = standard.SilenceRate(thresholds = [20,30,60]) #20db, 30db, 60db
	spectralComplexity = standard.SpectralComplexity(sampleRate=samplingRate)
	zcr = standard.ZeroCrossingRate()
	inharmonicity = standard.Inharmonicity()
	tristimulus = standard.Tristimulus()
	oddToEven = standard.OddToEvenHarmonicEnergyRatio()
	afterMaxToBefore = standard.AfterMaxToBeforeMaxEnergyRatio()
	dissonance = standard.Dissonance()
	erbBands = standard.ERBBands(sampleRate = samplingRate)
	envelope = standard.Envelope()
	#frequencyBands = standard.FrequencyBands(sampleRate = samplingRate) #inherited by bark bands
	gfcc = standard.GFCC(sampleRate = samplingRate)
	highResFeatures = standard.HighResolutionFeatures()
	lpc = standard.LPC(sampleRate = samplingRate, order = 11) # coefficients generated 1 more than the order given and Lin paper specifies 12 coeff so order = 11
	larm = standard.Larm(sampleRate = samplingRate)
	leq = standard.Leq()
	levelExtractor = standard.LevelExtractor(frameSize = int(frameLength), hopSize = int(hop))
	logAttackTime = standard.LogAttackTime(sampleRate = samplingRate)
	loudness = standard.Loudness()
	loudnessVickers = standard.LoudnessVickers(sampleRate = samplingRate)
	maxMagFreq = standard.MaxMagFreq()
	maxToTotal = standard.MaxToTotal()
	minToTotal = standard.MinToTotal()
	noveltyCurve = standard.NoveltyCurve()
	onsetDetection = standard.OnsetDetection()
	onsetRate = standard.OnsetRate()
	onsets = standard.Onsets()
	predominantMelody = standard.PredominantMelody(frameSize = int(frameLength), hopSize = int(hop), sampleRate = samplingRate, minDuration = 2)
	spectralPeaks = standard.SpectralPeaks(sampleRate = samplingRate)
	strongDecay = standard.StrongDecay()
	tCToTotal = standard.TCToTotal()
	
	
	#Statistics of features
	centralMoments = standard.CentralMoments()
	distributionShape = standard.DistributionShape()
	crest = standard.Crest()	
	decrease = standard.Decrease()
	energy = standard.Energy()
	energyBandLow = standard.EnergyBand(sampleRate = samplingRate, startCutoffFrequency = 20,stopCutoffFrequency = 150)
	energyBandMiddleLow = standard.EnergyBand(sampleRate = samplingRate, startCutoffFrequency = 150,stopCutoffFrequency = 800)
	energyBandMiddleHigh = standard.EnergyBand(sampleRate = samplingRate, startCutoffFrequency = 800,stopCutoffFrequency = 4000)
	energyBandHigh = standard.EnergyBand(sampleRate = samplingRate, startCutoffFrequency = 4000,stopCutoffFrequency = 20000)
	flatness = standard.FlatnessDB()
	flux = standard.Flux()
	rms = standard.RMS()
	rollOff = standard.RollOff(sampleRate = samplingRate)
	strongPeak = standard.StrongPeak()
	centroid = standard.Centroid(range = samplingRate/2)
	derivative = standard.Derivative()
	singleGaussian = standard.SingleGaussian()
	
	#global features
	dynamicComplexity = standard.DynamicComplexity(frameSize = int(frameLength), sampleRate = samplingRate)
	effectiveDuration = standard.EffectiveDuration()
	rhythmDescriptors = standard.RhythmDescriptors()
	tonalExtractor = standard.TonalExtractor(frameSize = int(frameLength), hopSize = int(hop), tuningFrequency = 440.0)
	tuningFrequencyExtractor = standard.TuningFrequencyExtractor(frameSize = int(frameLength), hopSize = int(hop))
	#print fname + ' length: ' + str(length)
	# Specify the length of the segment
	# Start computing segment by segment
	featureFileName = ''
	#a,b = enumerate(trueTimeStarts)
	#print trueTimeStarts
	#print len(phones)
	#print len(trueTimeStarts)
	for idx in np.arange(len(trueTimeStarts)):
		#print phones
		first = phones[idx]
		if (idx+1) < len(phones):
			second = phones[idx+1]
		else:
			break
		#print first
		#print second
		#UnvoicedPlosives VoicedPlosives UnvoicedAffricates UnvoicedFricatives Nasals Liquids Semivowels Vowels Silence
		if first in UnvoicedPlosives:
			featureFileName = 'UnvoicedPlosives'
		elif first in VoicedPlosives:
			featureFileName = 'VoicedPlosives'
		elif first in UnvoicedAffricates:
			featureFileName = 'UnvoicedAffricates'
		elif first in UnvoicedFricatives:
			featureFileName = 'UnvoicedFricatives'
		elif first in Nasals:
			featureFileName = 'Nasals'
		elif first in Liquids:
			featureFileName = 'Liquids'
		elif first in Semivowels:
			featureFileName = 'Semivowels'
		elif first in Vowels:
			featureFileName = 'Vowels'
		elif first in Silence:
			featureFileName = 'Silence'

		if second in UnvoicedPlosives:
			featureFileName += '_to_UnvoicedPlosives'
		elif second in VoicedPlosives:
			featureFileName += '_to_VoicedPlosives'
		elif second in UnvoicedAffricates:
			featureFileName += '_to_UnvoicedAffricates'
		elif second in UnvoicedFricatives:
			featureFileName += '_to_UnvoicedFricatives'
		elif second in Nasals:
			featureFileName += '_to_Nasals'
		elif second in Liquids:
			featureFileName += '_to_Liquids'
		elif second in Semivowels:
			featureFileName += '_to_Semivowels'
		elif second in Vowels:
			featureFileName += '_to_Vowels'
		elif second in Silence:
			featureFileName += '_to_Silence'
		#print featureFileName
		#print idx 
		#print len(phones)
		print str(idx) + ' of ' + str(len(phones))
		nameListcln[featureFileName] += 1
		#print (2.0/3)*nameList[featureFileName]
		#print nameListcln[featureFileName]
		if nameListcln[featureFileName] <= ((2.0/3)*nameList[featureFileName]):
				start_time = trueTimeStarts[idx]
				end_time = trueTimeEnds[idx]
				train_flag = 1
		else:
				tru_time = trueTimes
				start_time = hmmTimeStarts[idx]
				end_time = hmmTimeEnds[idx]
				train_flag = 0
		#print train_flag
		#Simple check to ensure that boundaries for feature extraction are within the length of the file
		
		#if idx == 1:
		#	break
		#if end_time > length:
		#	end_time = length
		#if start_time < 0 :
		#	start_time = 0

		segAudio = audio[start_time*samplingRate:end_time*samplingRate]
		pool = essentia.Pool()
		
		pitchVector,confidence = predominantMelody(segAudio)
		DpitchVector = derivative(pitchVector)
		Dconfidence = derivative(confidence)
		for value in pitchVector:
			pool.add('pitch',value) #1 value per frame
		for value in confidence:
			pool.add('pitch_confidence',value) #1 value per frame
		for value in DpitchVector:
			pool.add('Dpitch',value) #1 value per frame
		for value in Dconfidence:
			pool.add('Dpitch_confidence',value) #1 value per frame
		try:
			afttobefore = afterMaxToBefore(pitchVector)
		except:
			afttobefore = np.nan
		for k in np.arange(0,numBoundaries):	
			pool.add('after_to_before_max', afttobefore) #1 value per segment
		
		#Time Consuming
		#dyncomplx,louds = dynamicComplexity(segAudio)
		#pool.add('dynamic_complexity', np.repeat(dyncomplx,numBoundaries)) #1 value per segment
		#pool.add('loudness', np.repeat(louds,numBoundaries)) #1 value per segment
		
		#Little time consuming
		effdur = effectiveDuration(segAudio)
		for k in np.arange(0,numBoundaries):
			pool.add('effective_duration', effdur) #1 value per segment
		
		#Dimensions not understood
		#envl = envelope(segAudio)
		#pool.add('effective_duration', envl) #1 value per segment
		
		#Larm loudness used. Loudness and LoudnessVickers skipped
		loudnss = larm(segAudio)
		for k in np.arange(0,numBoundaries):
			pool.add('loudnss', loudnss) #1 value per segment
		
		#Log attack time
		envl = envelope(segAudio)
		lgatt = logAttackTime(envl)
		for k in np.arange(0,numBoundaries):
			pool.add('log_attack', lgatt) #1 value per segment
		
		#Max to total of envelope
		mxtot = maxToTotal(envl)
		for k in np.arange(0,numBoundaries):
			pool.add('max_to_total', mxtot) #1 value per segment
		
		# Novelty curve skipped
		
		#Temporal centroid to total length ratio
		tctot = tCToTotal(envl)
		for k in np.arange(0,numBoundaries):
			pool.add('tc_to_totalratio', tctot) #1 value per segment
		
		#Tonal extractor skipped
		
		# Windowing
		window = standard.Windowing(size = int(frameLength))
		for frame in standard.FrameGenerator(segAudio, frameSize = int(frameLength), hopSize = int(hop)):
			# spectral contrast
			s = spectrum(window(frame))
			contrast, valley = specContrast(s) 
			pool.add('spectral_contrast', contrast) #6 values
		 	pool.add('spectral_valley', valley) #6 values
			
			freqs, mags = spectralPeaks(s)
			hpcps = hpcp(freqs, mags)
			pool.add('hpcp', hpcps) #36 values
			
			specComplexity = spectralComplexity(s)
			pool.add('spectral_complexity', specComplexity) #1 value
			
			#bbands = barkBands(s)
			#pool.add('bark_bands', bbands) #27 values
			
			diss = dissonance(freqs,mags)
			pool.add('dissonance', diss) #1 value
			
			#erb = erbBands(s)
			#pool.add('erb_bands', erb) #40 values
			
			#Energy bands // EnergyBandRatio might be more meaningful
			enrgl  = energyBandLow(s)
			pool.add('energy_low', enrgl) #1 value
			
			enrgml  = energyBandMiddleLow(s)
			pool.add('energy_middle_low', enrgml) #1 value
			
			enrgmh  = energyBandMiddleHigh(s)
			pool.add('energy_middle_high', enrgmh) #1 value

			enrgh  = energyBandHigh(s)
			pool.add('energy_high', enrgh) #1 value
			
			flatns = flatness(s)
			pool.add('spectral_flatness', flatns) #1 value
			
			flx = flux(s)
			pool.add('spectral_flux', flx) #1 value
			
			#giving NANS
			#rm = rms(s)
			#try:
			#	pool.add('spectral_rms', rms) #1 value
			#except:
			#	pool.add('spectral_rms', np.nan)
			
			rolf = rollOff(s)		
			pool.add('spectral_rolloff', rolf) #1 value
			
			strngpk = strongPeak(s)
			pool.add('spectral_strong_peak', strngpk) #1 value
			
			#gfcband, gfc = gfcc(s)
			#pool.add('gfcc_bands', gfcband) #40 value
			#pool.add('gfcc_coeff', gfc) #13 value

			hf = hfc(s)
			pool.add('hfc', hf) #1 value

			#High res features
			hpcps2 = hpcp2(freqs, mags)
			hir,hire,hires = highResFeatures(hpcps2)
			pool.add('equal_tempered_deviation', hir) #1 value
			pool.add('non_tempered_energyratio', hire) #1 value
			pool.add('non_tempered_peaksenergyratio', hires) #1 value
			
			# harmonic spectral features
			if len(freqs) > 0:
				p, conf = pitch(s)
				if freqs[0] == 0:
					freqs = freqs[1:]
					mags = mags[1:]
				freqs, mags = harmonicPeaks(freqs, mags, p)
				_sum = 0
				if len(freqs) == 1:
					specEnvelope_i = [freqs[0]] #for hsd
					_sum = freqs[0]*mags[0]
				elif len(freqs) == 2:
					specEnvelope_i = [(freqs[0]+freqs[1])/2.0] #for hsd
					_sum = freqs[0]*mags[0]+freqs[1]*mags[1]
				elif len(freqs) > 2:
					specEnvelope_i = [(freqs[0]+freqs[1])/2.0] #for hsd
					_sum = freqs[0]*mags[0]
					for i in xrange(1, len(freqs)-1):
						_sum += freqs[i]*mags[i] #for hsc_i
						specEnvelope_i.append((freqs[i-1]+freqs[i]+freqs[i+1])/3.0)
					specEnvelope_i.append((freqs[i]+freqs[i+1])/2.0)
					_sum += freqs[i+1]*mags[i+1]
				hsc_i = _sum/sum(mags)
				hsd_i = sum(abs(log10(mags)-log10(specEnvelope_i)))/sum(log10(mags))
				hss_i = sqrt(sum(square(freqs-hsc_i)*square(mags))/sum(square(mags)))/hsc_i
			else:
				hsc_i = 0
				hsd_i = 0
				hss_i = 0
				
			pool.add('harmonic_spectral_centroid', hsc_i) #1 value
			pool.add('harmonic_spectral_deviation', hsd_i) #1 value
			pool.add('harmonic_spectral_spread', hss_i) #1 value
						
			# to be fed output of harmonicpeaks algorithm
			inharm = inharmonicity(freqs, mags)
			pool.add('inharmonicity', inharm) #1 value
			
			coef, refl = lpc(frame)
			pool.add('lpc_coefficients', coef) #12 value
			pool.add('lpc_reflections', refl) #11 value
			
			mfcband, mfc = mfcc(s)
			pool.add('mfcc_bands', mfcband) #40 value
			pool.add('mfcc_coeff', mfc) #13 value
			
			mx = maxMagFreq(s)
			pool.add('max_freq', mx) #1 value
			
			#Odd to even harmonic energy ratio
			oddtoeven = oddToEven(freqs,mags)
			pool.add('odd_to_evenharmonic', oddtoeven) #1 value
			
			#threshold parameter not clear, skipped
			#silrt = silenceRate(frame)
			#pool.add('silence_rate', silrt)#1 value
			
			# Tristimulus
			tstmul = tristimulus(freqs, mags)
			pool.add('tristimulus', tstmul) #3 values
			
			zc = zcr(frame)
			pool.add('zcr', zc) #1 value
		
		#Calculating derivatives of frame features
		deriv = np.gradient(pool['spectral_contrast'])[0]
		for val in deriv:
			pool.add('Dspectral_contrast',val) #6 values
		
		deriv = np.gradient(pool['spectral_valley'])[0]
		for val in deriv:
			pool.add('Dspectral_valley',val) #6 values
		
		deriv = np.gradient(pool['hpcp'])[0]
		for val in deriv:
			pool.add('Dhpcp', val) #36 values
		
		deriv = np.gradient(pool['spectral_complexity'])
		for val in deriv:
			pool.add('Dspectral_complexity', val) #1 value
		
		#deriv = np.gradient(pool['bark_bands'])[0]
		#for val in deriv:
		#	pool.add('Dbark_bands', val) #27 values
		
		deriv = np.gradient(pool['dissonance'])
		for val in deriv:
			pool.add('Ddissonance', val) #1 value
		
		#deriv = np.gradient(pool['erb_bands'])[0]
		#for val in deriv:
		#	pool.add('Derb_bands', val) #40 values
		
		deriv = np.gradient(pool['energy_low'])
		for val in deriv:
			pool.add('Denergy_low', val) #1 value
		
		deriv = np.gradient(pool['energy_middle_low'])
		for val in deriv:
			pool.add('Denergy_middle_low', val) #1 value
		
		deriv = np.gradient(pool['energy_middle_high'])
		for val in deriv:
			pool.add('Denergy_middle_high', val) #1 value
		
		deriv = np.gradient(pool['energy_high'])
		for val in deriv:
			pool.add('Denergy_high', val) #1 value
		
		deriv = np.gradient(pool['spectral_flatness'])
		for val in deriv:
			pool.add('Dspectral_flatness', val) #1 value
		
		deriv = np.gradient(pool['spectral_flux'])
		for val in deriv:
			pool.add('Dspectral_flux', val) #1 value	
		
		#Giving NANS
		#deriv = np.gradient(pool['spectral_rms'])
		#for val in deriv:
		#	pool.add('Dspectral_rms', val) #1 value
		
		deriv = np.gradient(pool['spectral_rolloff'])
		for val in deriv:
			pool.add('Dspectral_rolloff', val) #1 value
		
		deriv = np.gradient(pool['spectral_strong_peak'])
		for val in deriv:
			pool.add('Dspectral_strong_peak', val) #1 value
		
		#deriv = np.gradient(pool['gfcc_bands'])[0]
		#for val in deriv:
		#	pool.add('Dgfcc_bands', val) #40 value
		
		#deriv = np.gradient(pool['gfcc_coeff'])[0]
		#for val in deriv:
		#	pool.add('Dgfcc_coeff', val) #13 value
		
		deriv = np.gradient(pool['hfc'])
		for val in deriv:
			pool.add('Dhfc', val) #1 value
		
		deriv = np.gradient(pool['equal_tempered_deviation'])
		for val in deriv:
			pool.add('Dequal_tempered_deviation', val) #1 value
		deriv = np.gradient(pool['non_tempered_energyratio'])
		for val in deriv:
			pool.add('Dnon_tempered_energyratio', val) #1 value
		deriv = np.gradient(pool['non_tempered_peaksenergyratio'])
		for val in deriv:
			pool.add('Dnon_tempered_peaksenergyratio', val) #1 value
		
		deriv = np.gradient(pool['harmonic_spectral_centroid'])
		for val in deriv:
			pool.add('Dharmonic_spectral_centroid', val) #1 value
		deriv = np.gradient(pool['harmonic_spectral_deviation'])
		for val in deriv:
			pool.add('Dharmonic_spectral_deviation', val) #1 value
		deriv = np.gradient(pool['harmonic_spectral_spread'])
		for val in deriv:
			pool.add('Dharmonic_spectral_spread', val) #1 value
		
		deriv = np.gradient(pool['inharmonicity'])
		for val in deriv:
			pool.add('Dinharmonicity', val) #1 value
		
		deriv = np.gradient(pool['lpc_coefficients'])[0]
		for val in deriv:
			pool.add('Dlpc_coefficients', val) #12 value
		deriv = np.gradient(pool['lpc_reflections'])[0]
		for val in deriv:
			pool.add('Dlpc_reflections', val) #11 value
		
		deriv = np.gradient(pool['mfcc_bands'])[0]
		for val in deriv:
			pool.add('Dmfcc_bands', val) #40 value
		deriv = np.gradient(pool['mfcc_coeff'])[0]
		for val in deriv:
			pool.add('Dmfcc_coeff', val) #13 value
		
		deriv = np.gradient(pool['max_freq'])
		for val in deriv:
			pool.add('Dmax_freq', val) #1 value
		
		deriv = np.gradient(pool['odd_to_evenharmonic'])
		for val in deriv:
			pool.add('Dodd_to_evenharmonic', val) #1 value
		
		deriv = np.gradient(pool['tristimulus'])[0]
		for val in deriv:
			pool.add('Dtristimulus', val) #3 values
		
		deriv = np.gradient(pool['zcr'])
		for val in deriv:
			pool.add('Dzcr', val) #1 value	
			#try:
			#	aggrPool = standard.PoolAggregator(defaultStats = ['mean', 'var', 'skew', 'kurt'])(pool)
			#except:
			#	print start_time/step, "failed"
			#	continue
			
			#aggrPool.add('onset_rate', rate)
			
			#print start_time, segment_length, start_time/segment_length
			#fileout = segmentArffDir+fname[:-4]+"_%003d%s"%(start_time/step, ".sig")
		featureFileName += '.arff'
		pwd = os.getcwd()
		if train_flag == 1:
			btime = trueTimes[idx]
			os.chdir(train_dir)
		else:
			btime = times[idx]
			os.chdir(test_dir)
			
		fname = open(featureFileName,'a')
		poolScaled = essentia.Pool()
		boundNumbers = np.arange(-(numBoundaries/2),(numBoundaries/2))
		score1 = norm.pdf(boundNumbers,0,5)
		score2 = norm.pdf(boundNumbers,0,10)
		score3 = norm.pdf(boundNumbers,0,20)
		score4 = norm.pdf(boundNumbers,0,40)
		score1 = scaleData(score1,1,0)
		score2 = scaleData(score2,1,0)
		score3 = scaleData(score3,1,0)
		score4 = scaleData(score4,1,0)
		
		for featName in pool.descriptorNames():
			poolScaled.add(featName,scaleData(pool[featName],1.0,-1.0)) # converts feature arrays to lists ??
		for numb in np.arange(0,numBoundaries):
			flist = ''
			for featName in poolScaled.descriptorNames():
				#for idx,feat in enumerate(pool[fname]
				#print numb
				conv = poolScaled[featName][0][numb] # compensating for the above list conversion
				intmd = np.array2string(conv)
				intmd = " ".join(intmd.split())
				intmd = intmd.strip('[')
				intmd = intmd.strip(']')
				intmd = intmd.strip()
				intmd += ' '
				flist += intmd
			flist += str(score1[numb])+ ' ' + str(score2[numb]) + ' ' + str(score3[numb]) +' '+ str(score4[numb])+ ' ' + str(btime)+' ' + str(trueTimes[idx])
			flist += '\n'
			fname.write(flist)
		
		fname.close()
		#print nameList
		os.chdir(pwd)
	#return (pool, poolScaled, nameList, nameListcln)
	#print nameListcln
	#print '\n'
	return (nameList, nameListcln)
	#return 0

def boundaryExtract(audioFile,nameList,nameListcln):
	song_folder = os.getcwd()+'/'
	wav_file = song_folder + 'song.wav'

	if audioFile[-4:] == '.mp3' or audioFile[-4:] == '.MP3':
		mp3_file = song_folder + audioFile
		command = "ffmpeg " + "-i " +'"'+ mp3_file +'"'+" -y " + " -ac 2 " + " -ar 44100 " +'"'+ wav_file +'"'
		#print '\n'
		print command
		p = subprocess.call(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True)
	
	elif audioFile[-4:] == '.wav' or audioFile[-4:] == '.WAV':
		shutil.copyfile(audioFile,wav_file)
		
	boundaryFile = audioFile.replace('.wav','.seg')
	[phones,times] = segFileRead(boundaryFile)
	pwd = os.getcwd()
	os.chdir('./true_seg')
	[truePhones,trueTimes] = segFileRead(boundaryFile)
	os.chdir(pwd)
	#print nameList
	#print nameListcln
	print audioFile
	[nameList, nameListcln] = featureExtractMain(wav_file,phones,times,trueTimes,nameList,nameListcln, 50.0/1000, 1.0/1000)
	print '\n'
	os.remove(wav_file)
	#return (pol, polScaled, nameList, nameListcln)
	return (nameList, nameListcln)
	
def analyseFolder(inputFolder):
	start = time()
	pwd = os.getcwd()
	os.chdir(inputFolder)
	fileList = glob.glob('*.wav') #fileList = glob.glob('*.mp3') if scanning working with mp3s
	nameList = {}
	if os.path.isdir(train_dir) == False:
		os.mkdir(train_dir)

	if os.path.isdir(test_dir) == False:
		os.mkdir(test_dir)
	
	for types in transitionTypes:
		  for typ in transitionTypes:
		  	name = types+'_to_'+typ
		  	nameList[name] = 0
	nameListcln = copy.deepcopy(nameList)
	#nameListret = copy.deepcopy(nameList)
	#fileList = fileList[0:2]
	for files in fileList:
		nameList = featureExtractDry(files,nameList)
	
	for idx,files in enumerate(fileList):
		print str(idx) + ' of ' + str(len(fileList))
		#[pol, polScaled, nameList, nameListcln] = boundaryExtract(files,nameList,nameListcln)
		[nameList, nameListcln] = boundaryExtract(files,nameList,nameListcln)
		#print nameListret
		#nameListcln = copy.deepcopy(nameListret)
		#print '\n'
	os.chdir(pwd)
	elapsed = (time() - start)
	print 'TIME TAKEN'
	print elapsed
	return (nameList, nameListcln)
	
#connecting sources and sinks
	'''loader.audio >> slices.audio
	slices.frame >> lowLevelFeat.signal
	lowLevelFeat.barkbands >> (lowLevelPool,'barkbands')
	lowLevelFeat.barkbands_kurtosis >> (lowLevelPool,'barkbands_kurtosis')
	lowLevelFeat.barkbands_skewness >> (lowLevelPool,'barkbands_skewness')
	lowLevelFeat.barkbands_spread >> (lowLevelPool,'barkbands_spread')
	lowLevelFeat.hfc >> (lowLevelPool,'hfc')
	lowLevelFeat.mffc >> (lowLevelPool,'mffc')
	lowLevelFeat.pitch >> (lowLevelPool,'pitch')
	lowLevelFeat.pitch_instantaneous_confidence >> (lowLevelPool,'pitch_instantaneous_confidence')
	lowLevelFeat.pitch_salience >> (lowLevelPool,'pitch_salience')
	lowLevelFeat.silence_rate_20dB >> (lowLevelPool,'silence_rate_20dB')
	lowLevelFeat.silence_rate_30dB >> (lowLevelPool,'silence_rate_30dB')
	lowLevelFeat.silence_rate_60dB >> (lowLevelPool,'silence_rate_60dB')
	lowLevelFeat.spectral_complexity >> (lowLevelPool,'spectral_complexity')
	lowLevelFeat.spectral_crest >> (lowLevelPool,'spectral_crest')
	lowLevelFeat.spectral_decrease >> (lowLevelPool,'spectral_decrease')
	lowLevelFeat.spectral_energy >> (lowLevelPool,'spectral_energy')trueTimes
	lowLevelFeat.spectral_energyband_low >> (lowLevelPool,'spectral_energyband_low')
	lowLevelFeat.spectral_energyband_middle_low >> (lowLevelPool,'spectral_energyband_middle_low')
	lowLevelFeat.spectral_energyband_middle_high >> (lowLevelPool,'spectral_energyband_middle_high')
	lowLevelFeat.spectral_energy_high >> (lowLevelPool,'spectral_energy_high')
	lowLevelFeat.spectral_flatness_db >> (lowLevelPool,'spectral_flatness_db')
	lowLevelFeat.spectral_flux >> (lowLevelPool,'spectral_flux')
	lowLevelFeat.spectral_rms >> (lowLevelPool,'spectral_rms')
	lowLevelFeat.spectral_rolloff >> (lowLevelPool,'spectral_rolloff')
	lowLevelFeat.spectral_strongpeak >> (lowLevelPool,'spectral_strongpeak')
	lowLevelFeat.zero_crossing_rate >> (lowLevelPool,'zero_crossing_rate')
	lowLevelFeat.inharmonicity >> (lowLevelPool,'inharmonicity')
	lowLevelFeat.tristimulus >> (lowLevelPool,'tristimulus')
	lowLevelFeat.oddtoevenharmonicenergyratio >> (lowLevelPool,'oddtoevenharmonicenergyratio')
	
		loader = stream.MonoLoader(filename = audioFile)
	slices = stream.Slicer(startTimes = timeStarts, endTimes = timeEnds)
	lowLevelPool = essentia.Pool()
	lowLevelFeat = stream.LowLevelSpectralfflaExtractor(frameSize = int(frameLength),hopSize = int(hop), sampleRate = samplingRate)
			# Read audio and some more info

	#run the extraction stream
	essentia.run(loader)'''
'''				for i in xrange(0, len(features[0])):
				pool.add('barkbands', features[0][i])
				pool.add('hfc', features[4][i])
				pool.add('pitch', features[6][i])
				pool.add('pitch_instantaneous_confidence', features[7][i])
				pool.add('pitch_salience', features[8][i])
				pool.add('silence_rate_20dB', features[9][i])
				pool.add('silence_rate_30dB', features[10][i])
				pool.add('silence_rate_60dB', features[11][i])
				pool.add('spectral_complexity', features[12][i])
				pool.add('spectral_crest', features[13][i])
				pool.add('spectral_decrease', features[14][i])
				pool.add('spectral_energy', features[15][i])
				pool.add('spectral_energyband_low', features[16][i])
				pool.add('spectral_energyband_middle_low', features[17][i])
				pool.add('spectral_energyband_middle_high', features[18][i])
				pool.add('spectral_energy_high', features[19][i])
				pool.add('spectral_flatness_db', features[20][i])
				pool.add('spectral_flux', features[21][i])
				pool.add('spectral_rms', features[22][i])
				pool.add('spectral_rolloff', features[23][i])
				pool.add('spectral_strongpeak', features[24][i])
				pool.add('zero_crossing_rate', features[25][i])
				pool.add('inharmonicity',  features[26][i])
				pool.add('tristimulus',  features[27][i])

			onsetRate = standard.OnsetRate()
			onsets, rate = onsetRate(segAudio)'''
