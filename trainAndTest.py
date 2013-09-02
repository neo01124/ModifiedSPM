#!/usr/bin/python
#import cgitb
import numpy as np
import scipy
import sys
#import arff
import os
#from matplotlib import pyplot as plt
from sklearn.svm import NuSVR
#from sklearn.preprocessing import StandardScaler
from scipy import sparse
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from time import time,clock
from time import gmtime, strftime
import shlex
def MyScore(y_real,y_pred):
    count = 0
    err = 0
    countsc = 1
    for k in y_real:
        if(k==1):
            err = err+(y_real[count] - y_pred[count])
            countsc = countsc+1
            #print y_pred[count]
        count = count+1
    return err/countsc


#barr = ''
#fname = './bruno/2_mid_pitch/TestFeatures/Vowels_To_UnvoicedFricatives.arff'
#fname = './Vowels_To_Nasals.arff'
#fname2 = '../TestFeatures/Vowels_To_Nasals.arff'
def main(arg1):
	#print arg1
	fname = '../EssentiaTrainFeatures/'+ arg1 #Liquids_To_UnvoicedPlosives.arff'
	fname2 = './'+ arg1	#Liquids_To_UnvoicedPlosives.arff'
	start = time()
	try:
		f = open(fname,'r')
	except:
		return('error')
	#lines = f.readlines()[:]
	#f.close()       
	#floats = []
	#for line in lines:   
	#	floats.append(shlex.split(line))
	
	#array = np.asarray(floats)
	#for (x,y), value in np.ndenumerate(array):
	#	if value == np.nan or value == 'NaN':
	#		array[x][y] = 0;
	#	elif value == np.infty:
	#		array[x][y] = 1;
	array = np.loadtxt(f)
	f.close()
	array = np.nan_to_num(array)
	#array = array.astype(np.float)
	print 'Data size'
	print np.shape(array)
	#scale = StandardScaler()
	#array = scale.fit_transform(array)
	trainY = array[:,305]
	trainX = np.delete(array, [302,303,304,305,306,307],1)
	elapsed = time() - start
	print 'Training array loading time'
	print elapsed/60
	f = open(fname2,'r')
	#lines = f.readlines()[:]
	#f.close()       
	#floats = []
	#for line in lines:     
	#	floats.append(shlex.split(line))
	#array2 = np.asarray(floats)
	#for (x,y), value in np.ndenumerate(array2):
	#	if value == np.nan or value == 'NaN':
	#		array2[x][y] = 0;
         #       elif value == np.infty:
          #              array2[x][y] = 2;
	array2 = np.loadtxt(f)
	f.close()
	array2 = np.nan_to_num(array2)
	#array2 = array2.astype(np.float)
	print 'Test size'
	print np.shape(array2)
	#scale = StandardScaler()
	#array = scale.fit_transform(array)
	#traiY = array[:,38]
	#Position = array2[:,36]
	#Hmmboundary = array2[:,37]
	#Manualboundary = array2[:,38]
	hmm_true = array2[:,305]
	hmmX = np.delete(array2, [302,303,304,305,306,307],1)
	
	#trainY, realY, trainX, testX = train_test_split(traiY,traiX,test_size=0.8,random_state=42)
	#Cost = np.power(2,np.arange(1,12));
	#g = [0.5,0.25,0.125,0.0625,0.03125,0.015625,0.0078125,0.00390625,0.001953125,0.0009765625,0.00048828125,0.00048828125]
	#print '\nCost values'
	#print Cost
	#print '\ngamma values'
	#print g
	#scorebest = 0
	#Cbest = 0
	#gammabest = 0
	#model_to_set = NuSVR(C=32, cache_size=2048, coef0=0.0, degree=3, gamma=0.03125, kernel='rbf',
 	#  max_iter=-1, nu=0.5, probability=False, shrinking=True, tol=0.001,
  	# verbose=True)
	#parameters = {'C':Cost,'gamma':g}#,'nu':[0.5],'kernel':['rbf'],'verbose':[True]}
	#k =[0.5,1]#2,5,7,8];
	model_to_set = RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=5000, min_samples_split=2000, min_samples_leaf=10,min_density=0.1, max_features='auto', bootstrap=True, compute_importances=False, oob_score=False, n_jobs=3, random_state=None, verbose=0)
	#parameters = {'n_estimators':[10,100,500],'max_depth':[1,5,20,100,None],'min_samples_split':[1,5,20,100],}
	#trainY, realY, trainX, testX = train_test_split(traiY,traiX,test_size=0,random_state=42)
	print '\nparams'
	print model_to_set.get_params()
	start = time()
	print '\ntraining start time'
	print strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
	model_to_set.fit(trainX,trainY)
	print '\ntraining end time'
	print strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
	elapsed = (time() - start)
	print elapsed/60

	y_pred = model_to_set.predict(trainX)
	#return(y_pred,trainY)
	#score1 = model_to_set.score(trainX,trainY)
	#print 'score1'
	#print score1
	#print 'Myscore1'
	#print MyScore(trainY,y_pred)
	
	#y_pred = model_to_set.predict(testX)
	#score2 = model_to_set.score(testX,realY)
	#print '\nscore2'
	#print score2
	#print 'Myscore2'
	#print MyScore(realY,y_pred)
	
	'''TESTING'''
	
	hmm_pred = model_to_set.predict(hmmX)
	#baseName = arg1.replace('.arff','')
	#np.savetxt((baseName+'_hmm_pred.txt'),hmm_pred)
	#np.savetxt((baseName+'_hmm_true.txt'),hmm_true)
	#np.savetxt((baseName+'_Bhmm.txt'),array2[:,306])
	#np.savetxt((baseName+'_Btrue.txt'),array2[:,307])
	#np.savetxt((baseName+'_train_pred.txt'),y_pred)
	#np.savetxt((baseName+'_train_true.txt'),hmm_pred)
	#np.savetxt((baseName+'_Btrain.txt'),hmm_pred)	
	return(hmm_pred,hmm_true,array2[:,306],array2[:,307], y_pred,trainY, array[:,307])
	#cnt = 0;
	#print 'asdasd'
	#print hmm_pred
	'''for pred in hmm_pred:
		if pred > 0.85:
			print'\n'
			print pred
			print Position[cnt]
    	    	print Hmmboundary[cnt]
    	    	print Manualboundary[cnt]
			print 1000*(Manualboundary[cnt] - Position[cnt])
		cnt = cnt+1'''
	#change = Hmmboundary[0]
	#cnt = 0
	#max = 0
	#UHmmboundary = set(Hmmboundary)
	#change = Hmmboundary[0]
	#print len(UHmmboundary)
	#print UHmmboundary
	#totcnt = 0
	#gcnt = 0

	#for bound in Hmmboundary:
		#bound = Hmmboundary[cnt]
		#print cnt
		#print bound 
		#print '\n'
		#if  bound == change:
		#	if hmm_pred[cnt] >= hmm_pred[max]:
		#		max = cnt;
		
		#else:
		
		#	change = bound
		#	totcnt = totcnt+1
		#	print '\n'
		#	if (abs(Manualboundary[max]-Position[max]))<(abs(Manualboundary[max]-Hmmboundary[max])):
		#		print 'G'
		#		gcnt = gcnt+1
		#	else:
		#		print 'B'
		#	print cnt
		#	print hmm_pred[max]
		#	print Position[max]
		#	print Hmmboundary[max]
		#	print Manualboundary[max]
		#	print 1000*(Manualboundary[max] - Position[max])
		#	max = cnt
		
		#if cnt==(len(Hmmboundary)-1):
		#	totcnt = totcnt+1
		#	print '\n'
		#	if (abs(Manualboundary[max]-Position[max]))<(abs(Manualboundary[max]-Hmmboundary[max])):
		#		print 'G'
		#		gcnt = gcnt+1
		#	else:
		#		print 'B'
		#	print cnt
		#	print hmm_pred[max]
		#	print Position[max]
		#	print Hmmboundary[max]
		#	print Manualboundary[max]
		#	print 1000*(Manualboundary[max] - Position[max])	
		#cnt = cnt+1
	#print '\n'
	#print totcnt
	#print gcnt

if __name__ == "__main__":
    main(sys.argv[0])
#, sys.argv[2], sys.argv[3])
#score2 = model_to_set.score(testX,realY)
#print '\nscore2'
#print score2
#print 'Myscore2'
#print MyScore(realY,y_pred)
