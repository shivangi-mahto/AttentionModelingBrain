import cottoncandy as cc
import numpy as np

access_key = 'SSE14CR7P0AEZLPC7X0R'
secret_key = 'K0MmeXiXotrGIiTeRwEKizkkhR4qFV8tr8cIXprI'
endpoint_url = 'http://c3-dtn02.corral.tacc.utexas.edu:9002/'
cci = cc.get_interface('story-mri-data', ACCESS_KEY=access_key, SECRET_KEY=secret_key,endpoint_url=endpoint_url)
a = False; b = False; c = True 

sub = 'AHfs'
#1. For downloading fmri data for a subject
if (a):
    R_obs = cci.download_raw_array(sub+'/wheretheressmoke-10sessions')
    np.save('fmri_'+sub+'.dat',R_obs)

#2. For fetching JSON file of subject xfms 
if (b):
    xfm_file =  cci.download_json('subject_xfms')
    print (xfm_file[sub])

#3. For finding voxels with highest correlation over time. 
if (c):
    for sub in ['SJ']: #,'SS','EB03','EB05','S03']:
        R_obs = cci.download_raw_array(sub+'/wheretheressmoke-10sessions')
        #R_obs = np.load('fullmatrix_R_'+ sub +'.dat.npy')
        Voxels = R_obs.shape[2]; Repeat = R_obs.shape[0]
        score = np.zeros([Voxels])
        for v in range(Voxels):
            Corr = np.zeros([Repeat, Repeat])
            for i in range(Repeat):
                for j in range(Repeat): #(i, Repeat)
                    #score[v] += (np.corrcoef(R_obs[i,:,v], R_obs[j,:,v])[0,1])  
                    Corr[i,j] = (np.corrcoef(R_obs[i,:,v], R_obs[j,:,v])[0,1])
                    Corr[j,i] = Corr[i,j]
            score[v] = np.sum(Corr)
    
        argmax = np.argsort(score)[-10000:]
        Total_corr = np.zeros([Repeat, Repeat])
        for v in argmax:
            Corr = np.zeros([Repeat, Repeat])
            for i in range(Repeat):
                for j in range(Repeat): #(i, Repeat)       
                    Corr[i,j] = (np.corrcoef(R_obs[i,:,v], R_obs[j,:,v])[0,1])
                    Corr[j,i] = Corr[i,j]
            Total_corr += Corr
        
        Total_corr /= 10000.0
        Total_corr.dump('Total_corr_'+sub+'.dat')
        #score[argmax]
        #print(argmax)
"""
[38115 28545 35296 46609 26006 35295 31626 43965 38103 41070 37703 32681
 85047 32528 28601 35560 46807 52217 52868 46969 37834 32305 40694 81105
 31575 28992 52218 43855 31681 44078 11349 37704 32304 43768 35617 46917
 35619 28424 32250 46608 52867 34717 32249 49381 44273 35116 35240 32251
 85046 38288 37747 35563 32474 31794 11347 34770 43815 28533 38289 28544
 32473 52171 37833 52170 31963 32527 35565 31739 32306 43769 46478 35238
 38159 43756 46691 40654 31574 49520 37791 35115 43854 31628 35124 26269
 52215 46744 52169 52168 35561 35562 35239 46915 28600 37790 28478 11348
 46916 35559 37748 46522 35114 43806 31683 31737 35181 31738 34718 46968
 43759 38052 38104 43904 49570 43714 41018 38162 41071 38161 38106 38105]
"""
"""
[35473 17478 35896 34926 32327 19723 76324 76186 35844 76132 52435 26244
 82852 49678 19724 29232 31588 29176 31762 19833 78459 34968 25687 83026
 32057 84845 29361 29676 52478 35377 78644 80936 52479 32402 37996 32339
 32213 44166 73887 78692 23120 13744 31546 52436 35267 76372 32401 20356
 83071 29363 29060 41039 52476 31996 78557 35471 85007 41267 35057 52434
 31800 29132 34731 22674 32109 29418 82940 29677 31918 32282 19778 34885
 52394 35791 52395 34835 22064 31801 35842 35843 52433 52438 32058 28750
 29071 31686 34925 16841 22838 29362 35472 25686 32270 35790 32110 31730
 35009 31917 49760 28611 28714 31957 32214 35430 76373 31853 28664 34730
 49800 52393 28663 32271 31773 34693 28701 31774 31814 28612 31732 31731]
"""
