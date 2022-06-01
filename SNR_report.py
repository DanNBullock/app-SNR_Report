#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 17:33:52 2022

@author: dan
"""


import json
import os
import sys
import pandas as pd
import numpy as np
import nibabel as nib
import nilearn
#import scilpy as scil
sys.path.append('wma_pyTools')
import wmaPyTools.roiTools
import wmaPyTools.streamlineTools
import wmaPyTools.analysisTools 
from dipy.segment.tissue import TissueClassifierHMRF

# load inputs from config.json
with open('config.json') as config_json:
	config = json.load(config_json)

#identity reminder:
# 1 = gm
# 2 = subscortical
# 3 = wm
# 4 = CSF
# 5 = Lesions
# if os.path.exists(config['5tt']):
#     inputTissueNifti=nib.load(config['5tt'])
# else:
#     inputTissueNifti=None

# if os.path.exists(config['fa']):
#     fa_nifti=nib.load(config['fa'])
# else:
#     fa_nifti=None
    
# if os.path.exists(config['rd']):
#     rd_nifti=nib.load(config['rd'])
# else:
#     rd_nifti=None

# if os.path.exists(config['ad']):
#     ad_nifti=nib.load(config['ad'])
# else:
#     ad_nifti=None
    
# if os.path.exists(config['md']):
#     md_nifti=nib.load(config['md'])
# else:
#     md_nifti=None
    
# if os.path.exists(config['anat']):
#     refT1=nib.load(config['anat'])
# else:
#     refT1=None


inputTissueNifti=nib.load('5tt.nii.gz')

#everything except dwi should have been created and on the main path
dwi=nib.load(config['diff'])
bval=config['bval']
bvec=config['bvec']
refT1=nib.load(config['anat'])


tissues=['cortGM','subCortGM','WM','CSF','Path']
#run some clever conversions here for the 5TT

#first resample it to the dwi data
#WARNING this is going to cause all kinds of problems
if not inputTissueNifti.shape[0:3]==dwi.shape[0:3]:
    print('resampling 5 tissue mask to fit diffusion data')
    resampledFiveTissue=nilearn.image.resample_img(inputTissueNifti,target_affine=dwi.affine,target_shape=dwi.shape[0:3])
else:
    #just rename it, I guess
    resampledFiveTissue=inputTissueNifti
#if it's a 4d nifiti
if len(resampledFiveTissue.shape)==4:
    print('4D input 5TT nifti detected, converting to int based 3D')
    outTissueData=np.zeros(resampledFiveTissue.shape[0:3])
    #iterate across the tissue labels
    for iTissues in range(resampledFiveTissue.shape[3]):
        roundedDataMask=np.around(resampledFiveTissue.get_data()[:,:,:,iTissues]).astype(bool)
        print(str(np.sum(roundedDataMask))+' voxels for ' + tissues[iTissues])
        outTissueData[roundedDataMask]=iTissues+1
    tissueNifti=nib.Nifti1Image(outTissueData, resampledFiveTissue.affine)
else:
    #figure out what to do for 3d vis based input
    print('3D input 5TT nifti detected, converting to int based 3D')

#initialize the method
#tissueClassifier=TissueClassifierHMRF()

#initial_segmentation, final_segmentation, PVE=tissueClassifier.classify(refAnatT1.get_data(), nclasses=5, beta=0.1)

#outTissue=nib.Nifti1Image(final_segmentation, refAnatT1.affine, refAnatT1.header)
#nib.save(outTissue,'test5tt.nii.gz')

def compute_snr(dwi, bval, bvec, mask):
    #stealing from scilpy and dipy


    from dipy.segment.mask import median_otsu
    from scipy.ndimage.morphology import binary_dilation
    from dipy.io.gradients import read_bvals_bvecs
    import nibabel as nib
    
    #extract the mask if an input nifti was passed
    if isinstance(mask, nib.nifti1.Nifti1Image):
        mask=mask.get_data()
    
    
    bvals, bvecs = read_bvals_bvecs(bval, bvec)
    data = dwi.get_fdata(dtype=np.float32)
    #arbitrary value to compartmentalize bvals with
    bvalBinSize=500
    bvalBins=np.around(np. divide(bvals,bvalBinSize))
    roundedBvals=np.multiply(bvalBins,bvalBinSize)
    b0Indexes=np.where(roundedBvals==0)[0]

    #temporarily create and save down a noise mask.  Helps speed up processing
    if not os.path.exists('noise_mask.nii.gz'):
        print('No noise mask found.  Computing new noise mask')
        b0_mask, noise_mask = median_otsu(data, vol_idx=b0Indexes)
    
        # we inflate the mask, then invert it to recover only the noise
        noise_mask = binary_dilation(noise_mask, iterations=10).squeeze()
    
        # Add the upper half in order to delete the neck and shoulder
        # when inverting the mask
        noise_mask[..., :noise_mask.shape[-1]//2] = 1
    
        # Reverse the mask to get only noise
        noise_mask = (~noise_mask).astype('float32')
        
        noise_maskNifti=nib.Nifti1Image(noise_mask, dwi.affine)
        nib.save(noise_maskNifti,'noise_mask.nii.gz')
    else:
        print('Noise mask found in working directory.  Loading...')
        noise_maskNifti=nib.load('noise_mask.nii.gz')
        noise_mask=noise_maskNifti.get_data()
        
    val = {0: {'bvec': [0, 0, 0], 'bval': 0, 'mean': 0, 'std': 0}}
    for idx in range(data.shape[-1]):
        val[idx] = {}
        val[idx]['bvec'] = bvecs[idx]
        val[idx]['bval'] = bvals[idx]
        val[idx]['mean'] = np.mean(data[..., idx:idx+1][mask > 0])
        #because I want a report about the std in the specific tissue as well
        noiseMaskSTD= np.std(data[..., idx:idx+1][noise_mask > 0])
        val[idx]['std'] = np.std(data[..., idx:idx+1][mask > 0])
        if noiseMaskSTD == 0:
            raise ValueError('Your noise mask does not capture any data'
                             '(std=0). Please check your noise mask.')
    
        val[idx]['snr'] = val[idx]['mean'] / noiseMaskSTD

    return val

def fullSNR_report(dwi, bval, bvec, refT1=None, fiveTissue=None, other_niftiList=None,other_niftiNames=None):
    #test
    import nibabel as nib
    import numpy as np
    from dipy.io.gradients import read_bvals_bvecs
    import pandas as pd
    
    if np.logical_and(refT1==None,fiveTissue==None):
        raise ValueError('Either refT1 or five tissue type needed as input')
    
    if isinstance(dwi, str):
        dwi=nib.load(dwi)
    
    if isinstance(refT1, str):
        refT1=nib.load(refT1)
    if isinstance(fiveTissue, str):
        fiveTissue=nib.load(fiveTissue)
    
    #initialize a vector for the fa/rd/etc
    #additionalNiftiTypes=['fa','rd','ad','md']
    #additionalNiftiVec=[fa_nifti,rd_nifti,ad_nifti,md_nifti]
    additionalNiftiTypes=other_niftiNames
    additionalNiftiVec=other_niftiList
    
    #using the 5tt mrtrix convention
    tissues=['cortGM','subCortGM','WM','CSF','Path']
    metrics=['mean','std','snr']
    columnLabels=['source']
    #generate the column labels
    for iTissues in tissues:
        for iMetrics in metrics:
            columnLabels.extend([iTissues + '_'+ iMetrics])
            
    snrTable=pd.DataFrame(columns=columnLabels)
    print(str(range(len(np.unique(fiveTissue.get_data())))))
    #better have some information about the tissues in the 5tt mask
    for tissueIterator in range(len(np.unique(fiveTissue.get_data()))):
        currentMask=wmaPyTools.roiTools.multiROIrequestToMask(fiveTissue,tissueIterator+1,inflateIter=0)
        snrOut=compute_snr(dwi, bval, bvec, currentMask)
        #compute for dwi
        bvals, bvecs = read_bvals_bvecs(bval, bvec)
        #same as in the compute snr code    
        bvalBinSize=500
        bvalBins=np.around(np. divide(bvals,bvalBinSize))
        roundedBvals=np.multiply(bvalBins,bvalBinSize)
        #b0Indexes=np.where(roundedBvals==0)[0]
        uniqueBvals= np.unique(roundedBvals).astype(int)
        
        print ('Computing stastics for tissue type ' + tissues[tissueIterator] )
        for bvalIterator,curBval in enumerate(uniqueBvals):
            #get the current indexes for this bval
            curBvalIndexes=np.where(roundedBvals==np.unique(roundedBvals)[bvalIterator])[0]
            #place the current bvalue in the source column
            snrTable.at[bvalIterator,'source']=curBval
            #get the mean, std, and snr respectively
            snrTable.at[bvalIterator,tissues[tissueIterator]+'_'+metrics[0]]=np.mean([snrOut[icurBvalIndexes][metrics[0]] for icurBvalIndexes in curBvalIndexes])
            snrTable.at[bvalIterator,tissues[tissueIterator]+'_'+metrics[1]]=np.mean([snrOut[icurBvalIndexes][metrics[1]] for icurBvalIndexes in curBvalIndexes])
            snrTable.at[bvalIterator,tissues[tissueIterator]+'_'+metrics[2]]=np.mean([snrOut[icurBvalIndexes][metrics[2]] for icurBvalIndexes in curBvalIndexes])
            print ('performing SNR analysis for bval level ' + str(curBval) )
        
        
        for iAddIterator,iAdditionalNifti in enumerate(additionalNiftiVec):
            if not iAdditionalNifti==None:
                if isinstance(iAdditionalNifti, str):
                    currentAddNif=nib.load(iAdditionalNifti)
                else:
                    currentAddNif=iAdditionalNifti
                    #does this actually work
                print('Computing tissue-specific metrics for ' + additionalNiftiTypes[iAddIterator])
                currentSubsetFullData=currentAddNif.get_data()
                snrTable.at[iAddIterator+len(uniqueBvals),'source']=additionalNiftiTypes[iAddIterator]
                snrTable.at[iAddIterator+len(uniqueBvals),tissues[tissueIterator]+'_'+metrics[0]]=np.nanmean(currentSubsetFullData[currentMask.get_data()> 0])
                snrTable.at[iAddIterator+len(uniqueBvals),tissues[tissueIterator]+'_'+metrics[1]]=np.nanstd(currentSubsetFullData[currentMask.get_data()> 0])
                snrTable.at[iAddIterator+len(uniqueBvals),tissues[tissueIterator]+'_'+metrics[2]]='NaN' 
                
        
    return snrTable

other_niftiNames=['fa', 'md', 'ad', 'rd', 'cl', 'cp', 'cs', 'dk']
other_niftiList=[iNames+'.nii.gz' for iNames in other_niftiNames]

testOut=fullSNR_report(dwi, bval, bvec, refT1, tissueNifti,other_niftiList,other_niftiNames)

outDir='output'
if not os.path.exists(outDir):
    os.makedirs(outDir)

testOut.to_csv(os.path.join(outDir,'snr_report.csv'))

# get ROIS for each tissue type from NMT_segmentation_in_FA.nii.gz

#get mean snr for FA AD RD MD + each shell

