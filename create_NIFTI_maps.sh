#!/bin/bash

#stolen blatently from McPherson's racetracks
#https://github.com/brainlife/app-mrtrix3-act/tree/1.4

## define number of threads to use
NCORE=16

## raw inputs
DIFF=`jq -r '.diff' config.json`
BVAL=`jq -r '.bval' config.json`
BVEC=`jq -r '.bvec' config.json`
ANAT=`jq -r '.anat' config.json`

rm -rf ./tmp
mkdir ./tmp

difm=dwi
mask=mask
anat=t1

## convert anatomy
mrconvert $ANAT ${anat}.mif -force -nthreads $NCORE -quiet

echo "Converting raw data into MRTrix3 format..."
mrconvert -fslgrad $BVEC $BVAL $DIFF ${difm}.mif --export_grad_mrtrix ${difm}.b -force -nthreads $NCORE -quiet

## create mask of dwi data - use bet for more robust mask
bet $DIFF bet -R -m -f 0.1 -g 0.3
mrconvert bet_mask.nii.gz ${mask}.mif -force -nthreads $NCORE -quiet
#basically just a rename?
mrconvert ${mask}.mif ${mask}.nii.gz -force -nthreads $NCORE -quiet


## estimate multishell tensor w/ kurtosis and b-value scaling
echo "Fitting multi-shell tensor model..."
dwi2tensor -mask ${mask}.mif ${difm}.mif -dkt dk.mif dt.mif -force -nthreads $NCORE -quiet

## create tensor metrics either way
tensor2metric -mask ${mask}.mif -adc md.mif -fa fa.mif -ad ad.mif -rd rd.mif -cl cl.mif -cp cp.mif -cs cs.mif dt.mif -force -nthreads $NCORE -quiet

echo "Creating 5-Tissue-Type (5TT) tracking mask..."
5ttgen fsl ${anat}.mif 5tt.mif -mask ${mask}.mif -nocrop -sgm_amyg_hipp -tempdir ./tmp $([ "$PREMASK" == "true" ] && echo "-premasked") -force -nthreads $NCORE -quiet

5tt2vis 5tt.mif 5ttvis.mif -force -nthreads $NCORE -quiet

mrconvert dk.mif -stride 1,2,3,4 dk.nii.gz -force -nthreads $NCORE -quiet
## tensor outputs
mrconvert fa.mif -stride 1,2,3,4 fa.nii.gz -force -nthreads $NCORE -quiet
mrconvert md.mif -stride 1,2,3,4 md.nii.gz -force -nthreads $NCORE -quiet
mrconvert ad.mif -stride 1,2,3,4 ad.nii.gz -force -nthreads $NCORE -quiet
mrconvert rd.mif -stride 1,2,3,4 rd.nii.gz -force -nthreads $NCORE -quiet

## westin shapes (also tensor)
mrconvert cl.mif -stride 1,2,3,4 cl.nii.gz -force -nthreads $NCORE -quiet
mrconvert cp.mif -stride 1,2,3,4 cp.nii.gz -force -nthreads $NCORE -quiet
mrconvert cs.mif -stride 1,2,3,4 cs.nii.gz -force -nthreads $NCORE -quiet

## tensor itself
mrconvert dt.mif -stride 1,2,3,4 tensor.nii.gz -force -nthreads $NCORE -quiet

## 5 tissue type visualization
mrconvert 5ttvis.mif -stride 1,2,3,4 5ttvis.nii.gz -force -nthreads $NCORE -quiet
mrconvert 5tt.mif -stride 1,2,3,4 5tt.nii.gz -force -nthreads $NCORE -quiet

chmod 777 *
