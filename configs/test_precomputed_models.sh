# Before runing this file, you need to first run "download_precomputed_results.sh" to download the precomputed models
echo "Start unzipping ..."
path="runs/paper_config/"
cd $path
unzip -o diligent_precomputed.zip
cd ../..

# run testing on precomputed models
echo "Start testing ..."
CUDA_NUM="0"
TESTING="True"
QUICKTESTING="True"   # Change QUICKTESTING to "False" for more visualization results
echo cuda:$CUDA_NUM/Testing:$TESTING/quick_testing:$QUICKTESTING

FILES="configs/diligent/*.yml"
for f in $FILES
do
  python train.py --config $f --cuda $CUDA_NUM   --testing $TESTING --quick_testing $QUICKTESTING
done