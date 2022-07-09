# downloading precomputed DiLiGenT models
echo "downloading precomputed models"
path="runs/paper_config/"
mkdir -p $path
cd $path
echo "Start downloading ..."
wget https://www.dropbox.com/s/74nbauzt1h8rkb3/diligent_precomputed.zip
echo "done!"

# The precomputed models is downloaded.
# Run "test_precomputed_results.sh" to get the tested results