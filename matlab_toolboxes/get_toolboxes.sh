wget https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/7943/versions/5/download/zip/freezeColors_v23_cbfreeze.zip
unzip freezeColors_v23_cbfreeze.zip -d freezeColors_v23_cbfreeze
rm freezeColors_v23_cbfreeze.zip

git clone https://github.com/chadagreene/cmocean.git
rm -rf cmocean/.git

wget https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/63731/versions/1/download/zip/MyCrustOpen070909.zip
unzip MyCrustOpen070909.zip -d MyCrustOpen
rm MyCrustOpen070909.zip