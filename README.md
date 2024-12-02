# Install LHAPDF

https://lhapdf.hepforge.org/install.html

```
wget https://lhapdf.hepforge.org/downloads/?f=LHAPDF-6.5.4.tar.gz -O LHAPDF-6.5.4.tar.gz
tar xf LHAPDF-6.5.4.tar.gz
cd LHAPDF-6.5.4
./configure --prefix=$HOME/LHAPDF
make
make install
```

# Install YADISM

https://github.com/NNPDF/yadism

```
pip install -e '.[mark, box]' --user
```

# Download PDF

```
lhapdf install NNPDF40_nnlo_as_01180
```
