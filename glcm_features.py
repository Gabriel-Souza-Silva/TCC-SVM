# run in python2, figure out in python3
import skimage.io
import skimage.feature

im = skimage.io.imread('./img/texture_sample1.jpeg', as_grey=True)
im = skimage.img_as_ubyte(im)
im /= 32

g = skimage.feature.greycomatrix(im, [1], [0], levels=8, symmetric=False, normed=True)

print (skimage.feature.greycoprops(g, 'contrast')[0][0])
print (skimage.feature.greycoprops(g, 'energy')[0][0])
print (skimage.feature.greycoprops(g, 'homogeneity')[0][0])
print (skimage.feature.greycoprops(g, 'correlation')[0][0])