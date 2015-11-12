import numpy as np
import matplotlib.pyplot as plt
import sys, re

caffe_root = "./"
deploy_protocol = caffe_root + "models/dermnet/vgg16_train_val.prototxt"
caffe_model = caffe_root + "models/dermnet/vgg16_iter_61956.caffemodel"
image_dictionary_file = caffe_root + "data/dermnet/image_dictionary.txt"
test_images_file = caffe_root + "data/dermnet/test.txt"
test_results_file = caffe_root + "data/dermnet/vgg16_outputs.txt"
sys.path.insert(0, caffe_root + "python")

import caffe

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

caffe.set_device(1)
caffe.set_mode_gpu()
net = caffe.Net(deploy_protocol,
                caffe_model,
                caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.load(caffe_root +
                                     'python/caffe/imagenet/' +
                                     'ilsvrc_2012_mean.npy').mean(1).mean(1))
transformer.set_raw_scale('data', 255)

net.blobs['data'].reshape(1, 3, 224, 224)

with open(test_images_file) as f:
    test_images = [line.strip().split(' ', 1) for line in f]

image_classes = []
pattern = re.compile("(^\d+\. .+)$")
with open(image_dictionary_file) as f:
    for line in f:
	match = pattern.match(line.strip())
	if match:
	    image_classes.append(match.group(1))

accuracy_top1 = 0
accuracy_top5 = 0
is_top1 = False
is_top5 = False
n = 0
with open(test_results_file, 'wb') as f:
    for test_image in test_images:
        net.blobs['data'].data[...] = transformer.preprocess('data',
    							 caffe.io.load_image(caffe_root + test_image[0]))
        out = net.forward()
	# Print to standard output
        print("\nImage: %s." % (test_image[0]))
        print("Actual class: #%s." % (test_image[1]))
        print("Predicted class: #{}.".format(out['prob'][0].argmax()))

	# Print to file
	print >>f, "\nImage: %s" % (test_image[0])
	print >>f, "Actual class: #%s." % (test_image[1])
	print >>f, "Predicted class: #{}.".format(out['prob'][0].argmax())
        
	# Output top-5 prediction 
        scores = zip(image_classes, out['prob'][0].tolist())
        scores.sort(key=lambda tup: tup[1])
        for score in scores[::-1][:5]:
            print("%s: %f" % (score[0], score[1]))
	    print >>f, "%s: %f" % (score[0], score[1])
	    if test_image[1] == score[0].split(' ', 1)[0][:-1]:
	       	accuracy_top5 += 1
		is_top5 = True
    
        if int(test_image[1]) == out['prob'][0].argmax():
	    accuracy_top1 += 1
	    is_top1 = True
    
        print("Top-1: %s Top-5: %s" % (str(is_top1), str(is_top5)))
	print >>f, "Top-1: %s Top-5: %s" % (str(is_top1), str(is_top5))
	is_top1 = False
        is_top5 = False
        
        n += 1
        print("Top-1 accuracy: %f" % (float(accuracy_top1) / n))
        print("Top-5 accuracy: %f" % (float(accuracy_top5) / n))
	print >>f, "Top-1 accuracy: %f" % (float(accuracy_top1) / n)
	print >>f, "Top-5 accuracy: %f" % (float(accuracy_top5) / n)
