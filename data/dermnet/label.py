#! /usr/bin/python
#
# Find skin diseases and label images
#

from os import listdir
from os.path import isfile, join
import re
import random

data_folder = "data/dermnet"
image_folder = data_folder + "/images"
image_taxonomy_file = data_folder + "/image_taxonomy.txt"
labeled_images_file = data_folder + "/train+test.txt"
test_images_file = data_folder + "/test.txt"
train_images_file = data_folder + "/train.txt"
image_trash_file = data_folder + "/invalid_images.txt"
image_dictionary_file = data_folder + "/image_dictionary.txt"

image_files = [image_file for image_file in listdir(image_folder)
               if isfile(join(image_folder, image_file))]
image_dictionary = {}
image_trash = []
image_taxonomy = {}
image_label = -1
image_classes = []
labeled_images = []
test_images = []
train_images = []

# find subclasses from image files
image_name_pattern = re.compile("(.*)-\d+.jpg$|(\d+[a-zA-Z]+)\d*.jpg$")
for image_file in image_files:
    image_name_match = image_name_pattern.match(image_file)
    if image_name_match:
        if image_name_match.group(1):
            image_class = image_name_match.group(1)
        else:
            image_class = image_name_match.group(2)
        if image_class in image_dictionary:
            image_dictionary[image_class].append(image_file)
        else:
            image_dictionary[image_class] = [image_file]
    else:
        image_trash.append(image_file)

# remove subclasses if it's size is too small
for image_class, image_files_per_class in image_dictionary.items():
    if len(image_files_per_class) < 20:
        image_trash.extend(image_files_per_class)
        del image_dictionary[image_class]

for image_class in image_dictionary.keys():
    if image_class.lower() in image_dictionary \
           and image_class != image_class.lower():
        image_files = image_dictionary.pop(image_class)
        image_dictionary[image_class.lower()].extend(image_files)
    else:
        image_files = image_dictionary.pop(image_class)
        image_dictionary[image_class.lower()] = image_files

# load class taxonomy (contains class to subclass mapping)
with open(image_taxonomy_file) as f:
    for line in f:
        if line[0] is "#":
            image_class = line[1:].strip()
            image_label += 1
	    image_classes.append(image_class)
        else:
            image_subclass = line.lower().strip().replace(" ", "-")
            image_taxonomy[image_subclass] = {"label": image_label,
                                              "class": image_class}

# build the entire class tree and label each image files in the tree
image_dictionary_temp = image_dictionary
image_dictionary = {}
for image_subclass in image_dictionary_temp.keys():
    if image_subclass in image_taxonomy:
        image_class = image_taxonomy[image_subclass]["class"]
        image_label = image_taxonomy[image_subclass]["label"]
        image_files = []
        for image_file in image_dictionary_temp[image_subclass]:
            image_file = join(image_folder, image_file)
            image_files.append((image_file, image_label))
        if image_class not in image_dictionary:
            image_dictionary[image_class] = {image_subclass: image_files}
        else:
            image_dictionary[image_class][image_subclass] = image_files
    else:
        image_trash.extend(image_dictionary_temp[image_subclass])

# collect all the labeled images
for image_class, image_subdictionary in image_dictionary.items():
    for image_subclass, image_files in image_subdictionary.items():
        labeled_images.extend(image_files)

# generate training and test image sets
s = len(labeled_images)
shuffled_indices = random.sample(xrange(s), s)
test_images = [labeled_images[i] for i in shuffled_indices[:1000]]
train_images = [labeled_images[i] for i in shuffled_indices[1000:]]

# write the labeled images to file
with open(labeled_images_file, 'wb') as f:
    f.write('\n'.join('%s %s' % x for x in labeled_images))

# write the test images to file
with open(test_images_file, 'wb') as f:
    f.write('\n'.join('%s %s' % x for x in test_images))

# write the train images to file
with open(train_images_file, 'wb') as f:
    f.write('\n'.join('%s %s' % x for x in train_images))

# write the discarded images to file
with open(image_trash_file, 'wb') as f:
    f.write('\n'.join(image_trash))

# write the image classes to file
with open(image_dictionary_file, 'wb') as f:
    image_dictionary_text = ""
    for image_label, image_class in enumerate(image_classes):
        image_dictionary_text += str(image_label) + ". " + image_class + "\n"
        for image_subclass in image_dictionary[image_class]:
            image_dictionary_text += "\t" + image_subclass + "\n"
    f.write(image_dictionary_text)
