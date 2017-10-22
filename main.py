import sys
import canny_edge_detector
from os import listdir
from os.path import isfile, join

def main(argv):
    edge_detector = canny_edge_detector.EdgeDetector()

    if len(argv) == 2:
        image_paths = [argv[0]+"/"+f for f in listdir(argv[0]) if isfile(join(argv[0], f))]
        output_path = argv[1]
    elif len(argv) == 0:
        image_paths = ["test_images/" + f for f in listdir("test_images") if isfile(join("test_images", f))]
        output_path = "output_images/"
    else:
        raise NotImplementedError("Arguments length must be 2! "
                                  "First argument is path for input images and second argument is path for output "
                                  "images.")

    for img in image_paths:
        img_file_name = img[img.rfind('/'):]
        edge_detector.detect_edges(img, output_path + img_file_name)

if __name__ == "__main__":
    main(sys.argv[1:])
