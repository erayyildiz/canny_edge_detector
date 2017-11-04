import sys
import canny_edge_detector
from os import listdir
from os.path import isfile, join

def main(argv):

    # Create EdgeDetector instance
    edge_detector = canny_edge_detector.EdgeDetector()

    # Check command line arguments
    if len(argv) == 2:
        # If command line arguments is equal to 2,
        # the first argument is the directory with input images
        # and the second argument is the directory for output images
        image_paths = [argv[0]+"/"+f for f in listdir(argv[0]) if isfile(join(argv[0], f))]
        output_path = argv[1]
    elif len(argv) == 0:
        # If no argument is provided, the program will use default directories for input and output images
        image_paths = ["test_images/" + f for f in listdir("test_images") if isfile(join("test_images", f))]
        output_path = "output_images/"
    else:
        raise NotImplementedError("Arguments length must be 2! "
                                  "First argument is path for input images and second argument is path for output "
                                  "images.")

    # Find images in input directory, find edges and writes edge images to the output directory
    for img in image_paths:
        img_file_name = img[img.rfind('/'):]
        edge_detector.detect_edges(img, output_path + img_file_name)

if __name__ == "__main__":
    main(sys.argv[1:])
