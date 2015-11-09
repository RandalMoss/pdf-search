import subprocess
import os
from PIL import Image, ImageEnhance
from tika import parser
import numpy as np
from skimage.filters import threshold_adaptive
from threading import Thread


class DetectImagePdf():
    def __init__(self):
        fullPath = os.path.realpath(__file__)
        config = open(('%s/' + 'config.txt') % os.path.dirname(fullPath), 'r')
        fileDirectory = config.readline().split('=')[1].strip('\n')
        outputDirectory = config.readline().split('=')[1].strip('\n')
        for subdir, dirs, files in os.walk(fileDirectory):
            for src in files:
                if(".DS_Store" in src):
                    continue
                thread = Thread(target=self.makeSearchable, args=(src, outputDirectory))
                thread.start()

    def makeSearchable(self, src, subdir):
        rootDir = subdir + "/examplePDFs"
        pdfPath = rootDir + "/" + "rawPdfs"
        finishedTextPath = rootDir + "/" + "finishedText"
        removed_text_path = rootDir + "/" + "removedText"
        gsPath = rootDir + "/" + "gsPdfs"
        imagesProcessedPath = rootDir + "/" + "imagesProcessed"
        imageText = rootDir + "/" + "imageText"

        if not os.path.exists(pdfPath):
            os.makedirs(pdfPath)
        if not os.path.exists(finishedTextPath):
            os.makedirs(finishedTextPath)
        if not os.path.exists(removed_text_path):
            os.makedirs(removed_text_path)
        if not os.path.exists(gsPath):
            os.makedirs(gsPath)
        if not os.path.exists(imagesProcessedPath):
            os.makedirs(imagesProcessedPath)
        if not os.path.exists(imageText):
            os.makedirs(imageText)

        filename, fileType = src.rsplit(".", 1)
        print("\n**********************")
        print("Processing file: " + filename)
        print("**********************\n")

        # Extact easy text
        print("Getting text that can be easily extracted...")
        rawText = parser.from_file(pdfPath + "/" + src)
        if rawText["content"] is None:
            print("Found no text to extract, continuing process")
        else:
            fileOutput = open(finishedTextPath + "/" + filename + ".txt", 'w')
            fileOutput.write(rawText["content"].encode("utf-8"))
            fileOutput.close()

        # Remove text from pdf
        print("Removing text from pdf")
        process1 = subprocess.Popen(['java', '-jar', 'PdfTextDeleter.jar', src, os.path.join(removed_text_path, src)])
        process1.wait()

        # Apply ghostscript to removed text pdfs
        if not os.path.exists(gsPath + "/" + filename + "-imgs"):
            os.makedirs(gsPath + "/" + filename + "-imgs")
        if not os.path.exists(rootDir + "/imagesProcessed/" + filename + "-imgs"):
            os.makedirs(rootDir + "/imagesProcessed/" + filename + "-imgs")
        if not os.path.exists(rootDir + "/imageText/" + filename + "-imgs"):
            os.makedirs(rootDir + "/imageText/" + filename + "-imgs")
        print("Converting left over pdf to images")
        process2 = subprocess.Popen(["gs", "-dNOPAUSE", "-sFONTPATH=/opt/local/share/ghostscript/9.16/Resource/Font/",
                   "-sDEVICE=pngalpha", "-r300", "-dBATCH", "-sOutputFile=" + gsPath + "/" + filename + "-imgs" + "/" + filename + "-%03d" ".png",
                   removed_text_path + "/" + src], env={'PATH': '/opt/local/bin/'})
        process2.wait()
        self.preprocessImages(rootDir, subdir, src)
        self.applyOCRToImages(rootDir, subdir, src)
        self.mergeTextFiles(rootDir, subdir, src)

    def preprocessImages(self, rootDir, subdir, srcFile):
        rootfilename, fileType = srcFile.rsplit(".", 1)
        for subdir, dirs, files in os.walk(rootDir + "/gsPdfs/" + rootfilename + "-imgs"):
            for src in files:
                if(".DS_Store" in src):
                    continue
                filename, fileType = src.rsplit(".", 1)
                print("Processing image")
                image = Image.open(subdir + "/" + src).convert('L')
                image = np.asarray(image)
                block_size = 7
                binary_adaptive = threshold_adaptive(image, block_size, method='gaussian', offset=-35, param=37)
                # scipy.misc.imsave(rootDir + "/imagesProcessed/" + rootfilename + "-imgs/" + "binary-" + src, binary_adaptive)
                # scipy.misc.imsave(rootDir + "/imagesProcessed/" + rootfilename + "-imgs/" + "reg-" + src, image)

    def applyOCRToImages(self, rootDir, subdir, src):
        rootfilename, fileType = src.rsplit(".", 1)
        for subdir, dirs, files in os.walk(rootDir + "/imagesProcessed/" + rootfilename + "-imgs"):
            for src in files:
                if(".DS_Store" in src):
                    continue
                filename, fileType = src.rsplit(".", 1)
                print("Extract text from image using tesseract")
                process = subprocess.Popen(["tesseract", rootDir + "/imagesProcessed/" + rootfilename + "-imgs" + "/" + src,
                                            rootDir + "/imageText/" + rootfilename + "-imgs" + "/" + filename, "-l", "eng"],
                                            env={'PATH': '/opt/local/bin/'}, stdout=subprocess.PIPE)
                process.wait()

    def mergeTextFiles(self, rootDir, subdir, src):
        rootfilename, fileType = src.rsplit(".", 1)
        for subdir, dirs, files in os.walk(rootDir + "/imageText/" + rootfilename + "-imgs"):
            for src in files:
                if(".DS_Store" in src):
                    continue
                print("Adding extracted image text to searchable text")
                unfinishedTextFile = open(rootDir + "/finishedText/" + rootfilename + ".txt", 'a')
                imageTextFile = open(subdir + "/" + src, 'r')
                unfinishedTextFile.write(imageTextFile.read())
                unfinishedTextFile.close()
                imageTextFile.close()

    def desaturateImage(self, rootDir, subdir, src):
        if ".ccitt" in src or ".params" in src or ".jb2e" in src:
            return
        img = Image.open(subdir + "/" + src)
        colorConverter = ImageEnhance.Color(img)
        img4 = colorConverter.enhance(0)
        contrastConverter = ImageEnhance.Contrast(img4)
        img2 = contrastConverter.enhance(1)
        sharpenConverter = ImageEnhance.Sharpness(img2.convert('RGB'))
        img3 = sharpenConverter.enhance(1)
        img3.save(rootDir + "/grayscale/" + src)

c1 = DetectImagePdf()
