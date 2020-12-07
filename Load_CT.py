import numpy as np 
import pydicom
import os
import pathlib
import glob
import pickle
from pydicom import dcmread
from pydicom.data import get_testdata_file
import matplotlib.pyplot as plt 
#For creating contour polygons for images: 
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from PIL import Image, ImageDraw

def NormalizeImage(image):
    ptp = np.ptp(image)
    amin = np.amin(image)
    return (image - amin) / ptp

def ImageUpsizer(array, factor):
    #Take an array and supersize it by the factor given
    xLen, yLen = array.shape
    newArray = np.zeros((factor * xLen, factor * yLen))
    #first get the original values in to the grid: 
    for i in range(xLen):
        for j in range(yLen):
            newArray[i * factor, j * factor] = array[i,j]
    #sample along first dim
    for j in range(yLen):
        for i in range(xLen - 1):
            insert = 1 
            while insert <= factor - 1:
                newArray[i * factor + insert, j * factor] = newArray[i * factor, j * factor] + (insert / factor) * (newArray[(i+1) * factor, j * factor]- newArray[i * factor, j * factor])
                insert += 1
    #sample along second dim
    for i in range(xLen * factor):
        for j in range(yLen - 1):
            insert = 1 
            while insert <= factor - 1:
                newArray[i, j * factor + insert] = newArray[i, j * factor] + (insert / factor) * (newArray[i, (j+1) * factor]- newArray[i, j * factor])
                insert += 1
    return newArray

def MakeContourImage(IPP, pixelSpacing, contours):
    #Need to go throug each contour point and convert x and y values to integer values indicating image indices
    for contour in contours:
        for point in contour:
            point[0] = round((point[0] - IPP[0]) / pixelSpacing[0]) 
            point[1] = round((point[1] - IPP[1]) / pixelSpacing[1]) 
    return contours        


def FindROINumber(metadata, containsList):
    for element in metadata:
        name = element.get("ROIName").lower()
        #now make sure all containsList words are in the name 
        allWordsIn = True
        for i in range(len(containsList)):
            if containsList[i] not in name:
                allWordsIn = False
        if allWordsIn:
            return element.get("ROINumber")
            
    return 1111 #keyword for didn't find anything

def Load_Data(patientsPath, containsList, save =True):
    #Get list of patient folders: 
    filesFolder = os.path.join(pathlib.Path(__file__).parent.absolute(), patientsPath)
    patientFolders = os.listdir(filesFolder)
    trainImages = []
    trainContours = []
    trainCombinedImages = []

    for p in range(len(patientFolders)):
        #Load the first patient
        patient = sorted(glob.glob(os.path.join(filesFolder, patientFolders[p], "*")))
        patient_CTs = []
        patient_Struct = []

        #Now get patient1 's CT scans: 
        for fileName in patient:
            if "CT" in fileName and "STRUCT" not in fileName:
                patient_CTs.append(fileName)
            elif "STRUCT" in fileName:
                patient_Struct.append(fileName)    
        #going to need the reference frame for the CT images, so get Image Position Patient and Image Orientation Patient
        #fix later, but for now assume that the files are in standard form for orientation (first dimension x axis , second y...)
        iop = dcmread(patient_CTs[0]).get("ImageOrientationPatient")
        ipp = dcmread(patient_CTs[0]).get("ImagePositionPatient")
        #also need the pixel spacing
        pixelSpacing = dcmread(patient_CTs[0]).get("PixelSpacing")
        sliceThickness = dcmread(patient_CTs[0]).get("SliceThickness")
        #Now I want numpy arrays of all these CT images. I save these in a list, where each element is itself a list, with index 0: CT image, index 1: z-value for that slice
        CTs = []
        ssFactor = 1 #at the moment don't need to upsample
        for item in pixelSpacing:
            item *= ssFactor
        for CTFile in patient_CTs:
            resizedArray = ImageUpsizer(np.array(dcmread(CTFile).pixel_array) , ssFactor)
            #Now normalize the image so its values are between 0 and 1
            resizedArray = NormalizeImage(resizedArray)
            CTs.append( [ resizedArray, dcmread(CTFile).data_element("ImagePositionPatient").value[2]])
        CTs.sort(key=lambda x:x[1]) #not necessarily in order, so sort according to z-slice.





        structsMeta = dcmread(patient_Struct[0]).data_element("ROIContourSequence")
        oC_Number= FindROINumber(dcmread(patient_Struct[0]).data_element("StructureSetROISequence"), containsList)
        #print(dcmread(patient1_Struct[0]).data_element("ROIContourSequence")[0])
        contours = []
        for contourInfo in structsMeta:
            if contourInfo.get("ReferencedROINumber") == oC_Number:
                for contoursequence in contourInfo.ContourSequence: #take away 0!!
                    contours.append(contoursequence.ContourData)
                    #But this is easier to work with if we convert from a 1d to a 2d list for contours
                    tempContour = []
                    i = 0
                    while i < len(contours[-1]):
                        x = float(contours[-1][i])
                        y = float(contours[-1][i + 1])
                        z = float(contours[-1][i + 2])
                        tempContour.append([x, y, z ])
                        i += 3
                    contours[-1] = tempContour    

        #Right now I need the contour points in terms of the image dimensions so that they can be turned into an image for training:
        contourIndices = MakeContourImage(ipp, pixelSpacing, contours)

        #Now need to make images of the same size as CTs, with just contours showing 
        contourImages = []
        combinedImages = []
        for CT in CTs:
            contourOnImage = False #keep track if a contour is on this image, if not just add a blank image.
            xLen = np.size(CTs[0][0], axis = 0)
            yLen = np.size(CTs[0][0], axis = 1)
            
            #Now need to add the contour polygon to this image, if one exists on the current layer
            #loop over contours to check if z-value matches current CT.
            for contour in contourIndices:
                if contour[0][2] == CT[1]:    #if the contour is on the current slice
                    contourImage = Image.new('L', (xLen, yLen), 0 )#np.zeros((xLen, yLen))
                    combinedImage = Image.fromarray(CT[0])
                    contourPoints = []
                    #now add all contour points to contourPoints as Point objects
                    for pointList in contour:
                        contourPoints.append((int(pointList[0]), int(pointList[1])))
                    contourPolygon = Polygon(contourPoints)
                    ImageDraw.Draw(contourImage).polygon(contourPoints, outline= 1, fill = 0)
                    ImageDraw.Draw(combinedImage).polygon(contourPoints, outline= 1, fill = 0)

                    contourImage = np.array(contourImage)
                    contourImages.append(contourImage)

                    combinedImage = np.array(combinedImage)
                    # plt.imshow(CT[0], cmap = "gray")
                    # plt.show()
                    # plt.imshow(contourImage, cmap = "gray")
                    # plt.show()
                    combinedImages.append(combinedImage)
                    contourOnImage = True
                    break
            if not contourOnImage:
                #if no contour on that layer, add just zeros array
                contourImage = np.zeros((xLen,yLen))
                contourImages.append(contourImage)
                combinedImages.append(CT[0])   
        trainImages.append(CTs)
        trainContours.append(contourImages)
        trainCombinedImages.append(combinedImages)
        #If wanted, save the objects
        if save == True:
            with open(os.path.join(pathlib.Path(__file__).parent.absolute(), "SavedImages/trainImages.txt"), "wb") as fp:
                pickle.dump(trainImages, fp)
            with open(os.path.join(pathlib.Path(__file__).parent.absolute(), "SavedImages/trainContours.txt"), "wb") as fp:
                pickle.dump(trainContours, fp)
            with open(os.path.join(pathlib.Path(__file__).parent.absolute(), "SavedImages/trainCombinedImages.txt"), "wb") as fp:
                pickle.dump(trainCombinedImages, fp)
    return trainImages, trainContours, trainCombinedImages

def LoadData(load = False):
    path = 'Patient_Files/'
    organKeywords = ["body"]
    if load == False:
        trainImages, trainContours, trainCombinedImages = Load_Data(path, organKeywords, save=True)
    else:
        trainImages = pickle.load(open(os.path.join(pathlib.Path(__file__).parent.absolute(), "SavedImages/trainImages.txt"), 'rb'))
        trainContours = pickle.load(open(os.path.join(pathlib.Path(__file__).parent.absolute(), "SavedImages/trainContours.txt"), 'rb'))
        trainCombinedImages = pickle.load(open(os.path.join(pathlib.Path(__file__).parent.absolute(), "SavedImages/trainCombinedImages.txt"), 'rb'))    
    #Plot a CT scan with the contour: 
    plt.imshow(trainImages[0][57][0], cmap = "gray")
    plt.show()
    plt.imshow(trainContours[0][57], cmap = "gray")
    plt.show()

if __name__ == "__main__":
    LoadData(load = True)