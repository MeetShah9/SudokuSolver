import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from io import BytesIO

class SudokuSolver:
    def __init__(self):
        pass

    @staticmethod
    def solve(bo):
        """
        Solves a Sudoku puzzle using backtracking algorithm.

        Args:
        - bo: 2D list representing the Sudoku board

        Returns:
        - True if the Sudoku puzzle is solved successfully, False otherwise
        """
        find = SudokuSolver.find_empty(bo)
        if not find:
            return True
        else:
            row, col = find
        for i in range(1, 10):
            if SudokuSolver.valid(bo, i, (row, col)):
                bo[row][col] = i
                if SudokuSolver.solve(bo):
                    return True
                bo[row][col] = 0
        return False

    @staticmethod
    def valid(bo, num, pos):
        """
        Checks if a number is valid to be placed in the given position on the Sudoku board.

        Args:
        - bo: 2D list representing the Sudoku board
        - num: Number to be checked for validity
        - pos: Tuple representing the position (row, column) to check the validity

        Returns:
        - True if the number is valid for the given position, False otherwise
        """
        # Check row
        for i in range(len(bo[0])):
            if bo[pos[0]][i] == num and pos[1] != i:
                return False
        # Check column
        for i in range(len(bo)):
            if bo[i][pos[1]] == num and pos[0] != i:
                return False
        # Check box
        box_x = pos[1] // 3
        box_y = pos[0] // 3
        for i in range(box_y * 3, box_y * 3 + 3):
            for j in range(box_x * 3, box_x * 3 + 3):
                if bo[i][j] == num and (i, j) != pos:
                    return False
        return True

    @staticmethod
    def print_board(bo):
        """
        Prints the Sudoku board in a human-readable format.

        Args:
        - bo: 2D list representing the Sudoku board
        """
        for i in range(len(bo)):
            if i % 3 == 0 and i != 0:
                print("- - - - - - - - - - - - - ")
            for j in range(len(bo[0])):
                if j % 3 == 0 and j != 0:
                    print(" | ", end="")
                if j == 8:
                    print(bo[i][j])
                else:
                    print(str(bo[i][j]) + " ", end="")

    @staticmethod
    def find_empty(bo):
        """
        Finds an empty cell (with value 0) on the Sudoku board.

        Args:
        - bo: 2D list representing the Sudoku board

        Returns:
        - Tuple representing the position (row, column) of an empty cell, or None if no empty cell is found
        """
        for i in range(len(bo)):
            for j in range(len(bo[0])):
                if bo[i][j] == 0:
                    return (i, j)  # row, col
        return None

class Utlis:
    def __init__(self, model_path):
        self.model = self.intializePredectionModel(model_path)

    def intializePredectionModel(self, model_path):
        """
        Initialize the CNN model for digit recognition.

        Args:
        - model_path: Path to the pre-trained model file

        Returns:
        - model: Loaded Keras model
        """
        model = load_model(model_path)
        return model

    def preProcess(self, img):
        """
        Preprocess the input image for digit extraction.

        Args:
        - img: Input image

        Returns:
        - imgThreshold: Preprocessed image
        """
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
        imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2)
        return imgThreshold

    def reorder(self, myPoints):
        """
        Reorder the points of a quadrilateral.

        Args:
        - myPoints: Array containing points of a quadrilateral

        Returns:
        - myPointsNew: Reordered points of the quadrilateral
        """
        myPoints = myPoints.reshape((4, 2))
        myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
        add = myPoints.sum(1)
        myPointsNew[0] = myPoints[np.argmin(add)]
        myPointsNew[3] = myPoints[np.argmax(add)]
        diff = np.diff(myPoints, axis=1)
        myPointsNew[1] = myPoints[np.argmin(diff)]
        myPointsNew[2] = myPoints[np.argmax(diff)]
        return myPointsNew

    def biggestContour(self, contours):
        """
        Find the biggest contour in the list.

        Args:
        - contours: List of contours

        Returns:
        - biggest: Biggest contour
        - max_area: Maximum area of contour
        """
        biggest = np.array([])
        max_area = 0
        for i in contours:
            area = cv2.contourArea(i)
            if area > 50:
                peri = cv2.arcLength(i, True)
                approx = cv2.approxPolyDP(i, 0.02 * peri, True)
                if area > max_area and len(approx) == 4:
                    biggest = approx
                    max_area = area
        return biggest, max_area

    def splitBoxes(self, img):
        """
        Split the Sudoku image into 81 different boxes.

        Args:
        - img: Input image

        Returns:
        - boxes: List of boxes
        """
        rows = np.vsplit(img, 9)
        boxes = []
        for r in rows:
            cols = np.hsplit(r, 9)
            for box in cols:
                boxes.append(box)
        return boxes

    def getPredection(self, boxes, model):
        result = []
        for image in boxes:
            ## PREPARE IMAGE
            img = np.asarray(image)
            img = img[4:img.shape[0] - 4, 4:img.shape[1] - 4]
            img = cv2.resize(img, (28, 28))
            img = cv2.GaussianBlur(img, (5, 5), 0)  # Apply Gaussian blur
            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # Apply Otsu's thresholding
            img = img / 255
            img = img.reshape(1, 28, 28, 1)
            ## GET PREDICTION
            predictions = model.predict(img)
            classIndex = np.argmax(predictions, axis=-1)
            probabilityValue = np.amax(predictions)
            ## SAVE TO RESULT
            if probabilityValue > 0.5:  # Adjust the probability threshold
                result.append(classIndex[0])  # Append the predicted digit, not the class index
            else:
                result.append(0)
        return result

    def displayNumbers(self, img, numbers, color=(0, 255, 0)):
        """
        Display the solution numbers on the image.

        Args:
        - img: Input image
        - numbers: Solution numbers
        - color: Color for displaying numbers

        Returns:
        - img: Image with displayed numbers
        """
        secW = int(img.shape[1] / 9)
        secH = int(img.shape[0] / 9)
        for x in range(0, 9):
            for y in range(0, 9):
                if numbers[(y * 9) + x] != 0:
                    cv2.putText(img, str(numbers[(y * 9) + x]),
                                (x * secW + int(secW / 2) - 10, int((y + 0.8) * secH)), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                2, color, 2, cv2.LINE_AA)
        return img

    def drawGrid(self, img):
        """
        Draw grid lines on the image.

        Args:
        - img: Input image

        Returns:
        - img: Image with drawn grid lines
        """
        secW = int(img.shape[1] / 9)
        secH = int(img.shape[0] / 9)
        for i in range(0, 9):
            pt1 = (0, secH * i)
            pt2 = (img.shape[1], secH * i)
            pt3 = (secW * i, 0)
            pt4 = (secW * i, img.shape[0])
            cv2.line(img, pt1, pt2, (255, 255, 0), 2)
            cv2.line(img, pt3, pt4, (255, 255, 0), 2)
        return img

    def stackImages(self, imgArray, scale):
        """
        Stack all the images in one window.

        Args:
        - imgArray: Array of images
        - scale: Scaling factor

        Returns:
        - ver: Stacked image
        """
        rows = len(imgArray)
        cols = len(imgArray[0])
        rowsAvailable = isinstance(imgArray[0], list)
        width = imgArray[0][0].shape[1]
        height = imgArray[0][0].shape[0]
        if rowsAvailable:
            for x in range(0, rows):
                for y in range(0, cols):
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                    if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
            imageBlank = np.zeros((height, width, 3), np.uint8)
            hor = [imageBlank] * rows
            hor_con = [imageBlank] * rows
            for x in range(0, rows):
                hor[x] = np.hstack(imgArray[x])
                hor_con[x] = np.concatenate(imgArray[x])
            ver = np.vstack(hor)
            ver_con = np.concatenate(hor)
        else:
            for x in range(0, rows):
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
                if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
            hor = np.hstack(imgArray)
            hor_con = np.concatenate(imgArray)
            ver = hor
        return ver

class SudokuSolverUtils:
    def __init__(self, model_path):
        self.sudoku_solver = SudokuSolver()
        self.utils = Utlis(model_path)

    def solve_sudoku_from_image(self, img):
        if img is None or len(img.shape) != 3:  # Check if the image is valid
            return None

        # Define width and height of the resized image
        heightImg = 450
        widthImg = 450

        # Resize the image
        img = cv2.resize(img, (widthImg, heightImg))

        imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)
        imgThreshold = self.utils.preProcess(img)

        imgContours = img.copy()
        imgBigContour = img.copy()
        contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3)

        biggest, maxArea = self.utils.biggestContour(contours)
        print(biggest)
        if biggest.size != 0:
            biggest = self.utils.reorder(biggest)
            print(biggest)
            cv2.drawContours(imgBigContour, biggest, -1, (0, 0, 255), 25) # DRAW THE BIGGEST CONTOUR
            pts1 = np.float32(biggest) # PREPARE POINTS FOR WARP
            pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
            matrix = cv2.getPerspectiveTransform(pts1, pts2) # GER
            imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
            imgDetectedDigits = imgBlank.copy()
            imgWarpColored = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)

            #### 4. SPLIT THE IMAGE AND FIND EACH DIGIT AVAILABLE
            imgSolvedDigits = imgBlank.copy()
            boxes = self.utils.splitBoxes(imgWarpColored)
            # print(len(boxes))
            # cv2.imshow("Sample",boxes[65])
            numbers = self.utils.getPredection(boxes, self.utils.model)
            print(numbers)
            imgDetectedDigits = self.utils.displayNumbers(imgDetectedDigits, numbers, color=(255, 0, 255))
            numbers = np.asarray(numbers)
            posArray = np.where(numbers > 0, 0, 1)
            # print(posArray)


        #### 5. FIND SOLUTION OF THE BOARD
            board = np.array_split(numbers,9)
            print(board)
            solved_board = []
            for row in board:
                solved_row = []
                for num in row:
                    if num == 0:
                        solved_row.append(0)
                    else:
                        solved_row.append(num)
                solved_board.append(solved_row)

            try:
                self.sudoku_solver.solve(solved_board)
            except:
                pass
            # print(board)
            flatList = []
            for sublist in solved_board:
                for item in sublist:
                    flatList.append(item)
            solvedNumbers = flatList * posArray
            imgSolvedDigits= self.utils.displayNumbers(imgSolvedDigits,solvedNumbers)

            # #### 6. OVERLAY SOLUTION
            pts2 = np.float32(biggest) # PREPARE POINTS FOR WARP
            pts1 =  np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
            matrix = cv2.getPerspectiveTransform(pts1, pts2)  # GER
            imgInvWarpColored = img.copy()
            imgInvWarpColored = cv2.warpPerspective(imgSolvedDigits, matrix, (widthImg, heightImg))
            inv_perspective = cv2.addWeighted(imgInvWarpColored, 1, img, 0.5, 1)
            imgDetectedDigits = self.utils.drawGrid(imgDetectedDigits)
            imgSolvedDigits = self.utils.drawGrid(imgSolvedDigits)

            imageArray = ([img,imgThreshold,imgContours, imgBigContour],
                        [imgDetectedDigits, imgSolvedDigits,imgInvWarpColored,inv_perspective])
            stackedImage = self.utils.stackImages(imageArray, 1)
            return stackedImage
        else:
            return None

# Streamlit app
def main():
    st.title("Sudoku Solver")
    uploaded_file = st.sidebar.file_uploader("Upload Image")

    model_path = "digit_model.h5"
    solver_utils = SudokuSolverUtils(model_path)

    if uploaded_file is not None:
        # Read image as bytes
        image_bytes = uploaded_file.getvalue()
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        # Decode numpy array as OpenCV image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        st.image(img, caption='Uploaded Image.', use_column_width=True)
        if st.button("Solve"):
            solved_image = solver_utils.solve_sudoku_from_image(img)
            if solved_image is not None:
                st.image(solved_image, caption='Solved Sudoku.', use_column_width=True)
            else:
                st.write("No Sudoku puzzle found in the uploaded image.")

if __name__ == "__main__":
    main()
