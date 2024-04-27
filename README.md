# SudokuSolver Web App 🧩
## Overview
This project implements a Sudoku solver application using computer vision techniques and deep learning. The application allows users to upload an image of a Sudoku puzzle, which is then processed to extract the puzzle grid and solve it using an algorithm based on backtracking. The solution is overlaid onto the original image and displayed back to the user.

[working]()

## How it Works
1. The user uploads an image of a Sudoku puzzle.
2. The application processes the image to detect the Sudoku grid and extract individual cells.
3. Each cell containing a digit is passed through a CNN for digit recognition.
4. The recognized digits are used to construct a Sudoku grid.
5. The Sudoku grid is solved using a backtracking algorithm.
6. The solution is overlaid onto the original image and presented to the user.
   
## Technologies Used
- Python
- OpenCV (Open Source Computer Vision Library)
- NumPy
- Keras (Deep Learning Library)
- Streamlit (Web application framework)

## Features
1. **Image Upload:** Users can upload an image of a Sudoku puzzle.
2. **Sudoku Grid Extraction:** The application extracts the Sudoku grid from the uploaded image using contour detection techniques.
3. **Digit Recognition:** Each digit in the Sudoku grid is recognized using a convolutional neural network (CNN).
4. **Sudoku Solving:** The recognized digits are used to solve the Sudoku puzzle algorithmically.
5. **Overlay Solution:** The solution is overlaid onto the original image and displayed to the user.
6. **Interactive Interface:** The application is built using Streamlit, providing an interactive and user-friendly interface.





