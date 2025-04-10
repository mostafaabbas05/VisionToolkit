import scipy.ndimage as ndimage
import sys
import cv2
import numpy as np
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog, QGraphicsScene, QGraphicsPixmapItem, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSlider


class MyApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # Load the UI file
        uic.loadUi(r"C:\Users\S\Desktop\Amit\env\lect\VisionToolkit\design.ui", self)
        # self.last_processed_image = None 
        self.processed_image = None  # 
        # self.image = None  # Store the loaded color image
        # self.gray_image = None  # Store the grayscale version
        
        # Connect buttons to functions
        
        self.pushButton.clicked.connect(self.load_image) # Load image button
        
        # Edge Detection Buttons
        self.pushButton_2.clicked.connect(lambda: self.edge_detection("prewitt")) # Perwitt button
        self.pushButton_3.clicked.connect(lambda: self.edge_detection("robert")) # Robert button
        self.pushButton_4.clicked.connect(lambda: self.edge_detection("canny")) # Canny button
        self.pushButton_5.clicked.connect(lambda: self.edge_detection("log")) # LOG button
        
        # Segmentations Buttons
        self.pushButton_11.clicked.connect(lambda: self.imageSegmentations("Histo")) # Histo button
        self.pushButton_9.clicked.connect(lambda: self.imageSegmentations("Manual")) # Manual button
        self.pushButton_8.clicked.connect(lambda: self.imageSegmentations("Adabtive")) # Adabtive button
        self.pushButton_10.clicked.connect(lambda: self.imageSegmentations("Otsu")) # Otsu button

        # Smoothing Filters
        self.pushButton_gaussian.clicked.connect(self.apply_gaussian_filter)
        self.pushButton_mean.clicked.connect(self.apply_mean_filter)
        self.pushButton_median.clicked.connect(self.apply_median_filter)
        self.pushButton_bilateral.clicked.connect(self.apply_bilateral_filter)

        self.gaussian_slider.setMinimum(3)  # Set the minimum kernel size to 3
        self.gaussian_slider.setMaximum(25)  # Set the maximum kernel size to 25
        self.gaussian_slider.setValue(5)     # Set the default kernel size to 5
        self.gaussian_slider.valueChanged.connect(self.update_filter_with_slider)  # Update filter when slider value changes

        # Create a mean filter slider (if not defined in Qt Designer)
        self.mean_slider.setMinimum(3)  # Minimum kernel size
        self.mean_slider.setMaximum(25)  # Maximum kernel size
        self.mean_slider.setValue(5)  # Default value

        # Connect slider to the update function
        self.mean_slider.valueChanged.connect(self.update_mean_filter_with_slider) # Update filter when slider value changes
        



        # Create a median filter slider (if not defined in Qt Designer)
        # self.median_slider = QSlider(Qt.Horizontal)  # Horizontal slider
        self.median_slider.setMinimum(3)  # Minimum kernel size
        self.median_slider.setMaximum(25)  # Maximum kernel size
        self.median_slider.setValue(5)  # Default value
        # self.median_slider.setTickInterval(2)  # Step interval
        # self.median_slider.setTickPosition(QSlider.TicksBelow)  # Show ticks

        # Connect slider to the update function
        self.median_slider.valueChanged.connect(self.update_median_filter_with_slider) # Update filter when slider value changes



        self.bilateral_slider.setMinimum(3)  # Minimum kernel size
        self.bilateral_slider.setMaximum(25)  # Maximum kernel size
        self.bilateral_slider.setValue(5)  # Default value

        # Connect slider to the update function
        self.bilateral_slider.valueChanged.connect(self.update_bilateral_filter_with_slider) # Update filter when slider value changes

        # Save and Reset Buttons
        self.pushButton_12.clicked.connect(self.save_image)  # Save button
        self.pushButton_7.clicked.connect(self.reset_app)   # Reset button

        '''Translation and rotation''' 
        # Rotation Slider
        self.rotationSlider.valueChanged.connect(self.apply_rotation)
        self.rotationSlider.setMinimum(-180)
        self.rotationSlider.setMaximum(180)
        self.rotationSlider.setValue(0)

        # Translation Sliders
        self.slider_x.valueChanged.connect(self.apply_translation)
        self.slider_y.valueChanged.connect(self.apply_translation)
        self.slider_x.setMinimum(-100)
        self.slider_x.setMaximum(100)
        self.slider_y.setMinimum(-100)
        self.slider_y.setMaximum(100)

        ''' Threshold Control'''
        self.thresholdValue.valueChanged.connect(self.ThethresHold)
        self.thresholdValue.setMinimum(0) #Set the minimum value to 0
        self.thresholdValue.setMaximum(255)  # Set the maximum value to 20
        self.thresholdValue.setValue(10)  # Set the initial value of the slider
        self.thresholdValue.valueChanged.connect(self.update_display_value)
        
        # Connect slider value change to update segmentation result dynamically
        self.thresholdValue.valueChanged.connect(self.update_segmentation)

        # Graphics scenes
        self.scene_1 = QGraphicsScene()
        self.scene_2 = QGraphicsScene()
        self.scene_3 = QGraphicsScene()
        
    def load_image(self):
        """ Open file dialog to select an image """
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            self.image = cv2.imread(file_name)  # Load in color
            self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)  # Convert grayscale separately
            self.rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)   # Convert image from BGR to RGB
            # Display the image in its natural colors.
            self.display_image(self.rgb_image, self.graphicsView, self.scene_1)


    def edge_detection(self, method):
        """ Apply edge detection """
        if not hasattr(self, 'image') or self.image is None:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Warning")
            msg.setText("Please load an image first!")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return  # Stop function execution

        # Perwitt Method
        elif method == "prewitt":
            # Define Prewitt kernels
            Hx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            Hy = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
            
            # Apply convolution
            edge_x = cv2.filter2D(self.gray_image, -1, Hx) / 6.0
            edge_y = cv2.filter2D(self.gray_image, -1, Hy) / 6.0
            
            # Compute gradient magnitude
            perwitt = np.sqrt(np.power(edge_x, 2) + np.power(edge_y, 2))
            
            # Normalize result to 0-255
            perwitt = (perwitt / np.max(perwitt)) * 255
            perwitt = perwitt.astype(np.uint8)

            # Save the last processed image
            self.last_processed_image = perwitt.copy()

            # Show edge detection result
            self.display_image(perwitt, self.graphicsView_2, self.scene_2)


        # Robert Method
        elif method == "robert":
            # Define Roberts kernels
            blurred = cv2.GaussianBlur(self.gray_image, (5, 5), 0)
            roberts_cross_v = np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]])
            roberts_cross_h = np.array([[0, 0, 0], [0, 0, 1], [0, -1, 0]])

            # Convert image to float for processing
            # img_float = self.gray_image.astype('float64') / 255.0
            img_float = blurred.astype('float64') / 255.0


            # Apply Roberts filter
            vertical = ndimage.convolve(img_float, roberts_cross_v)  
            horizontal = ndimage.convolve(img_float, roberts_cross_h)

            # Compute gradient magnitude
            robert = np.sqrt(np.power(vertical, 2) + np.power(horizontal, 2))

            # Normalize result to 0-255
            robert = (robert / np.max(robert)) * 255
            robert = robert.astype(np.uint8)

            # Save the last processed image
            self.last_processed_image = robert.copy()

            # Show edge detection result
            self.display_image(robert, self.graphicsView_2, self.scene_2)



        # Canny Method 
        elif method == "canny":
            # Use the grayscale image
            gray_img = self.gray_image  

            # Apply Gaussian Blur to reduce noise
            blurred = cv2.GaussianBlur(gray_img, (3, 3), 1.2)

            # Apply Canny Edge Detection with appropriate thresholds
            canny = cv2.Canny(blurred, threshold1=50, threshold2=150)

            # Save the last processed image
            self.last_processed_image = canny.copy()

            
            # Display the result in the second graphics view    
            self.display_image(canny, self.graphicsView_2, self.scene_2)

        # LOG Method 
        elif method == "log":
            # Apply Gaussian Blur to reduce noise
            blurred = cv2.GaussianBlur(self.gray_image, (3, 3), 0)

            # Apply Laplacian filter to detect edges
            log = cv2.Laplacian(blurred, cv2.CV_64F)

            # Take absolute values and normalize to 0-255
            log = np.absolute(log)
            # log = (log / log.max()) * 255
            log = cv2.normalize(log, None, 0, 255, cv2.NORM_MINMAX)  # Better normalization
            log = log.astype(np.uint8)

            # Save the last processed image
            self.last_processed_image = log.copy()

            # Display the result in the GUI
            self.display_image(log, self.graphicsView_2, self.scene_2)


    #Yassin Part: Image Segmentaion

    def ThethresHold(self):
        self.slider_value = self.thresholdValue.value()

    def update_display_value(self):
        # Get the current value of the slider
        slider_value = self.thresholdValue.value()

        # Display the slider value on the QLabel
        self.label_7.setText(f"Slider Value: {slider_value}")

    def imageSegmentations(self, method):
        """This function is triggered when a segmentation button is clicked"""
        if not hasattr(self, 'image') or self.image is None:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Warning")
            msg.setText("Please load an image first!")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return  # Stop function execution

        self.selected_method = method  # Store the selected method
        self.update_segmentation()  # Immediately apply segmentation

    def update_segmentation(self):
        """This function updates the segmentation result when the slider is moved"""
        if not hasattr(self, 'selected_method') or self.selected_method is None:
            return  # Do nothing if no segmentation method is selected

        self.slider_value = self.thresholdValue.value()  # Update slider value

        if self.selected_method == "Histo":
            # Compute histogram
            hist = cv2.calcHist([self.gray_image], [0], None, [256], [0, 256]).ravel()
            
            # Find the peak intensity
            peak_intensity = np.argmax(hist)

            # Define lower and upper threshold bounds
            lower = max(0, peak_intensity - self.slider_value)
            upper = min(255, peak_intensity + self.slider_value)

            # Convert lower and upper bounds to numpy arrays
            lower = np.array([lower], dtype=np.uint8)
            upper = np.array([upper], dtype=np.uint8)

            # Create segmentation mask
            mask = cv2.inRange(self.gray_image, lower, upper)

            # Apply mask to extract the segmented region
            segmented_image = cv2.bitwise_and(self.gray_image, self.gray_image, mask=mask)

            # Save the last processed image
            self.last_processed_image = segmented_image.copy()

        elif self.selected_method == "Manual":
            _, binary_mask = cv2.threshold(self.gray_image, self.slider_value, 255, cv2.THRESH_BINARY)
            segmented_image = cv2.bitwise_and(self.image, self.image, mask=binary_mask)

            # Save the last processed image
            self.last_processed_image = segmented_image.copy()

        elif self.selected_method == "Adabtive":
            segmented_image = cv2.adaptiveThreshold(self.gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, self.slider_value)

            # Save the last processed image
            self.last_processed_image = segmented_image.copy()

        elif self.selected_method == "Otsu":
            blurred = cv2.GaussianBlur(self.gray_image, (5, 5), 0)
            _, segmented_image = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Save the last processed image
            self.last_processed_image = segmented_image.copy()

        # Display the segmented image for all methods
        self.display_image(segmented_image, self.graphicsView_2, self.scene_2)


    def apply_gaussian_filter(self):
        if not hasattr(self, 'image') or self.image is None:
            QMessageBox.warning(None, "Warning", "Please load an image first!")
            return  # Stop execution if no image is available

        # Use the last processed image if available; otherwise, use the original image
        if hasattr(self, 'last_processed_image') and self.last_processed_image is not None:
            image_to_filter = self.last_processed_image.copy()
        else:
            image_to_filter = self.image.copy()

        # Apply Gaussian Blur with Kernel Size = 3
        ksize = 3
        filtered_image = cv2.GaussianBlur(image_to_filter, (ksize, ksize), 0)

        # Convert to RGB for displaying (only for UI, not for processing)
        display_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB)

        # Display the filtered image
        self.display_image(display_image, self.graphicsView_2, self.scene_2)

        # Store the filtered image in BGR format for further processing
        self.last_processed_image = filtered_image.copy()
        self.base_filtered_image = filtered_image.copy()


    def update_filter_with_slider(self):
        if not hasattr(self, 'image') or self.image is None:
            return  # No image to update

        # Use the stored base filtered image if available; otherwise, use the original image
        if hasattr(self, 'base_filtered_image') and self.base_filtered_image is not None:
            image_to_filter = self.base_filtered_image.copy()
        else:
            image_to_filter = self.image.copy()

        # Get the kernel size from the slider
        ksize = self.gaussian_slider.value()

        # Ensure the kernel size is odd
        if ksize % 2 == 0:
            ksize += 1

        # Apply Gaussian Blur with the selected kernel size
        updated_image = cv2.GaussianBlur(image_to_filter, (ksize, ksize), 0)

        # Convert to RGB for displaying (only for UI, not for processing)
        display_image = cv2.cvtColor(updated_image, cv2.COLOR_BGR2RGB)

        # Display the updated image
        self.display_image(display_image, self.graphicsView_2, self.scene_2)

        # Store the updated image in BGR format
        self.last_processed_image = updated_image.copy()


    def apply_mean_filter(self):
        if not hasattr(self, 'image') or self.image is None:
            QMessageBox.warning(None, "Warning", "Please load an image first!")
            return  # Stop execution if no image is available

        # Use the last processed image if available; otherwise, use the original image
        if hasattr(self, 'last_processed_image') and self.last_processed_image is not None:
            image_to_filter = self.last_processed_image.copy()
        else:
            image_to_filter = self.image.copy()

        # Apply Mean filter (blur) with Kernel Size = 5
        ksize = 5
        filtered_image = cv2.blur(image_to_filter, (ksize, ksize))

        # Convert to RGB for displaying (only for UI, not for processing)
        display_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB)

        # Display the filtered image
        self.display_image(display_image, self.graphicsView_2, self.scene_2)

        # Store the filtered image in BGR format for further processing
        self.last_processed_image = filtered_image.copy()
        self.base_filtered_image = filtered_image.copy()


    def update_mean_filter_with_slider(self):
        if not hasattr(self, 'image') or self.image is None:
            return  # No image to update

        # Use the stored base filtered image if available; otherwise, use the original image
        if hasattr(self, 'base_filtered_image') and self.base_filtered_image is not None:
            image_to_filter = self.base_filtered_image.copy()
        else:
            image_to_filter = self.image.copy()

        # Get the kernel size from the slider
        ksize = self.mean_slider.value()

        # Ensure kernel size is odd (required for mean filter)
        if ksize % 2 == 0:
            ksize += 1

        # Apply Mean filter (blur) with the selected kernel size
        updated_image = cv2.blur(image_to_filter, (ksize, ksize))

        # Convert to RGB for displaying (only for UI, not for processing)
        display_image = cv2.cvtColor(updated_image, cv2.COLOR_BGR2RGB)

        # Display the updated image
        self.display_image(display_image, self.graphicsView_2, self.scene_2)

        # Store the updated image in BGR format
        self.last_processed_image = updated_image.copy()


    def apply_median_filter(self):
        if not hasattr(self, 'image') or self.image is None:
            QMessageBox.warning(None, "Warning", "Please load an image first!")
            return  # Stop execution if no image is available

        # Use the last processed image if available; otherwise, use the original image
        if hasattr(self, 'last_processed_image') and self.last_processed_image is not None:
            image_to_filter = self.last_processed_image.copy()
        else:
            image_to_filter = self.image.copy()

        # Apply Median Filter with Kernel Size = 5
        ksize = 5
        filtered_image = cv2.medianBlur(image_to_filter, ksize)

        # Convert to RGB for displaying (only for UI, not for processing)
        display_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB)

        # Display the filtered image
        self.display_image(display_image, self.graphicsView_2, self.scene_2)

        # Store the filtered image in BGR format for further processing
        self.last_processed_image = filtered_image.copy()
        self.base_filtered_image = filtered_image.copy()


    def update_median_filter_with_slider(self):
        if not hasattr(self, 'image') or self.image is None:
            return  # No image to update

        # Use the stored base filtered image if available; otherwise, use the original image
        if hasattr(self, 'base_filtered_image') and self.base_filtered_image is not None:
            image_to_filter = self.base_filtered_image.copy()
        else:
            image_to_filter = self.image.copy()

        # Get the kernel size from the slider
        ksize = self.median_slider.value()

        # Ensure the kernel size is odd
        if ksize % 2 == 0:
            ksize += 1

        # Apply Median filter with the selected kernel size
        updated_image = cv2.medianBlur(image_to_filter, ksize)

        # Convert to RGB for displaying (only for UI, not for processing)
        display_image = cv2.cvtColor(updated_image, cv2.COLOR_BGR2RGB)

        # Display the updated image
        self.display_image(display_image, self.graphicsView_2, self.scene_2)

        # Store the updated image in BGR format
        self.last_processed_image = updated_image.copy()


    def apply_bilateral_filter(self):
        if not hasattr(self, 'image') or self.image is None:
            QMessageBox.warning(None, "Warning", "Please load an image first!")
            return  # Stop execution if no image is available

        # Use the last processed image if available; otherwise, use the original image
        if hasattr(self, 'last_processed_image') and self.last_processed_image is not None:
            image_to_filter = self.last_processed_image.copy()
        else:
            image_to_filter = self.image.copy()

        # Apply Bilateral Filter with default parameters
        d = 9  # Diameter of pixel neighborhood
        sigma_color = 75  # Filter sigma in color space
        sigma_space = 75  # Filter sigma in coordinate space
        filtered_image = cv2.bilateralFilter(image_to_filter, d, sigma_color, sigma_space)

        # Convert to RGB for displaying (only for UI, not for processing)
        display_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB)

        # Display the filtered image
        self.display_image(display_image, self.graphicsView_2, self.scene_2)

        # Store the filtered image in BGR format for further processing
        self.last_processed_image = filtered_image.copy()
        self.base_filtered_image = filtered_image.copy()


    def update_bilateral_filter_with_slider(self):
        if not hasattr(self, 'image') or self.image is None:
            return  # No image to update

        # Use the stored base filtered image if available; otherwise, use the original image
        if hasattr(self, 'base_filtered_image') and self.base_filtered_image is not None:
            image_to_filter = self.base_filtered_image.copy()
        else:
            image_to_filter = self.image.copy()

        # Get values from sliders
        d = self.bilateral_slider.value()  # Adjust pixel neighborhood size
        sigma_color = self.bilateral_slider.value()  # Adjust color space sigma
        sigma_space = self.bilateral_slider.value()  # Adjust coordinate space sigma

        # Ensure d is an odd number
        if d % 2 == 0:
            d += 1

        # Apply Bilateral Filter with selected parameters
        updated_image = cv2.bilateralFilter(image_to_filter, d, sigma_color, sigma_space)

        # Convert to RGB for displaying (only for UI, not for processing)
        display_image = cv2.cvtColor(updated_image, cv2.COLOR_BGR2RGB)

        # Display the updated image
        self.display_image(display_image, self.graphicsView_2, self.scene_2)

        # Store the updated image in BGR format
        self.last_processed_image = updated_image.copy()



    # Rotation and translation implementation
    def apply_rotation(self):
        if not hasattr(self, 'last_processed_image') or self.last_processed_image is None:
            return
        
        # Make sure the image has 3 channels (if black and white, convert it to RGB)
        if len(self.last_processed_image.shape) == 2:
            self.last_processed_image = cv2.cvtColor(self.last_processed_image, cv2.COLOR_GRAY2BGR)

        angle = self.rotationSlider.value()
        rows, cols, _ = self.last_processed_image.shape
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        rotated_image = cv2.warpAffine(self.last_processed_image, rotation_matrix, (cols, rows))
        
        # Display the result
        self.display_image(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB), self.graphicsView_2, self.scene_2)



    def apply_translation(self):
        if not hasattr(self, 'last_processed_image') or self.last_processed_image is None:
            return

        tx = self.slider_x.value()
        ty = self.slider_y.value()

        # Use the modified image dimensions.
        rows, cols = self.last_processed_image.shape[:2]

        # Definition of translation matrix
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])

        # Apply translation to the last processed image
        translated_image = cv2.warpAffine(self.last_processed_image, translation_matrix, (cols, rows))

        # Display the result
        self.display_image(cv2.cvtColor(translated_image, cv2.COLOR_BGR2RGB), self.graphicsView_2, self.scene_2)



    def display_image(self, img, graphics_view, scene):
        """ Convert image and display it in QGraphicsView """
        if len(img.shape) == 2:  # Grayscale image
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        q_image = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        scene.clear()
        scene.addItem(QGraphicsPixmapItem(pixmap))
        graphics_view.setScene(scene)
        graphics_view.fitInView(scene.itemsBoundingRect(), Qt.KeepAspectRatio)


    def save_image(self):
        """Save the last processed image to a file."""
        if not hasattr(self, 'last_processed_image') or self.last_processed_image is None:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("Error")
            msg.setText("No processed image to save!")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return  # Stop function execution

        # Open file dialog to choose save location
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)", options=options)

        if file_path:
            cv2.imwrite(file_path, self.last_processed_image)
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setWindowTitle("Success")
            msg.setText("Image saved successfully!")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()


    def reset_app(self):
        """ Reset the application to its initial state with confirmation """

        # Check if there is an image to reset
        if not hasattr(self, "image") or self.image is None:
            QMessageBox.warning(self, "Warning", "No image loaded! Nothing to reset.", QMessageBox.Ok)
            return  # Stop execution

        # Ask for confirmation before resetting
        reply = QMessageBox.question(
            self, 
            "Confirm Reset", 
            "Are you sure you want to reset the application?", 
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.image = None
            self.gray_image = None
            self.filtered_image = None
            self.last_processed_image = None  # Clear last processed image
            self.scene_1.clear()
            self.scene_2.clear()
            QMessageBox.information(self, "Reset", "Application reset successfully.", QMessageBox.Ok)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())