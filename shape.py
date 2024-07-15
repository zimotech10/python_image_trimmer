import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw, ImageTk
import numpy as np
import cv2
import matplotlib.pyplot as plt

class ImageCropper:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Cropper")

        self.canvas = tk.Canvas(root, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.start_button = tk.Button(root, text="Start", command=self.start_processing)
        self.start_button.pack(side=tk.BOTTOM)

        self.progress_canvas = tk.Canvas(root, width=200, height=200)
        self.progress_canvas.pack(side=tk.BOTTOM)

        self.shape_images = []
        self.source_images = []
        self.results_folder = "results"

        self.progress = 0

    def start_processing(self):
        shape_folder = "./masks"
        source_folder = "./images"
        self.results_folder = "./results"
        # shape_folder = filedialog.askdirectory(title="Select Shape Images Folder")
        # source_folder = filedialog.askdirectory(title="Select Source Images Folder")
        # self.results_folder = filedialog.askdirectory(title="Select Results Folder")

        if not shape_folder or not source_folder or not self.results_folder:
            messagebox.showerror("Error", "All folders must be selected")
            return

        self.load_shape_images(shape_folder)
        self.load_source_images(source_folder)

        total_images = len(self.source_images)
        shape_images_count = len(self.shape_images)

        if total_images == 0 or shape_images_count == 0:
            messagebox.showerror("Error", "No images found in the selected folders")
            return

        self.progress = 0
        self.update_progress()

        for i, source_image in enumerate(self.source_images):
            shape_image = self.shape_images[i % shape_images_count]
            shape_profile = self.generate_shape_profile(np.array(shape_image))
            output_image = self.crop_image_with_shape_profile(source_image, shape_profile)
            output_image = self.remove_transparent_areas(output_image)
            output_image.save(os.path.join(self.results_folder, f"{i+1}.png"))

            self.progress = (i + 1) / total_images
            self.update_progress()

        messagebox.showinfo("Success", "Processing completed successfully!")

    def load_shape_images(self, shape_folder):
        self.shape_images = [Image.open(os.path.join(shape_folder, f"{i+1}.png")).convert("L") for i in range(11)]

    def load_source_images(self, source_folder):
        self.source_images = [Image.open(os.path.join(source_folder, f"{i+1}.png")).convert("RGBA") for i in range(110)]

    def generate_shape_profile(self, shape_image_array):
        contours, _ = cv2.findContours(shape_image_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        shape_profile = [(point[0][0], point[0][1]) for point in contour]
        return shape_profile

    def crop_image_with_shape_profile(self, source_image, shape_profile):
        mask = Image.new("L", source_image.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.polygon(shape_profile, outline=1, fill=255)

        masked_image = Image.new("RGBA", source_image.size)
        masked_image.paste(source_image, (0, 0), mask)
        return masked_image

    def remove_transparent_areas(self, image):
        image_array = np.array(image)
        non_transparent_points = np.where(image_array[:, :, 3] != 0)

        if non_transparent_points[0].size == 0 or non_transparent_points[1].size == 0:
            return image

        top_left_y = np.min(non_transparent_points[0])
        top_left_x = np.min(non_transparent_points[1])
        bottom_right_y = np.max(non_transparent_points[0])
        bottom_right_x = np.max(non_transparent_points[1])

        cropped_image = image.crop((top_left_x, top_left_y, bottom_right_x + 1, bottom_right_y + 1))
        return cropped_image

    def update_progress(self):
        self.progress_canvas.delete("all")
        fig, ax = plt.subplots(figsize=(2, 2), subplot_kw=dict(aspect="equal"))
        wedges, _ = ax.pie([self.progress, 1 - self.progress], colors=["lightblue", "white"])
        ax.text(0, 0, f"{int(self.progress * 100)}%", ha='center', va='center', fontsize=24)
        self.progress_canvas.create_image(0, 0, anchor=tk.NW, image=self._plot_to_image(fig))
        plt.close(fig)

    def _plot_to_image(self, fig):
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
        img = Image.fromarray(img)
        return ImageTk.PhotoImage(img)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCropper(root)
    root.mainloop()
