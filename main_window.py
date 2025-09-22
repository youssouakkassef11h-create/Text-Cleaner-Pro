import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import os
from pathlib import Path

from src.core.text_detector import TextDetector
from src.core.mask_generator import MaskGenerator
from src.core.image_cleaner import ImageCleaner

class TextCleanerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Text Cleaner Pro")
        self.root.geometry("1000x700")
        self.root.resizable(True, True)
        
        # المتغيرات
        self.image_path = None
        self.original_image = None
        self.cleaned_image = None
        self.detections = []
        
        self.setup_ui()
        
    def setup_ui(self):
        # إنشاء الإطارات الرئيسية
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # إطار التحكم
        control_frame = ttk.LabelFrame(main_frame, text="التحكم", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # أزرار التحكم
        ttk.Button(control_frame, text="تحميل صورة", command=self.load_image).grid(row=0, column=0, padx=5)
        ttk.Button(control_frame, text="اكتشاف النصوص", command=self.detect_text).grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="تنظيف الصورة", command=self.clean_image).grid(row=0, column=2, padx=5)
        ttk.Button(control_frame, text="حفظ النتيجة", command=self.save_image).grid(row=0, column=3, padx=5)
        
        # إطار الإعدادات
        settings_frame = ttk.LabelFrame(main_frame, text="الإعدادات", padding="10")
        settings_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # خيارات اللغة
        ttk.Label(settings_frame, text="اللغات:").grid(row=0, column=0, sticky=tk.W)
        self.languages_var = tk.StringVar(value="en,ar")
        ttk.Entry(settings_frame, textvariable=self.languages_var, width=15).grid(row=0, column=1, padx=5)
        
        # عتبة الثقة
        ttk.Label(settings_frame, text="عتبة الثقة:").grid(row=0, column=2, sticky=tk.W, padx=(20, 0))
        self.confidence_var = tk.DoubleVar(value=0.5)
        ttk.Scale(settings_frame, from_=0.1, to=1.0, variable=self.confidence_var, 
                 orient=tk.HORIZONTAL, length=100).grid(row=0, column=3, padx=5)
        ttk.Label(settings_frame, textvariable=self.confidence_var).grid(row=0, column=4)
        
        # طريقة التنظيف
        ttk.Label(settings_frame, text="طريقة التنظيف:").grid(row=0, column=5, sticky=tk.W, padx=(20, 0))
        self.method_var = tk.StringVar(value="inpaint")
        method_combo = ttk.Combobox(settings_frame, textvariable=self.method_var, 
                                   values=["inpaint", "white"], width=10, state="readonly")
        method_combo.grid(row=0, column=6, padx=5)
        
        # إطار الصور
        images_frame = ttk.Frame(main_frame)
        images_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # الصورة الأصلية
        original_frame = ttk.LabelFrame(images_frame, text="الصورة الأصلية")
        original_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        self.original_canvas = tk.Canvas(original_frame, width=400, height=400, bg="white")
        self.original_canvas.pack()
        
        # الصورة النظيفة
        cleaned_frame = ttk.LabelFrame(images_frame, text="الصورة النظيفة")
        cleaned_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        
        self.cleaned_canvas = tk.Canvas(cleaned_frame, width=400, height=400, bg="white")
        self.cleaned_canvas.pack()
        
        # إطار المعلومات
        info_frame = ttk.LabelFrame(main_frame, text="المعلومات", padding="10")
        info_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.info_text = tk.Text(info_frame, height=5, width=80)
        self.info_text.grid(row=0, column=0)
        
        # تكبير العناصر مع النافذة
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        images_frame.columnconfigure(0, weight=1)
        images_frame.columnconfigure(1, weight=1)
        images_frame.rowconfigure(0, weight=1)
        
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if file_path:
            self.image_path = file_path
            self.original_image = cv2.imread(file_path)
            if self.original_image is not None:
                self.display_image(self.original_image, self.original_canvas)
                self.add_info(f"تم تحميل الصورة: {os.path.basename(file_path)}")
            else:
                messagebox.showerror("خطأ", "تعذر تحميل الصورة")
    
    def detect_text(self):
        if self.original_image is None:
            messagebox.showwarning("تحذير", "يرجى تحميل صورة أولاً")
            return
            
        try:
            languages = [lang.strip() for lang in self.languages_var.get().split(',')]
            detector = TextDetector(languages=languages)
            self.detections = detector.detect_text(self.image_path, self.confidence_var.get())
            
            # رسم المربعات حول النصوص المكتشفة
            image_with_boxes = self.original_image.copy()
            for bbox, text, conf in self.detections:
                pts = np.array(bbox, np.int32).reshape((-1,1,2))
                cv2.polylines(image_with_boxes, [pts], True, (0, 255, 0), 2)
                cv2.putText(image_with_boxes, f"{text} ({conf:.2f})", 
                           (int(bbox[0][0]), int(bbox[0][1])-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            self.display_image(image_with_boxes, self.original_canvas)
            self.add_info(f"تم اكتشاف {len(self.detections)} نص في الصورة")
            
        except Exception as e:
            messagebox.showerror("خطأ", f"فشل في اكتشاف النصوص: {str(e)}")
    
    def clean_image(self):
        if self.original_image is None:
            messagebox.showwarning("تحذير", "يرجى تحميل صورة أولاً")
            return
            
        if not self.detections:
            messagebox.showwarning("تحذير", "لم يتم اكتشاف نصوص. يرجى اكتشاف النصوص أولاً")
            return
            
        try:
            mask_generator = MaskGenerator()
            cleaner = ImageCleaner(method=self.method_var.get())
            
            mask = mask_generator.create_mask(self.original_image.shape, self.detections)
            self.cleaned_image = cleaner.clean_image(self.original_image, mask)
            
            self.display_image(self.cleaned_image, self.cleaned_canvas)
            self.add_info("تم تنظيف الصورة بنجاح")
            
        except Exception as e:
            messagebox.showerror("خطأ", f"فشل في تنظيف الصورة: {str(e)}")
    
    def save_image(self):
        if self.cleaned_image is None:
            messagebox.showwarning("تحذير", "لا توجد صورة نظيفة للحفظ")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("All files", "*.*")]
        )
        if file_path:
            cv2.imwrite(file_path, self.cleaned_image)
            self.add_info(f"تم حفظ الصورة النظيفة في: {file_path}")
    
    def display_image(self, image, canvas):
        # تحويل الصورة من BGR إلى RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # تغيير حجم الصورة لتناسب Canvas مع الحفاظ على النسبة
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            img_ratio = pil_image.width / pil_image.height
            canvas_ratio = canvas_width / canvas_height
            
            if img_ratio > canvas_ratio:
                new_width = canvas_width
                new_height = int(canvas_width / img_ratio)
            else:
                new_height = canvas_height
                new_width = int(canvas_height * img_ratio)
                
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        photo = ImageTk.PhotoImage(pil_image)
        canvas.delete("all")
        canvas.create_image(canvas_width//2, canvas_height//2, anchor=tk.CENTER, image=photo)
        canvas.image = photo  # حفظ المرجع لمنع جمع القمامة
    
    def add_info(self, message):
        self.info_text.insert(tk.END, message + "\n")
        self.info_text.see(tk.END)