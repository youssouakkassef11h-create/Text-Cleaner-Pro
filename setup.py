from setuptools import setup, find_packages

setup(
    name="text-cleaner-pro",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.5.0",
        "easyocr>=1.6.0",
        "numpy>=1.21.0",
        "Pillow>=8.0.0",
        "click>=8.0.0",
        "tqdm>=4.62.0",
        "torch>=1.9.0",
        "torchvision>=0.10.0",
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': ['text-cleaner-pro=main:main']
    },
    author="Text Cleaner Pro Team",
    author_email="info@textcleanerpro.com",
    description="أداة متقدمة لإزالة النصوص من الصور باستخدام الذكاء الاصطناعي",
    keywords="text removal, image processing, AI, computer vision",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Multimedia :: Graphics",
    ],
)