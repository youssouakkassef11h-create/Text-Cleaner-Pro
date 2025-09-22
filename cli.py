#!/usr/bin/env python3
"""
واجهة سطر الأوامر لتطبيق تنظيف النصوص من الصور
"""

import os
import sys
import click
from pathlib import Path
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.text_detector import TextDetector
from src.core.mask_generator import MaskGenerator
from src.core.image_cleaner import ImageCleaner

@click.group()
@click.version_option(version='1.0.0', prog_name='Text Cleaner Pro')
def cli():
    """Text Cleaner Pro - أداة لإزالة النصوص من الصور باستخدام الذكاء الاصطناعي"""
    pass

@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='مسار حفظ الصورة المنظفة')
@click.option('--languages', '-l', default='en,ar', help='لغات النصوص المراد اكتشافها (مفصولة بفواصل)')
@click.option('--confidence', '-c', default=0.5, type=float, help='حد الثقة لاكتشاف النصوص (0-1)')
@click.option('--method', '-m', default='inpaint', type=click.Choice(['inpaint', 'white']), 
              help='طريقة التنظيف: inpaint (ذكي) أو white (تبييض)')
def clean(input_path, output, languages, confidence, method):
    """تنظيف صورة واحدة من النصوص"""
    try:
        input_path = Path(input_path)
        if not output:
            output = input_path.parent / f"{input_path.stem}_cleaned{input_path.suffix}"
        
        click.echo(f"🔍 معالجة الصورة: {input_path}")
        
        languages_list = [lang.strip() for lang in languages.split(',')]
        detector = TextDetector(languages=languages_list)
        mask_generator = MaskGenerator()
        cleaner = ImageCleaner(method=method)
        
        detections = detector.detect_text(str(input_path), confidence)
        if not detections:
            click.echo("⚠️  لم يتم العثور على نصوص في الصورة")
            return
        
        click.echo(f"✅ تم العثور على {len(detections)} نص")
        
        image = cv2.imread(str(input_path))
        mask = mask_generator.create_mask(image.shape, detections)
        cleaned_image = cleaner.clean_image(image, mask)
        cv2.imwrite(str(output), cleaned_image)
        click.echo(f"✅ تم حفظ الصورة المنظفة في: {output}")
        
    except Exception as e:
        click.echo(f"❌ خطأ: {e}")

@cli.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.option('--output-dir', '-o', type=click.Path(), help='مجلد حفظ الصور المنظفة')
@click.option('--languages', '-l', default='en,ar', help='لغات النصوص المراد اكتشافها (مفصولة بفواصل)')
@click.option('--confidence', '-c', default=0.5, type=float, help='حد الثقة لاكتشاف النصوص (0-1)')
@click.option('--method', '-m', default='inpaint', type=click.Choice(['inpaint', 'white']), 
              help='طريقة التنظيف: inpaint (ذكي) أو white (تبييض)')
def batch_clean(input_dir, output_dir, languages, confidence, method):
    """تنظيف جميع الصور في مجلد من النصوص"""
    try:
        input_dir = Path(input_dir)
        
        if not output_dir:
            output_dir = input_dir.parent / f"{input_dir.name}_cleaned"
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in supported_formats]
        
        if not image_files:
            click.echo("❌ لم يتم العثور على صور مدعومة في المجلد")
            return
            
        click.echo(f"🔍 معالجة {len(image_files)} صورة من: {input_dir}")
        
        languages_list = [lang.strip() for lang in languages.split(',')]
        detector = TextDetector(languages=languages_list)
        mask_generator = MaskGenerator()
        cleaner = ImageCleaner(method=method)
        
        success_count = 0
        for img_path in image_files:
            try:
                output_path = output_dir / f"{img_path.stem}_cleaned{img_path.suffix}"
                
                detections = detector.detect_text(str(img_path), confidence)
                if not detections:
                    click.echo(f"⚠️  لم يتم العثور على نصوص في: {img_path.name}")
                    continue
                
                image = cv2.imread(str(img_path))
                mask = mask_generator.create_mask(image.shape, detections)
                cleaned_image = cleaner.clean_image(image, mask)
                cv2.imwrite(str(output_path), cleaned_image)
                
                click.echo(f"✅ تم تنظيف: {img_path.name} → {len(detections)} نص")
                success_count += 1
                
            except Exception as e:
                click.echo(f"❌ خطأ في معالجة {img_path.name}: {e}")
        
        click.echo(f"🎉 تم الانتهاء من معالجة {success_count} من أصل {len(image_files)} صورة")
        
    except Exception as e:
        click.echo(f"❌ خطأ: {e}")

if __name__ == '__main__':
    cli()