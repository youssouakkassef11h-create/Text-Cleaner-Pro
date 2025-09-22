#!/usr/bin/env python3
"""
نقطة الدخول الرئيسية لتطبيق تنظيف النصوص من الصور
"""

import argparse
import os
import sys
import tkinter as tk

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from text_cleaner_pro.cli.cli import cli
from text_cleaner_pro.gui.main_window import TextCleanerGUI


def main():
    parser = argparse.ArgumentParser(
        description="تطبيق تنظيف النصوص من الصور باستخدام الذكاء الاصطناعي",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--gui", action="store_true", help="تشغيل الواجهة الرسومية")
    parser.add_argument("--cli", action="store_true", help="تشغيل واجهة سطر الأوامر")
    parser.add_argument(
        "--version", action="version", version="Text Cleaner Pro 1.0.0"
    )

    args = parser.parse_args()

    if not args.cli and not args.gui:
        args.gui = True

    try:
        if args.gui:
            print("🚀 تشغيل الواجهة الرسومية...")
            root = tk.Tk()
            TextCleanerGUI(root)
            root.mainloop()

        elif args.cli:
            print("🚀 تشغيل واجهة سطر الأوامر...")
            cli()

    except Exception as e:
        print(f"❌ خطأ في تشغيل التطبيق: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()