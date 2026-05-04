import argparse
import os
from PIL import Image
import fitz  # PyMuPDF
import imageio
import numpy as np


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Combine pdf files into an animation.")
    args.add_argument('run_dir', nargs="?" , type=str,
                      help='Directory of the trained model run containing stats and checkpoints.')
    args.add_argument('--run_dir', type=str, dest='run_dir_flag', required=False,
                      help='Directory of the trained model run containing stats and checkpoints.')
    args.add_argument('--format', type=str, default='mp4', choices=['mp4', 'gif'],
                      help='Output format: mp4 (higher quality, smaller) or gif')
    args.add_argument('--dpi', type=int, default=200,
                      help='DPI for PDF rendering (default: 200, higher = better quality but larger files)')
    args.add_argument('--quality', type=int, default=8,
                      help='MP4 quality: 0-10 for libx264, lower is better quality (default: 8)')
    args = args.parse_args()

    # Support both positional and flag-based arguments
    run_dir = args.run_dir_flag or args.run_dir
    output_format = args.format
    dpi = args.dpi
    quality = args.quality
    
    animation_files_dir = os.path.join('runs' , run_dir, 'figures', 'animation_frames')
    output_path = os.path.join('runs', run_dir, 'figures', f'training_animation.{output_format}')
    # list all pdfs in the directory and sort them by epoch number
    pdf_files = [f for f in os.listdir(animation_files_dir) if f.endswith('.pdf')]
    pdf_files.sort(key=lambda x: int(x.split('epoch')[1].split('.pdf')[0]))
    pdf_paths = [os.path.join(animation_files_dir, f) for f in pdf_files]
    
    fps = 8  # frames per second (1 fps = 1 second per frame)
    
    print(f"Converting {len(pdf_paths)} PDFs to images for {output_format.upper()} format...")
    print(f"Using DPI: {dpi}, Quality: {quality}")
    
    # Calculate zoom factor from DPI (72 DPI = 1.0 zoom)
    zoom_factor = dpi / 72.0
    
    # Convert PDFs to images (required for both MP4 and GIF)
    images = []
    for pdf_path in pdf_paths:
        # Open PDF and convert first page to image
        pdf_document = fitz.open(pdf_path)
        page = pdf_document[0]  # First page
        # Render page to a pixmap with specified DPI
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom_factor, zoom_factor))
        # Convert to numpy array for imageio
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(np.array(img))
        pdf_document.close()
    
    print(f"Saving animation as {output_format.upper()}...")
    
    if output_format == 'mp4':
        # Save as MP4 with high quality
        imageio.mimsave(output_path, images, fps=fps, codec='libx264', quality=quality, pixelformat='yuv420p')
    else:  # gif
        # Save as GIF
        imageio.mimsave(output_path, images, fps=fps, loop=0)
    
    print(f"Animation saved to {output_path}")
    
    