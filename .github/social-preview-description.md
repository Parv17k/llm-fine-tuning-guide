# Social Preview Image Instructions

The social preview image (`.github/social-preview.svg`) is used when the repository is shared on Twitter, LinkedIn, and other platforms.

## To upload the social preview image:

1. Go to your repository on GitHub: https://github.com/Parv17k/llm-fine-tuning-guide
2. Click **Settings** tab
3. Scroll down to **Social preview** section
4. Click **Upload image**
5. Use the SVG file at `.github/social-preview.svg` or convert to PNG:
   - Open the SVG in a browser
   - Take a screenshot or export as PNG (1200x630 pixels)
6. Upload the PNG file

## Image specifications:
- **Size:** 1200 x 630 pixels (GitHub's recommended size)
- **Format:** PNG or JPG
- **File size:** Under 1MB

## Alternative: Generate PNG programmatically

If you have Python installed, you can convert the SVG to PNG:

```bash
pip install cairosvg
cairosvg .github/social-preview.svg -o .github/social-preview.png -w 1200 -h 630
```

Then upload `.github/social-preview.png` to GitHub Settings.
