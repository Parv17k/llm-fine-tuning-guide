#!/bin/bash

# Generate PDF from all markdown files
# This script combines all .md files in order and converts to PDF

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
CONTENT_DIR="$ROOT_DIR/content"
OUTPUT_DIR="$ROOT_DIR/output"
TEMP_FILE="$OUTPUT_DIR/combined.md"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "📚 LLM Fine-Tuning Guide - PDF Generator"
echo "========================================"
echo ""

# Define the order of files
FILES=(
    "content/00-preface.md"
    "content/00-neural-networks-basics/00-neural-networks-basics.md"
    "content/00-neural-networks-basics/01-llm-architectures.md"
    "content/01-foundations/01-foundations-overview.md"
    "content/02-introduction/02-introduction-overview.md"
    "content/02-introduction/00-setup.md"
    "content/02-introduction/01-llm-architecture.md"
    "content/02-introduction/02-what-is-fine-tuning.md"
    "content/02-introduction/03-workflows.md"
    "content/02-introduction/04-first-fine-tune.md"
    "content/03-hardware-setup/03-hardware-setup-guide.md"
    "content/04-data-engineering/04-data-engineering-guide.md"
    "content/05-training-dynamics/05-training-dynamics-guide.md"
    "content/06-parameter-efficiency/06-parameter-efficiency-guide.md"
    "content/07-alignment/07-alignment-guide.md"
    "content/08-evaluation/08-evaluation-guide.md"
    "content/09-model-deployment/09-model-deployment-guide.md"
    "content/10-mlops-pipelines/10-mlops-pipelines-guide.md"
    "content/11-appendices/11-appendices-guide.md"
    "content/11-appendices/02-quick-revision.md"
)

echo "📝 Combining markdown files..."

# Clear temp file
> "$TEMP_FILE"

# Add title
cat >> "$TEMP_FILE" << 'EOF'
# LLM Fine-Tuning Guide

> A Step-by-Step Guide for Technical People

*Generated on: $(date +%Y-%m-%d)*

---

EOF

# Combine files in order
for file in "${FILES[@]}"; do
    if [ -f "$CONTENT_DIR/../$file" ]; then
        echo "  Adding: $file"
        echo "" >> "$TEMP_FILE"
        cat "$CONTENT_DIR/../$file" >> "$TEMP_FILE"
        echo "" >> "$TEMP_FILE"
        echo "---" >> "$TEMP_FILE"
        echo "" >> "$TEMP_FILE"
    else
        echo "  ⚠️  Not found: $file"
    fi
done

echo ""
echo "✅ Combined markdown saved to: $TEMP_FILE"
echo ""

# Check for conversion tools
if command -v pandoc &> /dev/null; then
    echo "📄 Converting to PDF using Pandoc..."
    pandoc "$TEMP_FILE" \
        --from markdown \
        --to pdf \
        --output "$OUTPUT_DIR/llm-fine-tuning-guide.pdf" \
        --table-of-contents \
        --metadata title="LLM Fine-Tuning Guide" \
        --metadata author="Parv Khatri" \
        --pdf-engine=xelatex \
        -V geometry:margin=1in \
        -V fontsize=11pt \
        -V documentclass=report \
        -V colorlinks=true

    echo ""
    echo "✅ PDF generated: $OUTPUT_DIR/llm-fine-tuning-guide.pdf"

elif command -v mdbook &> /dev/null; then
    echo "📄 Converting to PDF using mdbook..."
    # mdbook method would require additional setup
    echo "mdbook found but requires additional configuration"

else
    echo "⚠️  No PDF conversion tool found."
    echo ""
    echo "To generate PDF, install one of these tools:"
    echo ""
    echo "Option 1: Pandoc (recommended)"
    echo "  macOS:     brew install pandoc texlive"
    echo "  Linux:     sudo apt install pandoc texlive-latex-base texlive-xetex"
    echo "  Windows:   choco install pandoc"
    echo ""
    echo "Option 2: Using Node.js (already installed)"
    echo "  Run: npm run generate-pdf"
    echo ""
    echo "Combined markdown file is ready at:"
    echo "  $TEMP_FILE"
    echo ""
    echo "You can also:"
    echo "  - Open $TEMP_FILE in a markdown editor"
    echo "  - Print to PDF from your browser"
fi

echo ""
echo "Done!"
