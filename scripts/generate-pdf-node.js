#!/usr/bin/env node

/**
 * PDF Generator for LLM Fine-Tuning Guide
 * Uses markdown-pdf package to convert combined markdown to PDF
 */

const fs = require('fs');
const path = require('path');

// Configuration
const ROOT_DIR = path.join(__dirname, '..');
const CONTENT_DIR = path.join(ROOT_DIR, 'content');
const OUTPUT_DIR = path.join(ROOT_DIR, 'output');
const TEMP_FILE = path.join(OUTPUT_DIR, 'combined.md');
const PDF_FILE = path.join(OUTPUT_DIR, 'llm-fine-tuning-guide.pdf');

// File order
const files = [
    'content/00-preface.md',
    'content/00-neural-networks-basics/00-neural-networks-basics.md',
    'content/00-neural-networks-basics/01-llm-architectures.md',
    'content/01-foundations/01-foundations-overview.md',
    'content/02-introduction/02-introduction-overview.md',
    'content/02-introduction/00-setup.md',
    'content/02-introduction/01-llm-architecture.md',
    'content/02-introduction/02-what-is-fine-tuning.md',
    'content/02-introduction/03-workflows.md',
    'content/02-introduction/04-first-fine-tune.md',
    'content/03-hardware-setup/03-hardware-setup-guide.md',
    'content/04-data-engineering/04-data-engineering-guide.md',
    'content/05-training-dynamics/05-training-dynamics-guide.md',
    'content/06-parameter-efficiency/06-parameter-efficiency-guide.md',
    'content/07-alignment/07-alignment-guide.md',
    'content/08-evaluation/08-evaluation-guide.md',
    'content/09-model-deployment/09-model-deployment-guide.md',
    'content/10-mlops-pipelines/10-mlops-pipelines-guide.md',
    'content/11-appendices/11-appendices-guide.md',
    'content/11-appendices/02-quick-revision.md',
];

console.log('📚 LLM Fine-Tuning Guide - PDF Generator');
console.log('========================================\n');

// Create output directory
if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
}

// Combine markdown files
console.log('📝 Combining markdown files...\n');

let combined = `# LLM Fine-Tuning Guide

> A Step-by-Step Guide for Technical People

*Generated on: ${new Date().toISOString().split('T')[0]}*

---

`;

for (const file of files) {
    const fullPath = path.join(ROOT_DIR, file);
    if (fs.existsSync(fullPath)) {
        console.log(`  Adding: ${file}`);
        const content = fs.readFileSync(fullPath, 'utf8');
        combined += content + '\n\n---\n\n';
    } else {
        console.log(`  ⚠️  Not found: ${file}`);
    }
}

// Write combined markdown
fs.writeFileSync(TEMP_FILE, combined);
console.log(`\n✅ Combined markdown saved to: ${TEMP_FILE}\n`);

// Try to convert to PDF
console.log('📄 Converting to PDF...');

// Check if markdown-pdf is installed
try {
    const markdownPDF = require('markdown-pdf');

    markdownPDF({
        cssPath: path.join(__dirname, 'pdf-styles.css'),
        paper: { format: 'A4', orientation: 'portrait' },
        remarkable: { html: true }
    })
    .from(TEMP_FILE)
    .to(PDF_FILE, () => {
        console.log(`\n✅ PDF generated: ${PDF_FILE}`);
        console.log('\nDone!');
    });
} catch (error) {
    console.log('❌ markdown-pdf not installed');
    console.log('\nTo generate PDF, run:');
    console.log('  npm install markdown-pdf');
    console.log('  node scripts/generate-pdf-node.js\n');
    console.log('Or use the bash script:');
    console.log('  ./scripts/generate-pdf.sh\n');
}
