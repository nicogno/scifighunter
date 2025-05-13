````markdown
# FigureFinder

FigureFinder is a lightweight command-line tool that helps you locate figures from a collection of scientific PDFs. It extracts all images from PDFs, embeds them using CLIP (Contrastive Language–Image Pretraining), and allows you to search for relevant figures using 
natural language queries.

## Features
- 🔍 Extract all figures from PDF files using PyMuPDF
- 🧠 Encode figures with OpenAI's CLIP for semantic similarity search
- 💬 Retrieve relevant images using descriptive text prompts
- ⚡ Fast and simple CLI interface

## Installation

```bash
pip install -r requirements.txt
````

You'll need:

* `PyMuPDF`
* `torch`
* `clip-by-openai`
* `Pillow`
* `tqdm`
* `numpy`

## Usage

### 1. Extract Figures from PDFs

```bash
python figurefinder.py extract --pdf_dir ./papers --output_dir ./figures
```

This will process all PDFs in `./papers` and store extracted images in `./figures/{pdf_name}/`.

### 2. Embed Figures with CLIP

```bash
python figurefinder.py embed --figures_dir ./figures --index_file ./clip_index.pkl
```

This creates an index (`clip_index.pkl`) of image embeddings and associated file paths.

### 3. Search for Figures by Description

```bash
python figurefinder.py search --index_file ./clip_index.pkl --query "a brain scan with a large lesion" --top_k 5
```

This will return the top 5 most semantically similar images to your query.

## Output Structure

```
figures/
├── Paper1/
│   ├── page_1_img_1.png
│   └── page_3_img_2.jpg
├── Paper2/
│   └── page_2_img_1.jpeg

clip_index.pkl  # contains CLIP image embeddings and paths
```

## Roadmap Ideas

* 📋 Extract figure captions and associate them with images
* 🌐 Add a Streamlit or Flask interface
* 📎 Support reverse image search

## License

MIT

## Acknowledgments

* [OpenAI CLIP](https://github.com/openai/CLIP)
* [PyMuPDF](https://github.com/pymupdf/PyMuPDF)
* [Pillow](https://python-pillow.org)

```
