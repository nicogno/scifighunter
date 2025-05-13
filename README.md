# SciFigHunter

SciFigHunter is a lightweight command-line tool that helps you locate figures from a collection of scientific PDFs. It extracts all images and their potential captions from PDFs, embeds them using CLIP (Contrastive Language–Image Pretraining), and allows you to search for relevant figures using natural language queries.

## Features
- 🔍 Extract all figures and attempt to find corresponding captions from PDF files using PyMuPDF.
- 🧠 Encode figures and their captions with OpenAI's CLIP, with **configurable weighting** for image vs. text features in the combined embedding.
- 💬 Retrieve relevant images using descriptive text prompts.
- ⚡ Fast and simple CLI interface.

## Installation

```bash
pip install -r requirements.txt
```

You'll need (typically managed by `requirements.txt`):

* `PyMuPDF`
* `torch`
* `openai-clip` (or the specific CLIP package you are using, e.g., `clip-by-openai`)
* `Pillow`
* `tqdm`
* `numpy`

## Usage

### 1. Extract Figures and Captions from PDFs

```bash
python scifighunter.py extract --pdf_dir ./papers --output_dir ./figures
```

This will process all PDFs in `./papers` and store extracted images and a `captions.json` file (containing image paths and their extracted captions) in subdirectories like `./figures/{pdf_name}/`.

### 2. Embed Figures with CLIP

```bash
python scifighunter.py embed --figures_dir ./figures --index_file ./clip_index.pkl --alpha 0.7
```

This creates an index (`clip_index.pkl`) of combined image and caption embeddings.
- The `--figures_dir` should point to the directory created by the `extract` command.
- The `--alpha` parameter (optional, default is 0.5) controls the weighting between image and text features. For example, `--alpha 0.7` means the combined embedding will be 70% from the image and 30% from the caption. An alpha of 1.0 would use only image features, and 0.0 would use only text features.

### 3. Search for Figures by Description

```bash
python scifighunter.py search --index_file ./clip_index.pkl --query "a brain scan with a large lesion" --top_k 5
```

This will return the top 5 most semantically similar images to your query based on the combined embeddings.

## Output Structure

```
figures/
├── Paper1/
│   ├── page_1_img_1.png
│   ├── page_3_img_2.jpg
│   └── captions.json  # Contains paths and extracted captions for images in Paper1
├── Paper2/
│   ├── page_2_img_1.jpeg
│   └── captions.json  # Contains paths and extracted captions for images in Paper2
...
clip_index.pkl  # Contains combined CLIP image+caption embeddings and paths
```

## Roadmap Ideas

* 🌐 Add a Streamlit or Flask interface
* 📎 Support reverse image search
* 🧐 Improve caption extraction heuristics

## License

MIT

## Acknowledgments

* [OpenAI CLIP](https://github.com/openai/CLIP)
* [PyMuPDF](https://github.com/pymupdf/PyMuPDF)
* [Pillow](https://python-pillow.org)
