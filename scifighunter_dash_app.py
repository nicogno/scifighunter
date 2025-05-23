import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import base64
import os
import io
import json
from pathlib import Path
import time
import shutil
from scifighunter_core import extract_from_pdf, extract_from_directory, build_image_embeddings, search_images, reindex_database

# Define app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "SciFigHunter - Scientific Figure Search"

# Define layout
app.layout = html.Div([ # Outer container for overall padding
    html.H1("SciFigHunter - Scientific Figure Search", style={'textAlign': 'center', 'marginBottom': '30px', 'marginTop': '10px'}),
    
    dbc.Row([
        dbc.Col([
            html.H2("1. Upload PDFs", style={'marginTop': '20px', 'marginBottom': '15px'}),
            dcc.Upload(
                id='upload-pdfs',
                children=html.Div([
                    html.I(className="fas fa-cloud-upload-alt"),
                    ' Drag and Drop or Select PDF Files'
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                multiple=True
            ),
            
            html.Div([
                dbc.Checkbox(
                    id="use-classifier",
                    label="Use Image Classification Filter",
                    value=True
                ),
                dbc.Tooltip(
                    "When enabled, uses AI to filter out logos, simple shapes, and non-scientific images during extraction",
                    target="use-classifier",
                ),
            ], style={'margin': '10px 0'}),
            
            html.Button('Embed New PDFs', id='embed-button', className='btn btn-primary'),
            
            html.Button('Re-index All PDFs', id='reindex-button', className='btn btn-warning', style={'margin-left': '10px'}),
            
            dcc.Loading( # Wrap processing-output with dcc.Loading
                id="loading-processing",
                type="default",
                children=html.Div(id='processing-output', style={
                    'maxHeight': '300px',
                    'overflowY': 'auto',
                    'border': '1px solid #ddd',
                    'borderRadius': '5px',
                    'padding': '10px',
                    'marginTop': '20px', # Increased margin-top
                    'backgroundColor': '#f9f9f9'
                })
            ),
        ], width=6),
        
        dbc.Col([
            html.H2("2. Search Figures", style={'marginTop': '20px', 'marginBottom': '15px'}),
            dcc.Input(
                id='search-input',
                type='text',
                placeholder='Enter search query (e.g., a cat sitting on a car)',
                style={'width': '100%', 'padding': '10px'}
            ),
            
            html.Div([
                dbc.Checkbox(
                    id="use-post-filtering",
                    label="Use Enhanced Post-Retrieval Filtering",
                    value=True
                ),
                dbc.Tooltip(
                    "When enabled, applies additional filtering to search results to improve relevance and remove duplicates",
                    target="use-post-filtering",
                ),
            ], style={'margin': '10px 0'}),
            
            html.Div([
                dbc.Checkbox(
                    id="use-aggressive-filtering",
                    label="Use Aggressive Filtering",
                    value=True
                ),
                dbc.Tooltip(
                    "When enabled, completely removes problematic images like circles and logos from search results",
                    target="use-aggressive-filtering",
                ),
            ], style={'margin': '10px 0'}),
            
            html.Div([
                html.Label("Minimum Relevance Score:"),
                dcc.Slider(
                    id='min-score-slider',
                    min=0,
                    max=0.2,
                    step=0.01,
                    value=0.05,
                    marks={i/100: f'{i/100:.2f}' for i in range(0, 21, 5)},
                ),
            ], style={'margin': '10px 0'}),
            
            html.Button('Search', id='search-button', className='btn btn-success'),
        ], width=6),
    ], style={'marginBottom': '20px'}), # Added margin to the bottom of the main row
    
    html.Hr(style={'marginTop': '30px', 'marginBottom': '30px'}), # Added margins to Hr
    
    html.H2("3. Search Results", style={'marginTop': '30px', 'marginBottom': '15px'}),
    html.Div(id='search-results'),
    
    # Store for uploaded files
    dcc.Store(id='uploaded-files'),
    
    # Store for extracted metadata
    dcc.Store(id='extracted-metadata'),
], style={'padding': '20px'}) # Added padding to the outermost Div

@app.callback(
    [Output('uploaded-files', 'data'),
     Output('processing-output', 'children', allow_duplicate=True)],
    Input('upload-pdfs', 'contents'),
    State('upload-pdfs', 'filename'),
    prevent_initial_call=True
)
def store_uploaded_files(contents, filenames):
    if not contents:
        return [], dash.no_update # No change to processing-output if no content
    
    uploaded_files_paths = []
    valid_pdf_messages = []
    
    temp_dir = Path('temp_pdfs')
    temp_dir.mkdir(exist_ok=True)

    for content, filename in zip(contents, filenames):
        if not filename.lower().endswith('.pdf'):
            valid_pdf_messages.append(html.Li(f"{filename} - Skipped (not a PDF)", style={'color': 'orange'}))
            continue
        
        try:
            content_type, content_string = content.split(',')
            decoded = base64.b64decode(content_string)
            file_path = temp_dir / filename
            
            with open(file_path, 'wb') as f:
                f.write(decoded)
            
            uploaded_files_paths.append(str(file_path))
            valid_pdf_messages.append(html.Li(f"{filename} - Ready for processing"))
        except Exception as e:
            valid_pdf_messages.append(html.Li(f"Error uploading {filename}: {str(e)}", style={'color': 'red'}))

    if not uploaded_files_paths:
        output_div_children = [html.P("No valid PDF files were successfully uploaded.")]
        if valid_pdf_messages: # Show skipped/error messages if any
             output_div_children.extend(valid_pdf_messages)
        return [], html.Div(output_div_children)

    processing_messages = [html.H5("Selected Files:")]
    processing_messages.extend(valid_pdf_messages)
    if uploaded_files_paths:
        processing_messages.append(html.P("Click 'Embed New PDFs' to process the successfully uploaded files.", style={'marginTop': '10px'}))
    
    return uploaded_files_paths, html.Div(processing_messages)

@app.callback(
    [Output('processing-output', 'children'),
     Output('extracted-metadata', 'data')],
    [Input('embed-button', 'n_clicks'),
     Input('reindex-button', 'n_clicks')],
    [State('uploaded-files', 'data'),
     State('use-classifier', 'value')],
    prevent_initial_call=True
)
def process_pdfs(embed_clicks, reindex_clicks, uploaded_files, use_classifier):
    ctx = callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    if not ctx.triggered:
        return html.Div("Upload PDFs and click 'Embed New PDFs' to process."), []
    
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    if button_id == 'reindex-button':
        # Re-index all PDFs
        pdf_dir = Path('temp_pdfs')
        if not pdf_dir.exists() or not list(pdf_dir.glob('*.pdf')):
            return html.Div("No PDFs found to re-index. Please upload PDFs first."), []
        
        processing_output = [html.Div(f"Re-indexing all PDFs in {pdf_dir} with classifier {'enabled' if use_classifier else 'disabled'}...")]
        
        # Re-index the database
        success = reindex_database(pdf_dir, output_dir, use_classifier=use_classifier)
        
        if success:
            processing_output.append(html.Div("Database re-indexing completed successfully.", style={'color': 'green'}))
        else:
            processing_output.append(html.Div("Database re-indexing failed.", style={'color': 'red'}))
        
        return html.Div(processing_output), []
    
    elif button_id == 'embed-button':
        # Process new PDFs
        if not uploaded_files:
            return html.Div("No PDFs uploaded. Please upload PDFs first."), []
        
        processing_output = []
        all_metadata = []
        
        for pdf_path in uploaded_files:
            processing_output.append(html.Div(f"Processing: {os.path.basename(pdf_path)}..."))
            
            try:
                # Extract images from PDF
                metadata = extract_from_pdf(pdf_path, output_dir, use_classifier=use_classifier)
                all_metadata.extend(metadata)
                
                processing_output.append(html.Div(f"Extracted {len(metadata)} images from {os.path.basename(pdf_path)}"))
            except Exception as e:
                processing_output.append(html.Div(f"Error processing {os.path.basename(pdf_path)}: {str(e)}", style={'color': 'red'}))
        
        if all_metadata:
            # Build embeddings
            try:
                build_image_embeddings(output_dir, all_metadata)
                processing_output.append(html.Div(f"Built embeddings for {len(all_metadata)} images.", style={'color': 'green'}))
            except Exception as e:
                processing_output.append(html.Div(f"Error building embeddings: {str(e)}", style={'color': 'red'}))
        
        return html.Div(processing_output), all_metadata
    
    return html.Div("No action taken."), []

@app.callback(
    Output('search-results', 'children'),
    Input('search-button', 'n_clicks'),
    [State('search-input', 'value'),
     State('use-post-filtering', 'value'),
     State('use-aggressive-filtering', 'value'),
     State('min-score-slider', 'value')],
    prevent_initial_call=True
)
def search_and_display(n_clicks, query, use_post_filtering, use_aggressive_filtering, min_score):
    if not query:
        return html.Div("Please enter a search query.")
    
    try:
        # Search for images
        results = search_images(query, top_k=10, use_post_filtering=use_post_filtering, aggressive_filtering=use_aggressive_filtering)
        
        if not results or not results["ids"] or not results["ids"][0]:
            return html.Div("No results found.")
        
        # Display results
        result_cards = []
        
        for i in range(len(results["ids"][0])):
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            score = 1 - distance
            
            # Skip results below minimum score
            if score < min_score:
                continue
            
            # Get image path
            image_path = metadata.get('image_path', '')
            if not os.path.exists(image_path):
                continue
            
            # Read image
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            # Encode image
            encoded_image = base64.b64encode(image_data).decode('ascii')
            src = f'data:image/png;base64,{encoded_image}'
            
            # Create card
            card = dbc.Card([
                dbc.CardImg(src=src, top=True),
                dbc.CardBody([
                    html.H5(f"PDF: {metadata.get('pdf_name', 'unknown')}, Page: {metadata.get('page_number', 'unknown')}"),
                    html.P(f"Score: {score:.4f}"),
                    html.P(f"Size: {metadata.get('width', 0)}x{metadata.get('height', 0)}"),
                    html.P(f"Caption: {metadata.get('caption', '')}")
                ])
            ], style={'margin': '10px'})
            
            result_cards.append(dbc.Col(card, width=4))
        
        if not result_cards:
            return html.Div("No results found that meet the minimum score threshold.")
        
        # Arrange cards in rows of 3
        rows = []
        for i in range(0, len(result_cards), 3):
            row = dbc.Row(result_cards[i:i+3], style={'margin': '10px 0'})
            rows.append(row)
        
        return html.Div(rows)
    
    except Exception as e:
        return html.Div(f"Error during search: {str(e)}")

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')
