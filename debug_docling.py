import inspect
try:
    from docling.datamodel.pipeline_options import PipelineOptions, PdfPipelineOptions
    print("PipelineOptions fields:", [f for f in dir(PipelineOptions) if not f.startswith('_')])
    print("PdfPipelineOptions fields:", [f for f in dir(PdfPipelineOptions) if not f.startswith('_')])
except ImportError as e:
    print(e)
