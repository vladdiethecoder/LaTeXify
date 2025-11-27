import inspect
try:
    from docling.document_converter import DocumentConverter
    print(inspect.signature(DocumentConverter.__init__))
except ImportError as e:
    print(e)
