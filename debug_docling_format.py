try:
    from docling.document_converter import PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    import inspect
    print(inspect.signature(PdfFormatOption.__init__))
except ImportError as e:
    print(e)
