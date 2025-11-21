FROM docker.io/texlive/texlive:latest
RUN tlmgr install latexmk amsmath collection-fontsrecommended
WORKDIR /work
ENTRYPOINT ["latexmk", "-pdf", "-interaction=nonstopmode", "-output-directory=/build", "main.tex"]
