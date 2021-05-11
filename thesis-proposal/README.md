# Thesis proposals

Here I need:

- [ ] Compare scene graph and JPEG sizes.
- [x] Prepare different application variants and scenarios, how and where exactly we can use it.
- [x] Application niche and scenario, motivation.
- [x] Scene graph generation for image compression (take a look on existing papers)
- [x] Draw figures
- [x] Write thesis proposals and literature survey.
- [ ] Set up latex templates for pandoc in BUAA format
- [ ] Implement scene graph generation
- [ ] Prepare slides

## Compile 
`pandoc --filter pandoc-crossref --filter pandoc-citeproc --template='<template.latex>' '<input.md>' -o '<output.pdf>'`