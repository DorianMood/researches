pdf:
	pandoc  \
		--pdf-engine=xelatex \
		--filter=pandoc-crossref \
		--filter=pandoc-citeproc \
		--include-in-header=templates/preamble.latex \
		--output thesis.pdf \
		--template=templates/template.latex \
		--variable=fontsize:12pt \
		--variable=papersize:a4paper \
		--variable=documentclass:report \
		content/thesis-proposals.md \
		content/metadata.yml \
		--bibliography=content/index.bib \
		--csl=content/ref.csl \
		--number-sections \
		--resource-path=content

clean:
	rm tex2pdf* -r