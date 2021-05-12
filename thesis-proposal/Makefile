pdf:
	pandoc \
		--filter pandoc-crossref \
		--filter pandoc-citeproc \
		--template=./templates/template.latex \
		./content/thesis-proposals.md \
		./content/metadata.yml \
		--variable=fontsize:12pt \
		--variable=papersize:a4paper \
		--variable=documentclass:article \
		--output thesis.pdf \
		--resource-path=./content

clean:
	rm tex2pdf* -r