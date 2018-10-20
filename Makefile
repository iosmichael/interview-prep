all:
	pandoc understanding_of_MCMC.md --pdf-engine=pdflatex -o mcmc_explain.pdf

ml:
	pandoc machine_learning_cheatsheet.md --pdf-engine=pdflatex -o machine_learning.pdf
