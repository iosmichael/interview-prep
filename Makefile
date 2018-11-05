all:
	pandoc discrete_optimization_midterm_study_guide.md --pdf-engine=pdflatex -o discrete_optimization_midterm_study_guide.pdf

lr:
	pandoc linear_regression.md --pdf-engine=pdflatex -o linear_regression.pdf

mc:
	pandoc understanding_of_MCMC.md --pdf-engine=pdflatex -o mcmc_explain.pdf

ml:
	pandoc machine_learning_cheatsheet.md --pdf-engine=pdflatex -o machine_learning.pdf
