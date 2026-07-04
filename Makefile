.PHONY: install test bench clean
install:      ## install pinned dependencies
	pip install -r requirements.txt
test:         ## run the test suite
	pytest -q
bench:        ## regenerate every table and plot in results/
	python benchmarks/run_benchmark.py
clean:        ## remove caches (keeps committed results/)
	rm -rf __pycache__ */__pycache__ */*/__pycache__ .pytest_cache
