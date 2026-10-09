# Reproduce the paper.
#
#   make setup      install the pinned dependencies and this package
#   make notebooks  run notebooks 01-10 in order: rebuild every table and
#                   check every number in the paper (about 30 min, CPU only)
#   make tables     the same tables without notebooks (scripts/run_all.py)
#   make verify     check every number in the paper against results/tables/
#   make test       run the tests
#   make clean      remove executed notebook copies (never touches data/)

PYTHON   ?= python3
JUPYTER  ?= jupyter
EXEC_DIR := results/logs/executed

.PHONY: setup notebooks tables verify test clean

setup:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

notebooks:
	@mkdir -p $(EXEC_DIR)
	@for nb in notebooks/*.ipynb; do \
		echo "--- $$nb"; \
		$(JUPYTER) nbconvert --to notebook --execute --ExecutePreprocessor.timeout=3600 \
			--output-dir=$(EXEC_DIR) "$$nb" > /dev/null || exit 1; \
	done
	@echo "All notebooks ran. Copies with their outputs are in $(EXEC_DIR)/."

tables:
	$(PYTHON) scripts/run_all.py

verify:
	$(PYTHON) scripts/verify_paper_numbers.py --log results/logs/verification.md

test:
	$(PYTHON) -m pytest -q

clean:
	rm -rf $(EXEC_DIR)
	find . -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
