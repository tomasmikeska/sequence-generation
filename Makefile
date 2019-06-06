PYTHON=python

lint:
	pycodestyle ./src

train:
	$(PYTHON) src/train.py

sample:
	$(PYTHON) src/sample.py
