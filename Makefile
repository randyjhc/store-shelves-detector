MAIN_FILE = shelf_detection.py

install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	python -m pytest -vv --cov=test $(MAIN_FILE)

format:
	black *.py

lint:
	pylint --disable=R,C lib/*.py $(MAIN_FILE)

run:
	python $(MAIN_FILE)

clean:
	rm -rf 01_datasets
	rm -rf 02_preprocess

clean_train:
	rm -rf runs
