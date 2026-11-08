PY ?= python
export PYTHONPATH := src

.PHONY: install demo run test app report score clean

install:
	$(PY) -m pip install -r requirements.txt

demo:
	$(PY) -m reviews_nlp.pipeline --demo --skip-embeddings --llm-provider mock

run:
	$(PY) -m reviews_nlp.pipeline --sample-size 20000

test:
	$(PY) -m pytest -q

app:
	streamlit run app/streamlit_app.py

report:
	quarto render report/report.qmd

score:
	$(PY) -m reviews_nlp.evaluate_llm

clean:
	rm -rf outputs/*.csv outputs/*.json outputs/*.joblib outputs/figures/*.png data/*.db
