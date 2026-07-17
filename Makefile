.PHONY : run-checks
run-checks :
	isort --check .
	black --check .
	ruff check .
	# mypy .

.PHONY : format
format :
	isort .
	black .

.PHONY : build
build :
	rm -rf *.egg-info/
	python -m build

.PHONY : world-model-ci
world-model-ci :
	/usr/bin/python3 scripts/tools/world_model_ci_check.py --python /usr/bin/python3

.PHONY : world-model-tests
world-model-tests :
	PYTHONPATH=scripts /usr/bin/python3 -m unittest discover -s scripts/sonic_world/tests -p 'test_*.py'
