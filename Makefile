.PHONY: test doctor bench eval build run

test:
	python -m unittest discover -s tests

doctor:
	python -m draftfly doctor

bench:
	python -m draftfly bench

eval:
	python -m draftfly eval

build:
	docker build -t draftfly .

run:
	python -m draftfly serve
