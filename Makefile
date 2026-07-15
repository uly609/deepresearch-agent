.PHONY: run run-offline run-llm run-full resume eval eval-live eval-llm check clean legacy-java-test

run:
	python3 -m deepresearch --engine langgraph

run-offline:
	python3 -m deepresearch --engine langgraph --offline-tools

run-llm:
	python3 -m deepresearch --engine langgraph --llm

run-full:
	python3 -m deepresearch --engine langgraph --llm

resume:
	python3 -m deepresearch --engine langgraph --resume $(RUN_ID)

eval:
	python3 -m deepresearch --engine langgraph --eval

eval-live:
	python3 -m deepresearch --engine langgraph --eval --live-tools

eval-llm:
	python3 -m deepresearch --engine langgraph --eval --llm

check:
	PYTHONPYCACHEPREFIX=.pycache python3 -m compileall deepresearch

clean:
	rm -rf .deepresearch .pycache deepresearch/__pycache__ deepresearch/*/__pycache__

legacy-java-test:
	$(MAKE) -C _legacy/java_mvp test
