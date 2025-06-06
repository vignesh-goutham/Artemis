.PHONY: setup clean run build package deploy

# Python virtual environment directory
VENV_DIR := _bin/venv
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip

# Go binary targets
build:
	mkdir -p _bin
	GOOS=linux GOARCH=amd64 go build -o _bin/bootstrap main.go

package: clean build
	zip _bin/function.zip _bin/bootstrap

deploy: package
	aws lambda update-function-code \
		--function-name stbn-trading-bot \
		--zip-file fileb://_bin/function.zip \
		--profile personal \
		--region us-east-2

# Python backtester targets
setup: $(VENV_DIR)
	$(PIP) install -r backtest/requirements.txt
	chmod +x backtest/run_backtest.sh

$(VENV_DIR):
	python3 -m venv $(VENV_DIR)

clean:
	rm -rf _bin/
	rm -rf $(VENV_DIR)

run: setup
	./backtest/run_backtest.sh $(ARGS)

# Example usage:
# make run ARGS="--input trades.csv --initial_balance 10000 --detailed"
# make deploy  # for deploying Go function 