.PHONY: clean build package deploy

# Build the Go function into bootstrap
build:
	mkdir -p _bin
	GOOS=linux GOARCH=amd64 go build -o _bin/bootstrap main.go

# Clean up old files
clean:
	rm -rf _bin/

# Package the function
package: clean build
	zip _bin/function.zip _bin/bootstrap

# Deploy the function to AWS Lambda
deploy: package
	aws lambda update-function-code \
		--function-name stbn-trading-bot \
		--zip-file fileb://_bin/function.zip \
		--profile personal \
		--region us-east-2

# Default target
all: package 