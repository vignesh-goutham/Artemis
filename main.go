package main

import (
	"context"
	"log"

	"github.com/aws/aws-lambda-go/lambda"
	"github.com/vignesh-goutham/artemis/pkg/alpaca"
	"github.com/vignesh-goutham/artemis/pkg/dynamo"
	"github.com/vignesh-goutham/artemis/pkg/stocks"
)

func init() {
	// Initialize AWS clients
	if err := dynamo.Initialize(); err != nil {
		log.Fatal(err)
	}

	// Initialize Alpaca clients
	if err := alpaca.Initialize(); err != nil {
		log.Fatal(err)
	}
}

func handler(ctx context.Context) (map[string]interface{}, error) {
	return stocks.Handler(ctx)
}

func main() {
	lambda.Start(handler)
}
