package dynamo

import (
	"context"
	"log"
	"os"

	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/dynamodb"
)

var (
	client *dynamodb.Client
)

// Initialize sets up the DynamoDB client
func Initialize() error {
	cfg, err := config.LoadDefaultConfig(context.TODO())
	if err != nil {
		return err
	}
	client = dynamodb.NewFromConfig(cfg)
	return nil
}

// InitializeWithRegion sets up the DynamoDB client with a specific region
func InitializeWithRegion(region string) error {
	cfg, err := config.LoadDefaultConfig(context.TODO(),
		config.WithRegion(region),
	)
	if err != nil {
		return err
	}
	client = dynamodb.NewFromConfig(cfg)
	return nil
}

// GetClient returns the DynamoDB client instance
func GetClient() *dynamodb.Client {
	if client == nil {
		log.Fatal("DynamoDB client not initialized")
	}
	return client
}

// GetTableName returns the DynamoDB table name
func GetTableName() string {
	tableName := os.Getenv("DYNAMODB_TABLE")
	if tableName == "" {
		tableName = "trading_strategy"
	}
	return tableName
}
