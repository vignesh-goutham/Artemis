package dynamo

import (
	"context"
	"fmt"
	"log"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/feature/dynamodb/attributevalue"
	"github.com/aws/aws-sdk-go-v2/service/dynamodb"
	dynamotypes "github.com/aws/aws-sdk-go-v2/service/dynamodb/types"
	"github.com/vignesh-goutham/artemis/pkg/types"
)

// BatchWriteTrades performs a batch write operation for trades
func BatchWriteTrades(ctx context.Context, trades []types.Trade, deletes []string) error {
	if len(trades) == 0 && len(deletes) == 0 {
		return nil
	}

	// Prepare write requests
	var writeRequests []dynamotypes.WriteRequest
	tableName := GetTableName()

	// Add updates
	for _, trade := range trades {
		item, err := attributevalue.MarshalMap(trade)
		if err != nil {
			return fmt.Errorf("error marshaling trade: %v", err)
		}

		writeRequests = append(writeRequests, dynamotypes.WriteRequest{
			PutRequest: &dynamotypes.PutRequest{
				Item: item,
			},
		})
	}

	// Add deletes
	for _, ticker := range deletes {
		writeRequests = append(writeRequests, dynamotypes.WriteRequest{
			DeleteRequest: &dynamotypes.DeleteRequest{
				Key: map[string]dynamotypes.AttributeValue{
					"ticker": &dynamotypes.AttributeValueMemberS{Value: ticker},
				},
			},
		})
	}

	// Split into chunks of 25 (DynamoDB batch write limit)
	chunkSize := 25
	for i := 0; i < len(writeRequests); i += chunkSize {
		end := i + chunkSize
		if end > len(writeRequests) {
			end = len(writeRequests)
		}

		chunk := writeRequests[i:end]
		_, err := GetClient().BatchWriteItem(ctx, &dynamodb.BatchWriteItemInput{
			RequestItems: map[string][]dynamotypes.WriteRequest{
				tableName: chunk,
			},
		})
		if err != nil {
			return fmt.Errorf("error batch writing items: %v", err)
		}
	}

	log.Printf("Successfully batch wrote %d trades and deleted %d trades", len(trades), len(deletes))
	return nil
}

// AddTrade adds a single trade to DynamoDB
func AddTrade(ctx context.Context, trade types.Trade) error {
	item, err := attributevalue.MarshalMap(trade)
	if err != nil {
		return fmt.Errorf("failed to marshal trade: %v", err)
	}

	_, err = GetClient().PutItem(ctx, &dynamodb.PutItemInput{
		TableName: aws.String(GetTableName()),
		Item:      item,
	})
	if err != nil {
		return fmt.Errorf("failed to put item in DynamoDB: %v", err)
	}

	log.Printf("Successfully added trade for %s", trade.Ticker)
	return nil
}
