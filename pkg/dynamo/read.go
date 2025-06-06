package dynamo

import (
	"context"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/feature/dynamodb/attributevalue"
	"github.com/aws/aws-sdk-go-v2/service/dynamodb"
	"github.com/vignesh-goutham/artemis/pkg/types"
)

// GetAllTrades retrieves all trades from DynamoDB
func GetAllTrades(ctx context.Context) ([]types.Trade, error) {
	result, err := GetClient().Scan(ctx, &dynamodb.ScanInput{
		TableName: aws.String(GetTableName()),
	})
	if err != nil {
		return nil, err
	}

	var trades []types.Trade
	err = attributevalue.UnmarshalListOfMaps(result.Items, &trades)
	return trades, err
}
